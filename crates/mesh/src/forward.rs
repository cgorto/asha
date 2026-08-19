//! Mesh forward shading over the depth prepass's compacted clusters.
//!
//! `Equal + Read` depth shades each surviving fragment once.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_mesh::{ClusterInstance, IndirectData, MeshFrameData, MeshShadeLighting};
use gpu::{
    CommandBuffer, CompareOp, DepthFlags, DepthState, Gpu, HazardFlags, HeapSlots, LoadOp,
    RenderAttachment, RenderPassDesc, SamplerSlot, ShaderTypeGraphics, Stage, StoreOp,
};

use crate::{FrameAlloc, MeshInstances, MeshRasterView, MeshScene, mesh_frame_data};

#[derive(Clone, Copy)]
pub struct MeshForwardTargets {
    pub color: gpu::Texture,
    pub depth: gpu::Texture,
    pub size: UVec2,
    pub color_load_op: LoadOp,
    pub clear_color: [f32; 4],
}

/// Deferred-light MRT written by mesh forward at fragment outputs 1..3.
/// Formats are RG16F normal, RGBA16F albedo, and R32F material identity + 1.
#[derive(Clone, Copy)]
pub struct MeshForwardSurfaceTargets {
    pub normal: gpu::Texture,
    pub albedo: gpu::Texture,
    pub material: gpu::Texture,
}

/// Optional grid light field shared by forward shading and linework.
/// A null pointer disables the field.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshLightField {
    pub cells: GpuPtr<f32>,
    pub dims: [u32; 2],
    pub cell_size: f32,
    pub gate: f32,
}

impl MeshLightField {
    pub(crate) fn assert_valid(self) {
        assert!(
            self.gate.is_finite() && (0.0..=1.0).contains(&self.gate),
            "light-field gate must be finite and in 0..=1"
        );
        if !self.cells.is_null() {
            assert!(
                self.dims[0] > 0 && self.dims[1] > 0,
                "non-null light field needs positive dimensions"
            );
            assert!(
                u64::from(self.dims[0]) * u64::from(self.dims[1]) <= i32::MAX as u64,
                "light field must fit the shared signed cell index"
            );
            assert!(
                self.cell_size.is_finite() && self.cell_size > 0.0,
                "non-null light field needs a finite positive cell size"
            );
        }
    }
}

/// Contiguous batch range rendered by a replacement shader group.
/// The range indexes the shared indirect-command table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderGroupSlice {
    pub group: u32,
    pub batch_base: u32,
    pub batch_count: u32,
}

/// Contiguous batch range rendered additively after forward shading.
/// Surface writes are masked and `Equal + Read` depth is preserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderCoatSlice {
    pub group: u32,
    pub batch_base: u32,
    pub batch_count: u32,
}

/// Whether a shader group replaces forward shading or adds a coat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderGroupKind {
    Replace,
    Coat,
}

/// Registered shader pair. Custom vertex shaders must preserve prepass
/// positions; otherwise `Equal` depth rejects their fragments.
struct GroupShaders {
    vert: Option<gpu::Shader>,
    frag: gpu::Shader,
    kind: ShaderGroupKind,
}

/// One grouped multi-draw and its indirect-table window.
struct ShaderRun {
    vert: gpu::Shader,
    frag: gpu::Shader,
    args: gpu::Ptr<IndirectData>,
    count: gpu::Ptr<u32>,
}

pub struct MeshForwardPass {
    vert_shader: gpu::Shader,
    frag_shader: gpu::Shader,
    /// Registered shader groups, indexed by host-assigned group ID.
    groups: Vec<GroupShaders>,
    /// Rotating grouped-draw counts for standard, replacement, and coat runs.
    group_counts: Option<gpu::Ptr<u32>>,
    max_groups: u32,
    in_flight: usize,
}

impl MeshForwardPass {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            vert_shader: gpu.shader_create(
                &asha_assets::load_spv("mesh_vert"),
                ShaderTypeGraphics::Vertex,
                "mesh_vert",
            ),
            frag_shader: gpu.shader_create(
                &asha_assets::load_spv("mesh_frag"),
                ShaderTypeGraphics::Fragment,
                "mesh_frag",
            ),
            groups: Vec::new(),
            group_counts: None,
            max_groups: 0,
            in_flight: 0,
        }
    }

    /// Creates a forward pass with grouped-shader storage.
    pub fn with_groups(gpu: &Gpu, max_groups: u32, in_flight: usize) -> Self {
        assert!(
            max_groups > 0,
            "grouped forward needs nonzero group capacity"
        );
        assert!(in_flight >= 1);
        Self {
            group_counts: Some(gpu.alloc_slice::<u32>(
                u64::from(1 + 2 * max_groups) * in_flight as u64,
                gpu::Memory::Default,
            )),
            max_groups,
            in_flight,
            ..Self::new(gpu)
        }
    }

    /// Registers a shader group in host-assigned order.
    /// Asset names refer to `.spv` files; `None` uses `mesh_vert`.
    pub fn register_group(
        &mut self,
        gpu: &Gpu,
        index: u32,
        vert: Option<&str>,
        frag: &str,
        kind: ShaderGroupKind,
    ) {
        assert!(
            self.group_counts.is_some(),
            "shader groups need MeshForwardPass::with_groups"
        );
        assert!(
            index == self.groups.len() as u32,
            "shader group index authority violated: registering {index}, expected {}",
            self.groups.len()
        );
        assert!(
            index < self.max_groups,
            "shader group {index} exceeds capacity {}",
            self.max_groups
        );
        self.groups.push(GroupShaders {
            vert: vert.map(|name| {
                gpu.shader_create(
                    &asha_assets::load_spv(name),
                    ShaderTypeGraphics::Vertex,
                    name,
                )
            }),
            frag: gpu.shader_create(
                &asha_assets::load_spv(frag),
                ShaderTypeGraphics::Fragment,
                frag,
            ),
            kind,
        });
    }

    pub fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.vert_shader);
        gpu.shader_destroy(self.frag_shader);
        for group in self.groups {
            if let Some(vert) = group.vert {
                gpu.shader_destroy(vert);
            }
            gpu.shader_destroy(group.frag);
        }
        if let Some(counts) = self.group_counts {
            gpu.free(counts);
        }
    }

    /// Shades the depth prepass's compacted clusters.
    /// `draw_count` counts candidate batches, including empty commands.
    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        culled_args: gpu::Ptr<IndirectData>,
        clusters: GpuPtr<ClusterInstance>,
        draw_count: gpu::Ptr<u32>,
        targets: MeshForwardTargets,
        view: MeshRasterView,
        eye: abi_core::glam::Vec3,
        lighting: MeshShadeLighting,
        ramp_default_sampler: SamplerSlot,
    ) {
        self.record_with_light_field(
            gpu,
            cb,
            fa,
            heap,
            scene,
            instances,
            culled_args,
            clusters,
            draw_count,
            targets,
            view,
            eye,
            lighting,
            ramp_default_sampler,
            MeshLightField::default(),
        );
    }

    /// Records forward shading with an optional light-field rim gate.
    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    pub fn record_with_light_field(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        culled_args: gpu::Ptr<IndirectData>,
        clusters: GpuPtr<ClusterInstance>,
        draw_count: gpu::Ptr<u32>,
        targets: MeshForwardTargets,
        view: MeshRasterView,
        eye: abi_core::glam::Vec3,
        lighting: MeshShadeLighting,
        ramp_default_sampler: SamplerSlot,
        light_field: MeshLightField,
    ) {
        self.record_inner(
            gpu,
            cb,
            fa,
            heap,
            scene,
            instances,
            culled_args,
            clusters,
            draw_count,
            targets,
            view,
            eye,
            lighting,
            ramp_default_sampler,
            light_field,
            0.0,
            None,
            None,
            &[],
        );
    }

    /// Records forward shading with deferred-light surface attachments.
    /// Raster behavior remains `Equal + Read` against completed depth.
    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    pub fn record_with_surfaces(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        culled_args: gpu::Ptr<IndirectData>,
        clusters: GpuPtr<ClusterInstance>,
        draw_count: gpu::Ptr<u32>,
        targets: MeshForwardTargets,
        view: MeshRasterView,
        eye: abi_core::glam::Vec3,
        lighting: MeshShadeLighting,
        ramp_default_sampler: SamplerSlot,
        light_field: MeshLightField,
        surfaces: MeshForwardSurfaceTargets,
    ) {
        self.record_inner(
            gpu,
            cb,
            fa,
            heap,
            scene,
            instances,
            culled_args,
            clusters,
            draw_count,
            targets,
            view,
            eye,
            lighting,
            ramp_default_sampler,
            light_field,
            0.0,
            Some(surfaces),
            None,
            &[],
        );
    }

    /// Records grouped forward and additive coat runs.
    /// Replacement slices cover the batch suffix and name `Replace` groups.
    /// Coats name `Coat` groups and use optional ascending, disjoint windows.
    /// The count ring follows frame ownership; `time` reaches shaders.
    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    pub fn record_grouped(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        culled_args: gpu::Ptr<IndirectData>,
        clusters: GpuPtr<ClusterInstance>,
        standard_batches: u32,
        slices: &[ShaderGroupSlice],
        coats: &[ShaderCoatSlice],
        counter_slot: usize,
        time: f32,
        targets: MeshForwardTargets,
        view: MeshRasterView,
        eye: abi_core::glam::Vec3,
        lighting: MeshShadeLighting,
        ramp_default_sampler: SamplerSlot,
        light_field: MeshLightField,
        surfaces: MeshForwardSurfaceTargets,
    ) {
        let counts = self
            .group_counts
            .as_ref()
            .expect("record_grouped needs MeshForwardPass::with_groups");
        assert!(counter_slot < self.in_flight);
        assert!(
            slices.len() as u32 <= self.max_groups,
            "more group slices than group capacity"
        );
        assert!(
            coats.len() as u32 <= self.max_groups,
            "more coat slices than group capacity"
        );
        let mut expected_base = standard_batches;
        for slice in slices {
            assert!(
                (slice.group as usize) < self.groups.len(),
                "group slice names unregistered shader group {}",
                slice.group
            );
            assert!(
                self.groups[slice.group as usize].kind == ShaderGroupKind::Replace,
                "forward slice names coat group {}",
                slice.group
            );
            assert!(
                slice.batch_base == expected_base,
                "group {} slice does not begin at the prior run's end",
                slice.group
            );
            expected_base = expected_base
                .checked_add(slice.batch_count)
                .expect("group partition exceeds u32");
        }
        assert!(
            expected_base == instances.batch_count(),
            "group partition does not cover the batch table"
        );
        // Coat windows are ascending, disjoint, and in range.
        let mut coat_watermark = 0u32;
        for coat in coats {
            assert!(
                (coat.group as usize) < self.groups.len(),
                "coat slice names unregistered shader group {}",
                coat.group
            );
            assert!(
                self.groups[coat.group as usize].kind == ShaderGroupKind::Coat,
                "coat slice names forward group {}",
                coat.group
            );
            assert!(
                coat.batch_base >= coat_watermark,
                "coat slices must be ascending and disjoint"
            );
            coat_watermark = coat
                .batch_base
                .checked_add(coat.batch_count)
                .expect("coat window exceeds u32");
            assert!(
                coat_watermark <= instances.batch_count(),
                "coat window exceeds the batch table"
            );
        }

        // Candidate counts are CPU-authored; empty batches remain valid commands.
        let ring_base = counter_slot * (1 + 2 * self.max_groups as usize);
        // SAFETY: the frame ownership gate makes this ring slot GPU-idle;
        // its run is exactly `1 + 2 * max_groups` counters.
        unsafe {
            *counts.cpu.add(ring_base) = standard_batches;
            for (i, slice) in slices.iter().enumerate() {
                *counts.cpu.add(ring_base + 1 + i) = slice.batch_count;
            }
            for (i, coat) in coats.iter().enumerate() {
                *counts.cpu.add(ring_base + 1 + self.max_groups as usize + i) = coat.batch_count;
            }
        }
        let count_ptr = |i: usize| {
            gpu.mem_suballoc(
                counts.cast(),
                ((ring_base + i) * core::mem::size_of::<u32>()) as i64,
                core::mem::size_of::<u32>() as u64,
                1,
            )
            .cast::<u32>()
        };
        let args_at = |batch_base: u32| {
            gpu.mem_suballoc(
                culled_args.cast(),
                (batch_base as usize * core::mem::size_of::<IndirectData>()) as i64,
                core::mem::size_of::<IndirectData>() as u64,
                u64::from(instances.batch_count() - batch_base),
            )
            .cast::<IndirectData>()
        };

        let mut runs: Vec<ShaderRun> = Vec::with_capacity(1 + slices.len());
        if standard_batches > 0 {
            runs.push(ShaderRun {
                vert: self.vert_shader,
                frag: self.frag_shader,
                args: args_at(0),
                count: count_ptr(0),
            });
        }
        for (i, slice) in slices.iter().enumerate() {
            if slice.batch_count == 0 {
                continue;
            }
            let shaders = &self.groups[slice.group as usize];
            runs.push(ShaderRun {
                vert: shaders.vert.unwrap_or(self.vert_shader),
                frag: shaders.frag,
                args: args_at(slice.batch_base),
                count: count_ptr(1 + i),
            });
        }
        let mut coat_runs: Vec<ShaderRun> = Vec::with_capacity(coats.len());
        for (i, coat) in coats.iter().enumerate() {
            if coat.batch_count == 0 {
                continue;
            }
            let shaders = &self.groups[coat.group as usize];
            coat_runs.push(ShaderRun {
                vert: shaders.vert.unwrap_or(self.vert_shader),
                frag: shaders.frag,
                args: args_at(coat.batch_base),
                count: count_ptr(1 + self.max_groups as usize + i),
            });
        }

        self.record_inner(
            gpu,
            cb,
            fa,
            heap,
            scene,
            instances,
            culled_args,
            clusters,
            // Retained for the shared ungrouped argument shape.
            count_ptr(0),
            targets,
            view,
            eye,
            lighting,
            ramp_default_sampler,
            light_field,
            time,
            Some(surfaces),
            Some(&runs),
            &coat_runs,
        );
    }

    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    fn record_inner(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        culled_args: gpu::Ptr<IndirectData>,
        clusters: GpuPtr<ClusterInstance>,
        draw_count: gpu::Ptr<u32>,
        targets: MeshForwardTargets,
        view: MeshRasterView,
        eye: abi_core::glam::Vec3,
        lighting: MeshShadeLighting,
        ramp_default_sampler: SamplerSlot,
        light_field: MeshLightField,
        time: f32,
        surfaces: Option<MeshForwardSurfaceTargets>,
        runs: Option<&[ShaderRun]>,
        coat_runs: &[ShaderRun],
    ) {
        assert!(targets.size.x > 0 && targets.size.y > 0);
        assert!(eye.is_finite(), "mesh forward eye must be finite");
        assert!(time.is_finite(), "mesh forward time must be finite");
        assert_ne!(
            ramp_default_sampler.index(),
            0,
            "mesh forward needs a real default ramp sampler"
        );
        light_field.assert_valid();
        if let Some(surface) = surfaces {
            for (texture, format, label) in [
                (surface.normal, gpu::TextureFormat::Rg16Float, "normal"),
                (surface.albedo, gpu::TextureFormat::Rgba16Float, "albedo"),
                (surface.material, gpu::TextureFormat::R32Float, "material"),
            ] {
                assert_eq!(
                    &texture.dimensions[..2],
                    targets.size.as_ref(),
                    "mesh surface {label} size must match the forward target"
                );
                assert_eq!(
                    texture.format, format,
                    "mesh surface {label} format does not match the contract"
                );
            }
        }

        let frame = fa.frame_alloc(MeshFrameData {
            lighting,
            light_field: light_field.cells,
            light_field_dims: light_field.dims,
            light_field_cell_size: light_field.cell_size,
            light_field_gate: light_field.gate,
            time,
            eye: eye.to_array(),
            ramp_default_sampler: ramp_default_sampler.index(),
            ..mesh_frame_data(scene, instances, clusters, view)
        });

        // Slot 0 is HDR; slots 1..3 are optional cleared surface attachments.
        // Clearing prevents stale material identities on empty frames.
        let mut color_attachments = [RenderAttachment {
            texture: targets.color,
            load_op: targets.color_load_op,
            store_op: StoreOp::Store,
            clear_color: targets.clear_color,
            ..Default::default()
        }; 4];
        let color_attachment_count = if let Some(surfaces) = surfaces {
            let surface = |texture| RenderAttachment {
                texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [0.0; 4],
                ..Default::default()
            };
            color_attachments[1] = surface(surfaces.normal);
            color_attachments[2] = surface(surfaces.albedo);
            color_attachments[3] = surface(surfaces.material);
            4
        } else {
            1
        };

        heap.bind(gpu, cb);
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                render_area_size: targets.size.to_array(),
                color_attachments: &color_attachments[..color_attachment_count],
                depth_attachment: Some(RenderAttachment {
                    texture: targets.depth,
                    // The prepass owns the reverse-Z depth clear.
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4],
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        // Equal + Read shades exactly the fragments surviving the prepass.
        gpu.cmd_set_depth_state(
            cb,
            DepthState {
                mode: DepthFlags::READ,
                compare: CompareOp::Equal,
                ..Default::default()
            },
        );
        // Model-space CCW remains front-facing after parity mirrors.
        gpu.cmd_set_cull_mode(cb, true);
        gpu.cmd_set_front_face(cb, false);

        // Matching indirect and pushed-data windows preserve per-draw indices.
        let standard = [ShaderRun {
            vert: self.vert_shader,
            frag: self.frag_shader,
            args: culled_args,
            count: draw_count,
        }];
        for run in runs.unwrap_or(&standard) {
            gpu.cmd_set_shaders(cb, run.vert, run.frag);
            gpu.cmd_draw_instanced_indirect_multi(
                cb,
                frame.cast(),
                frame.cast(),
                run.args.cast(),
                core::mem::size_of::<abi_mesh::IndirectData>() as u32,
                run.count.cast(),
            );
        }

        // Coats add onto shaded fragments; surface MRT writes stay masked.
        if !coat_runs.is_empty() {
            let additive = gpu::BlendState {
                enable: true,
                color_op: gpu::BlendOp::Add,
                src_color_factor: gpu::BlendFactor::One,
                dst_color_factor: gpu::BlendFactor::One,
                alpha_op: gpu::BlendOp::Add,
                src_alpha_factor: gpu::BlendFactor::One,
                dst_alpha_factor: gpu::BlendFactor::One,
                color_write_mask: 0xf,
            };
            let masked = gpu::BlendState {
                color_write_mask: 0,
                ..Default::default()
            };
            let states = [additive, masked, masked, masked];
            gpu.cmd_set_blend_states(cb, &states[..color_attachment_count]);
            for run in coat_runs {
                gpu.cmd_set_shaders(cb, run.vert, run.frag);
                gpu.cmd_draw_instanced_indirect_multi(
                    cb,
                    frame.cast(),
                    frame.cast(),
                    run.args.cast(),
                    core::mem::size_of::<abi_mesh::IndirectData>() as u32,
                    run.count.cast(),
                );
            }
        }

        gpu.cmd_end_render_pass(cb);
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::FragmentShader,
            HazardFlags::COLOR_ATTACHMENT,
        );
    }
}
