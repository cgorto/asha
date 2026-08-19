//! Local lighting consumes the mesh forward pass's surface MRTs.
//!
//! The forward pass writes normal `Rg16Float`, albedo `Rgba16Float`, and
//! material `R32Float` images into stable sampled slots. Shadow producers run
//! before this pass; this pass then reconstructs world position, evaluates
//! lights in array order, and accumulates into HDR. Its final barrier exposes
//! HDR to later fragment consumers.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_light::PointLight;
use abi_light::{LocalLightData, MeshShadowData, SURFACE_MATERIAL_INDEX_MAX};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    CommandBuffer, Gpu, HazardFlags, HeapSlots, Memory, OwnedTexture, Queue, SampledSlot,
    SamplerSlot, Stage, StorageSlot, TextureDesc, TextureFormat, TextureViewDesc, UsageFlags,
};
use mesh::{
    MeshForwardSurfaceTargets, MeshInstances, MeshLightField, MeshRasterView, MeshScene,
    ShadowTlasBuilder, ShadowTlasStats,
};

/// `mesh_local_light` workgroup width and height.
const LOCAL_LIGHT_GROUP_SIZE: u32 = 8;

/// Forward surface MRTs and their stable sampled slots.
///
/// Formats are normal `Rg16Float`, albedo `Rgba16Float`, and material
/// `R32Float`; the material image carries the exact f32 marker.
pub struct MeshSurfaceTargets {
    size: UVec2,
    normal: OwnedTexture,
    albedo: OwnedTexture,
    material: OwnedTexture,
    normal_slot: SampledSlot,
    albedo_slot: SampledSlot,
    material_slot: SampledSlot,
}

impl MeshSurfaceTargets {
    /// Rebuilds targets when the physical extent changes.
    pub fn ensure(this: &mut Option<Self>, gpu: &Gpu, heap: &mut HeapSlots, size: UVec2) -> bool {
        assert!(size.x > 0 && size.y > 0);
        if this.as_ref().is_some_and(|t| t.size == size) {
            return false;
        }
        let [normal_slot, albedo_slot, material_slot] = match this.take() {
            Some(old) => {
                gpu.queue_wait_idle(Queue::Main);
                let slots = [old.normal_slot, old.albedo_slot, old.material_slot];
                old.free(gpu);
                slots
            }
            None => [
                heap.alloc_sampled(),
                heap.alloc_sampled(),
                heap.alloc_sampled(),
            ],
        };

        let create = |format| {
            gpu.texture_alloc_and_create(
                TextureDesc {
                    dimensions: [size.x, size.y, 1],
                    format,
                    usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::SAMPLED,
                    ..Default::default()
                },
                Queue::Main,
                None,
            )
        };
        let normal = create(TextureFormat::Rg16Float);
        let albedo = create(TextureFormat::Rgba16Float);
        let material = create(TextureFormat::R32Float);
        for (slot, owned) in [
            (normal_slot, normal),
            (albedo_slot, albedo),
            (material_slot, material),
        ] {
            heap.write_sampled(
                gpu,
                slot,
                gpu.texture_view_descriptor(owned.texture, TextureViewDesc::default()),
            );
        }

        *this = Some(Self {
            size,
            normal,
            albedo,
            material,
            normal_slot,
            albedo_slot,
            material_slot,
        });
        true
    }

    /// Returns the three attachment textures for the forward pass.
    pub fn forward_targets(&self) -> MeshForwardSurfaceTargets {
        MeshForwardSurfaceTargets {
            normal: self.normal.texture,
            albedo: self.albedo.texture,
            material: self.material.texture,
        }
    }

    pub fn normal_slot(&self) -> SampledSlot {
        self.normal_slot
    }
    pub fn albedo_slot(&self) -> SampledSlot {
        self.albedo_slot
    }
    pub fn material_slot(&self) -> SampledSlot {
        self.material_slot
    }

    pub fn size(&self) -> UVec2 {
        self.size
    }
}

impl Pass for MeshSurfaceTargets {
    const NAME: &'static str = "mesh_surface_targets";

    fn free(self, gpu: &Gpu) {
        gpu.texture_free_and_destroy(self.normal);
        gpu.texture_free_and_destroy(self.albedo);
        gpu.texture_free_and_destroy(self.material);
    }
}

/// Dense per-pixel shadow state consumed by [`LocalLightPass`].
#[derive(Clone, Copy)]
pub struct MeshShadowMask {
    states: GpuPtr<u32>,
    size: UVec2,
    light_count: u32,
}

impl MeshShadowMask {
    pub fn states(self) -> GpuPtr<u32> {
        self.states
    }

    pub fn size(self) -> UVec2 {
        self.size
    }

    pub fn light_count(self) -> u32 {
        self.light_count
    }
}

/// Produces the dense per-light shadow mask before local lighting.
pub struct MeshShadowPass {
    exact_shader: gpu::Shader,
    states: gpu::Ptr<u32>,
    size: UVec2,
    max_lights: u32,
    tlas: ShadowTlasBuilder,
}

impl MeshShadowPass {
    pub const GROUP_SIZE: u32 = 64;

    pub fn new(gpu: &Gpu, size: UVec2, max_lights: u32, instance_capacity: u32) -> Self {
        assert!(
            size.x > 0 && size.y > 0,
            "mesh shadow size must be positive"
        );
        assert!(max_lights > 0, "mesh shadow max_lights must be positive");
        let capacity = pair_capacity(size, max_lights);
        Self {
            exact_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("mesh_exact_shadow_mask"),
                Self::GROUP_SIZE,
                1,
                1,
                "mesh_exact_shadow_mask",
            ),
            states: gpu.alloc_slice(capacity, Memory::Gpu),
            size,
            max_lights,
            tlas: ShadowTlasBuilder::new(instance_capacity),
        }
    }

    /// Resizes buffers after waiting for in-flight GPU use.
    pub fn resize(&mut self, gpu: &Gpu, size: UVec2) -> bool {
        assert!(
            size.x > 0 && size.y > 0,
            "mesh shadow size must be positive"
        );
        if size == self.size {
            return false;
        }
        let capacity = pair_capacity(size, self.max_lights);
        gpu.queue_wait_idle(Queue::Main);
        gpu.free(self.states);
        self.states = gpu.alloc_slice(capacity, Memory::Gpu);
        self.size = size;
        true
    }

    pub fn allocated_bytes(&self) -> u64 {
        pair_capacity(self.size, self.max_lights) * core::mem::size_of::<u32>() as u64
    }

    pub fn states_buffer(&self) -> gpu::Ptr<u32> {
        self.states
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        surfaces: &MeshSurfaceTargets,
        view: MeshRasterView,
        lights: GpuPtr<PointLight>,
        light_count: u32,
        origin_bias: f32,
        destination_bias: f32,
        depth_slot: SampledSlot,
    ) -> (MeshShadowMask, ShadowTlasStats) {
        assert_eq!(
            surfaces.size(),
            self.size,
            "mesh shadow buffer and mesh surfaces must have the same size"
        );
        assert!(
            light_count <= self.max_lights,
            "mesh shadow light count {light_count} exceeds capacity {}",
            self.max_lights
        );
        assert!(
            origin_bias.is_finite() && origin_bias >= 0.0,
            "mesh shadow origin bias must be finite and nonnegative"
        );
        assert!(
            destination_bias.is_finite() && destination_bias >= 0.0,
            "mesh shadow destination bias must be finite and nonnegative"
        );
        let det = view.world_to_clip.determinant();
        assert!(
            view.world_to_clip.is_finite() && det.is_finite() && det.abs() > 1.0e-8,
            "mesh shadows reconstruct world position: world_to_clip must be finite and invertible"
        );
        if light_count != 0 {
            assert!(!lights.is_null(), "mesh shadow lights pointer is null");
        }

        let mask = MeshShadowMask {
            states: self.states.gpu,
            size: self.size,
            light_count,
        };
        let (world, stats) = self.tlas.build_instances(fa, scene, instances);
        let pixel_count = self
            .size
            .x
            .checked_mul(self.size.y)
            .expect("mesh shadow pixel count exceeds u32");
        let pair_count = pixel_count
            .checked_mul(light_count)
            .expect("mesh shadow pair count exceeds u32");
        if pair_count == 0 {
            return (mask, stats);
        }

        let data = fa.frame_alloc(MeshShadowData {
            clip_to_world: view.world_to_clip.inverse(),
            world_to_clip: view.world_to_clip,
            world,
            lights,
            states: self.states.gpu,
            depth_texture_id: depth_slot.index(),
            surface_material_texture_id: surfaces.material_slot().index(),
            screen_size: self.size.to_array(),
            light_count,
            origin_bias,
            destination_bias,
            _pad0: [0; 3],
        });

        heap.bind(gpu, cb);
        gpu.cmd_barrier(cb, Stage::All, Stage::Compute, HazardFlags::empty());
        gpu.cmd_set_compute_shader(cb, self.exact_shader);
        gpu.cmd_dispatch(cb, data, pair_count.div_ceil(Self::GROUP_SIZE), 1, 1);
        (mask, stats)
    }
}

impl Pass for MeshShadowPass {
    const NAME: &'static str = "mesh_shadow";

    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.exact_shader);
        gpu.free(self.states);
    }
}

fn pair_capacity(size: UVec2, max_lights: u32) -> u64 {
    u64::from(size.x)
        .checked_mul(u64::from(size.y))
        .and_then(|pixels| pixels.checked_mul(u64::from(max_lights)))
        .expect("mesh shadow capacity overflow")
}

/// Post-opaque local-light accumulation dispatch.
///
/// It consumes surface MRTs and optional shadow results, then preserves HDR
/// alpha while adding the selected lights in input-array order.
pub struct LocalLightPass {
    shader: gpu::Shader,
}

impl LocalLightPass {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            shader: gpu.shader_create_compute(
                &asha_assets::load_spv("mesh_local_light"),
                LOCAL_LIGHT_GROUP_SIZE,
                LOCAL_LIGHT_GROUP_SIZE,
                1,
                "mesh_local_light",
            ),
        }
    }

    /// Records lighting after shadows and forward MRT production.
    ///
    /// The surface targets, depth, light count, and shadow buffers must cover
    /// the same physical extent and light ordering. Zero lights record nothing.
    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        surfaces: &MeshSurfaceTargets,
        view: MeshRasterView,
        lights: GpuPtr<PointLight>,
        light_count: u32,
        wrap_w: f32,
        light_field: MeshLightField,
        mesh_shadows: Option<MeshShadowMask>,
        local_shadows: Option<crate::LocalShadowSlots>,
        ramp_default_sampler: SamplerSlot,
        depth_slot: SampledSlot,
        hdr_rw: StorageSlot,
    ) {
        let size = surfaces.size();
        assert!(size.x > 0 && size.y > 0);
        let det = view.world_to_clip.determinant();
        assert!(
            view.world_to_clip.is_finite() && det.is_finite() && det.abs() > 1.0e-8,
            "local lights reconstruct world position: world_to_clip must be finite and invertible"
        );
        assert_ne!(
            ramp_default_sampler.index(),
            0,
            "local lights need a real default ramp sampler"
        );
        assert!(wrap_w.is_finite(), "local-light wrap must be finite");
        assert!(
            light_field.gate.is_finite() && (0.0..=1.0).contains(&light_field.gate),
            "light-field gate must be finite and in 0..=1"
        );
        if !light_field.cells.is_null() {
            assert!(
                light_field.dims[0] > 0 && light_field.dims[1] > 0,
                "non-null light field needs positive dimensions"
            );
            assert!(
                u64::from(light_field.dims[0]) * u64::from(light_field.dims[1]) <= i32::MAX as u64,
                "light field must fit the shared signed cell index"
            );
            assert!(
                light_field.cell_size.is_finite() && light_field.cell_size > 0.0,
                "non-null light field needs a finite positive cell size"
            );
        }
        assert!(
            scene.material_count() <= SURFACE_MATERIAL_INDEX_MAX + 1,
            "material count exceeds the exact f32 surface-marker range"
        );
        if let Some(mask) = mesh_shadows {
            assert_eq!(
                mask.size, size,
                "mesh shadow mask and local-light surfaces must have the same size"
            );
            assert_eq!(
                mask.light_count, light_count,
                "mesh shadow mask and local-light dispatch must cover the same lights"
            );
            assert!(!mask.states.is_null(), "mesh shadow state pointer is null");
        }
        assert!(
            !(mesh_shadows.is_some() && local_shadows.is_some()),
            "dense mask and v2 slot shadows are mutually exclusive visibility sources"
        );
        if let Some(slots) = local_shadows {
            assert_eq!(
                slots.size(),
                size,
                "local shadow slots and local-light surfaces must have the same size"
            );
            assert_eq!(
                slots.light_count(),
                light_count,
                "local shadow slots and local-light dispatch must cover the same lights"
            );
            assert!(
                !slots.slot_map().is_null()
                    && !slots.slot_state().is_null()
                    && !slots.slot_rep().is_null(),
                "local shadow slot pointers must be non-null"
            );
        }

        if light_count == 0 {
            return;
        }
        assert!(
            light_count == 0 || !lights.is_null(),
            "nonzero light_count with a null light pointer"
        );

        let data = fa.frame_alloc(LocalLightData {
            clip_to_world: view.world_to_clip.inverse(),
            materials: scene.materials_ptr(),
            lights,
            light_field: light_field.cells,
            shadow_states: mesh_shadows.map_or(GpuPtr::null(), |mask| mask.states),
            slot_map: local_shadows.map_or(GpuPtr::null(), |s| s.slot_map()),
            slot_state: local_shadows.map_or(GpuPtr::null(), |s| s.slot_state()),
            slot_rep: local_shadows.map_or(GpuPtr::null(), |s| s.slot_rep()),
            slot_fraction: local_shadows.map_or(GpuPtr::null(), |s| s.slot_fraction()),
            depth_texture_id: depth_slot.index(),
            surface_normal_texture_id: surfaces.normal_slot().index(),
            surface_albedo_texture_id: surfaces.albedo_slot().index(),
            surface_material_texture_id: surfaces.material_slot().index(),
            hdr_texture_id: hdr_rw.index(),
            ramp_default_sampler: ramp_default_sampler.index(),
            screen_size: size.to_array(),
            light_count,
            wrap_w,
            light_field_dims: light_field.dims,
            light_field_cell_size: light_field.cell_size,
            light_field_gate: light_field.gate,
            half_size: local_shadows.map_or([0; 2], |s| s.half_size().to_array()),
            debug_overlay: u32::from(
                local_shadows.is_some()
                    && std::env::var("ASHA_SHADOW_OVERLAY").is_ok_and(|v| v != "0"),
            ),
            _pad0: [0; 3],
        });

        heap.bind(gpu, cb);
        if mesh_shadows.is_some() || local_shadows.is_some() {
            // Publish shadow state before this compute consumer.
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::Compute,
                HazardFlags::SHADER_BUFFER,
            );
        } else {
            // Publish forward MRT and HDR writes to the first compute consumer.
            gpu.cmd_barrier(
                cb,
                Stage::RasterColorOut,
                Stage::Compute,
                HazardFlags::empty(),
            );
        }
        gpu.cmd_set_compute_shader(cb, self.shader);
        gpu.cmd_dispatch(
            cb,
            data,
            size.x.div_ceil(LOCAL_LIGHT_GROUP_SIZE),
            size.y.div_ceil(LOCAL_LIGHT_GROUP_SIZE),
            1,
        );
        // Later fragment passes sample the completed HDR image.
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::FragmentShader,
            HazardFlags::SHADER_IMAGE,
        );
    }
}

impl Pass for LocalLightPass {
    const NAME: &'static str = "mesh_local_light";

    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.shader);
    }
}
