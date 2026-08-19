//! Fullscreen visibility-buffer linework for mesh depth hosts.

use abi_core::GpuPtr;
use abi_core::glam::{UVec2, Vec3};
use abi_mesh::{ClusterInstance, LineworkData, MeshFrameData};
use gpu::{
    BlendFactor, BlendOp, BlendState, CommandBuffer, Gpu, HeapSlots, LoadOp, RenderAttachment,
    RenderPassDesc, SampledSlot, ShaderTypeGraphics, StorageSlot, StoreOp, Texture,
};

use crate::{
    FrameAlloc, MeshInstances, MeshLightField, MeshRasterView, MeshScene, mesh_frame_data,
};

/// CPU-configured screen-space linework controls.
#[derive(Clone, Copy, Debug)]
pub struct LineworkDials {
    pub enabled: bool,
    pub normal_cos_threshold: f32,
    pub plane_epsilon: f32,
    pub crease_strength: f32,
    pub step_strength: f32,
    pub fade_near: f32,
    pub fade_far: f32,
}

/// Fullscreen visibility/depth resolve using caller-owned textures.
pub struct MeshLineworkPass {
    fullscreen_vert: gpu::Shader,
    frag_shader: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl MeshLineworkPass {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            fullscreen_vert: gpu.shader_create(
                &asha_assets::load_spv("fullscreen_vert"),
                ShaderTypeGraphics::Vertex,
                "fullscreen_vert",
            ),
            frag_shader: gpu.shader_create(
                &asha_assets::load_spv("mesh_linework_frag"),
                ShaderTypeGraphics::Fragment,
                "mesh_linework_frag",
            ),
            indices: gpu.fullscreen_triangle_indices(),
        }
    }

    /// Composites linework onto `target` using completed prepass data.
    /// Record before another cull reuses the compacted-cluster allocation.
    #[allow(clippy::too_many_arguments)] // Each argument is real pass dataflow.
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        clusters: GpuPtr<ClusterInstance>,
        view: MeshRasterView,
        eye: Vec3,
        target: Texture,
        depth_slot: SampledSlot,
        visibility_slot: StorageSlot,
        dials: LineworkDials,
    ) {
        self.record_with_light_field(
            gpu,
            cb,
            fa,
            heap,
            scene,
            instances,
            clusters,
            view,
            eye,
            target,
            depth_slot,
            visibility_slot,
            dials,
            MeshLightField::default(),
        );
    }

    /// Records linework with the forward pass's light-field sample.
    #[allow(clippy::too_many_arguments)] // Each argument is real pass dataflow.
    pub fn record_with_light_field(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        clusters: GpuPtr<ClusterInstance>,
        view: MeshRasterView,
        eye: Vec3,
        target: Texture,
        depth_slot: SampledSlot,
        visibility_slot: StorageSlot,
        dials: LineworkDials,
        light_field: MeshLightField,
    ) {
        let size = UVec2::new(target.dimensions[0], target.dimensions[1]);
        assert!(size.x > 0 && size.y > 0);
        assert!(eye.is_finite(), "linework eye must be finite");
        assert!(
            dials.normal_cos_threshold.is_finite()
                && (-1.0..1.0).contains(&dials.normal_cos_threshold),
            "linework normal cosine must be finite and in [-1, 1)"
        );
        assert!(
            dials.plane_epsilon.is_finite() && dials.plane_epsilon > 0.0,
            "linework plane epsilon must be finite and positive"
        );
        assert!(
            dials.crease_strength.is_finite()
                && dials.step_strength.is_finite()
                && dials.crease_strength >= 0.0
                && dials.step_strength >= 0.0,
            "linework strengths must be finite and nonnegative"
        );
        assert!(
            dials.fade_near.is_finite()
                && dials.fade_far.is_finite()
                && dials.fade_near >= 0.0
                && dials.fade_far > dials.fade_near,
            "linework fade range must be finite, nonnegative, and increasing"
        );
        let determinant = view.world_to_clip.determinant();
        assert!(
            determinant.is_finite() && determinant.abs() > 1.0e-8,
            "linework raster matrix must be invertible"
        );
        let clip_to_world = view.world_to_clip.inverse();
        assert!(
            clip_to_world.is_finite(),
            "linework raster matrix inverse must be finite"
        );
        light_field.assert_valid();

        let frame = fa.frame_alloc(MeshFrameData {
            light_field: light_field.cells,
            light_field_dims: light_field.dims,
            light_field_cell_size: light_field.cell_size,
            light_field_gate: light_field.gate,
            eye: eye.to_array(),
            ..mesh_frame_data(scene, instances, clusters, view)
        });
        let linework = fa.frame_alloc(LineworkData {
            clip_to_world,
            frame,
            eye: eye.to_array(),
            _pad0: 0.0,
            depth_texture_id: depth_slot.index(),
            visibility_texture_id: visibility_slot.index(),
            screen_size: size.to_array(),
            normal_cos_threshold: dials.normal_cos_threshold,
            plane_epsilon: dials.plane_epsilon,
            crease_strength: dials.crease_strength,
            step_strength: dials.step_strength,
            fade_near: dials.fade_near,
            fade_far: dials.fade_far,
            darkness_seat: 1.0,
            _pad1: [0; 3],
            light_field: light_field.cells,
            light_field_dims: light_field.dims,
            light_field_cell_size: light_field.cell_size,
            light_field_gate: light_field.gate,
            _pad2: [0; 2],
        });

        heap.bind(gpu, cb);
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                render_area_size: size.to_array(),
                color_attachments: &[RenderAttachment {
                    texture: target,
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4],
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        gpu.cmd_set_shaders(cb, self.fullscreen_vert, self.frag_shader);
        gpu.cmd_set_blend_state(
            cb,
            BlendState {
                enable: true,
                color_op: BlendOp::Add,
                src_color_factor: BlendFactor::SrcAlpha,
                dst_color_factor: BlendFactor::OneMinusSrcAlpha,
                alpha_op: BlendOp::Add,
                src_alpha_factor: BlendFactor::One,
                dst_alpha_factor: BlendFactor::OneMinusSrcAlpha,
                color_write_mask: 0x0f,
            },
        );
        gpu.cmd_draw_indexed_instanced(
            cb,
            GpuPtr::null(),
            linework.cast(),
            self.indices.cast(),
            3,
            1,
        );
        gpu.cmd_end_render_pass(cb);
    }

    pub fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.frag_shader);
        gpu.free(self.indices);
    }
}
