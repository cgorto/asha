//! Depth-visible mesh silhouette mask pass.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_mesh::{ClusterInstance, IndirectData};
use gpu::{
    CommandBuffer, CompareOp, DepthFlags, DepthState, Gpu, LoadOp, RenderAttachment,
    RenderPassDesc, ShaderTypeGraphics, StoreOp, Texture,
};

use crate::{FrameAlloc, MeshInstances, MeshRasterView, MeshScene, mesh_frame_data};

/// Renders selected surfaces into a caller-owned R8_UNORM mask.
pub struct MeshSilhouettePass {
    vert_shader: gpu::Shader,
    frag_shader: gpu::Shader,
}

impl MeshSilhouettePass {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            vert_shader: gpu.shader_create(
                &asha_assets::load_spv("mesh_silhouette_vert"),
                ShaderTypeGraphics::Vertex,
                "mesh_silhouette_vert",
            ),
            frag_shader: gpu.shader_create(
                &asha_assets::load_spv("mesh_silhouette_frag"),
                ShaderTypeGraphics::Fragment,
                "mesh_silhouette_frag",
            ),
        }
    }

    /// Renders prepass clusters into a cleared mask using shared depth.
    #[allow(clippy::too_many_arguments)] // Every argument is real pass dataflow.
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        scene: &MeshScene,
        instances: MeshInstances<'_>,
        culled_args: gpu::Ptr<IndirectData>,
        clusters: GpuPtr<ClusterInstance>,
        draw_count: gpu::Ptr<u32>,
        mask: Texture,
        depth: Texture,
        size: UVec2,
        view: MeshRasterView,
    ) {
        assert!(size.x > 0 && size.y > 0);

        let frame = fa.frame_alloc(mesh_frame_data(scene, instances, clusters, view));

        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                render_area_size: size.to_array(),
                color_attachments: &[RenderAttachment {
                    texture: mask,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4],
                    ..Default::default()
                }],
                depth_attachment: Some(RenderAttachment {
                    texture: depth,
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4],
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        gpu.cmd_set_shaders(cb, self.vert_shader, self.frag_shader);
        // Equal depth limits the mask to prepass-visible fragments.
        gpu.cmd_set_depth_state(
            cb,
            DepthState {
                mode: DepthFlags::READ,
                compare: CompareOp::Equal,
                ..Default::default()
            },
        );
        // Match prepass and forward raster state.
        gpu.cmd_set_cull_mode(cb, true);
        gpu.cmd_set_front_face(cb, false);
        gpu.cmd_draw_instanced_indirect_multi(
            cb,
            frame.cast(),
            frame.cast(),
            culled_args.cast(),
            core::mem::size_of::<IndirectData>() as u32,
            draw_count.cast(),
        );
        gpu.cmd_end_render_pass(cb);
    }

    pub fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.vert_shader);
        gpu.shader_destroy(self.frag_shader);
    }
}
