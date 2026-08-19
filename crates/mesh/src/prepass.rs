//! Mesh depth prepass and visibility-token writer.
//!
//! It clears reverse-Z depth and visibility, then draws culled clusters.
//! Forward consumes depth under `Equal + Read`; visibility consumers read tokens.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_mesh::{ClusterInstance, IndirectData};
use gpu::{
    CommandBuffer, CompareOp, DepthFlags, DepthState, Gpu, HazardFlags, LoadOp, Queue,
    RenderAttachment, RenderPassDesc, ShaderTypeGraphics, Stage, StoreOp, Texture, TextureDesc,
    TextureFormat, UsageFlags,
};

use crate::{FrameAlloc, MeshInstances, MeshRasterView, MeshScene, mesh_frame_data};

pub struct MeshDepthPrepass {
    vert_shader: gpu::Shader,
    frag_shader: gpu::Shader,
    /// Screen-sized R32_Uint tokens: 25-bit cluster index and 7-bit primitive ID.
    /// Unavailable until [`Self::resize`].
    visibility: Option<gpu::OwnedTexture>,
    size: UVec2,
}

impl MeshDepthPrepass {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            vert_shader: gpu.shader_create(
                &asha_assets::load_spv("mesh_prepass_vert"),
                ShaderTypeGraphics::Vertex,
                "mesh_prepass_vert",
            ),
            frag_shader: gpu.shader_create(
                &asha_assets::load_spv("mesh_prepass_frag"),
                ShaderTypeGraphics::Fragment,
                "mesh_prepass_frag",
            ),
            visibility: None,
            size: UVec2::ZERO,
        }
    }

    /// Resizes the visibility image. The old image must be GPU-idle.
    pub fn resize(&mut self, gpu: &Gpu, size: UVec2) {
        assert!(size.x > 0 && size.y > 0);
        if let Some(old) = self.visibility.take() {
            gpu.texture_free_and_destroy(old);
        }
        self.visibility = Some(gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [size.x, size.y, 1],
                format: TextureFormat::R32Uint,
                // Consumed by visibility resolves and verification readbacks.
                usage: UsageFlags::COLOR_ATTACHMENT
                    | UsageFlags::SAMPLED
                    | UsageFlags::STORAGE
                    | UsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            Queue::Main,
            None,
        ));
        self.size = size;
    }

    /// Returns tokens for visibility consumers; panics before [`Self::resize`].
    pub fn visibility_texture(&self) -> Texture {
        self.visibility
            .as_ref()
            .expect("prepass resized before use")
            .texture
    }

    /// Clears depth and visibility, then draws the culled indirect list.
    /// Vertex math must match `mesh_vert` exactly for forward `Equal` depth.
    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
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
        depth: Texture,
        size: UVec2,
        view: MeshRasterView,
    ) {
        assert!(size.x > 0 && size.y > 0);
        assert!(
            self.size == size,
            "prepass visibility not resized to the screen"
        );

        // Both raster passes share frame data and the cull's instance stream.
        let frame = fa.frame_alloc(mesh_frame_data(scene, instances, clusters, view));

        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: self.visibility_texture(),
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4], // All-zero = invalid/sky token.
                    ..Default::default()
                }],
                depth_attachment: Some(RenderAttachment {
                    texture: depth,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4], // Reverse-Z: clear to 0 (far).
                    ..Default::default()
                }),
                render_area_size: [size.x, size.y],
                ..Default::default()
            },
        );
        gpu.cmd_set_shaders(cb, self.vert_shader, self.frag_shader);
        // Reverse-Z uses greater depth for nearer fragments.
        gpu.cmd_set_depth_state(
            cb,
            DepthState {
                mode: DepthFlags::READ | DepthFlags::WRITE,
                compare: CompareOp::Greater,
                ..Default::default()
            },
        );
        // Match forward raster state so both passes cover identical fragments.
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
        Self::trailing_barrier(gpu, cb);
    }

    /// Clears depth and visibility without recording an indirect draw.
    pub fn record_clear_only(&self, gpu: &Gpu, cb: CommandBuffer, depth: Texture, size: UVec2) {
        assert!(size.x > 0 && size.y > 0);
        assert!(
            self.size == size,
            "prepass visibility not resized to the screen"
        );

        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: self.visibility_texture(),
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4], // All-zero = invalid/sky token.
                    ..Default::default()
                }],
                depth_attachment: Some(RenderAttachment {
                    texture: depth,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4], // Reverse-Z: clear to 0 (far).
                    ..Default::default()
                }),
                render_area_size: [size.x, size.y],
                ..Default::default()
            },
        );
        gpu.cmd_end_render_pass(cb);
        // Match the full prepass's trailing barrier.
        Self::trailing_barrier(gpu, cb);
    }

    /// Makes depth and visibility writes available to following compute.
    fn trailing_barrier(gpu: &Gpu, cb: CommandBuffer) {
        gpu.cmd_barrier(
            cb,
            Stage::All,
            Stage::All,
            HazardFlags::DEPTH_STENCIL | HazardFlags::COLOR_ATTACHMENT,
        );
    }

    pub fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.vert_shader);
        gpu.shader_destroy(self.frag_shader);
        if let Some(visibility) = self.visibility {
            gpu.texture_free_and_destroy(visibility);
        }
    }
}
