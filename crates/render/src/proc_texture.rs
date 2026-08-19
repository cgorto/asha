//! Procedural textures cross the seam through a reserved sampled-slot block.
//!
//! The main thread assigns `base + index` immediately and queues registrations.
//! The render-thread bridge reserves the same contiguous block, asserts the
//! base matches, and ingests registrations before geometry records. Pixel fills
//! use RGBA8 staging copies; bake fills render once into an RGBA8 sampled target.
//! Both paths are sampleable later in the same frame. Staging remains alive
//! until the frame-retirement gate proves its copy has completed.

use abi_bake::BakeData;
use abi_core::GpuPtr;
use bevy::prelude::*;
use gpu::{
    Gpu, HazardFlags, HeapSlots, LoadOp, Memory, OwnedTexture, Queue, RenderAttachment,
    RenderPassDesc, SampledSlot, ShaderTypeGraphics, Stage, StaticTexture, StoreOp, TextureDesc,
    TextureFormat, TextureViewDesc, UsageFlags,
};

use crate::{FRAMES_IN_FLIGHT, FrameCtx};

/// A registered texture; its value is the sampled-heap slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcTexture(u32);

impl ProcTexture {
    pub fn slot(self) -> u32 {
        self.0
    }
}

/// One ordered registration sent to the render thread with its reserved slot.
pub struct ProcTextureUpload {
    /// Position within the reserved block; must equal registration order.
    pub index: u32,
    pub slot: u32,
    pub width: u32,
    pub height: u32,
    pub fill: ProcTextureFill,
}

/// Who authors the texels.
pub enum ProcTextureFill {
    /// Linear row-major RGBA8, exactly `width * height * 4` bytes.
    Pixels(Vec<u8>),
    /// A GPU generator, run once at ingest — see `abi_bake`.
    Bake(BakeData),
}

/// Main-thread registry for an agreed reserved slot block.
///
/// `base_slot` and `capacity` must match [`ProcTextureBridge::new`].
#[derive(Resource)]
pub struct ProcTextures {
    base_slot: u32,
    capacity: u32,
    count: u32,
    pending: Vec<ProcTextureUpload>,
}

impl ProcTextures {
    pub fn new(base_slot: u32, capacity: u32) -> Self {
        Self {
            base_slot,
            capacity,
            count: 0,
            pending: Vec::new(),
        }
    }

    /// Queues an RGBA8 fill and returns its slot immediately.
    ///
    /// The slot is usable in a material now; texels arrive during next ingest.
    pub fn add(&mut self, width: u32, height: u32, pixels: Vec<u8>) -> ProcTexture {
        assert!(
            pixels.len() == (width as usize) * (height as usize) * 4,
            "procedural texture {width}x{height}: expected {} RGBA8 bytes, got {}",
            (width as usize) * (height as usize) * 4,
            pixels.len(),
        );
        self.register(width, height, ProcTextureFill::Pixels(pixels))
    }

    /// Queues a generator that renders once during ingest.
    ///
    /// Its slot follows the same immediate-material and next-ingest contract.
    pub fn bake(&mut self, width: u32, height: u32, data: BakeData) -> ProcTexture {
        self.register(width, height, ProcTextureFill::Bake(data))
    }

    fn register(&mut self, width: u32, height: u32, fill: ProcTextureFill) -> ProcTexture {
        assert!(
            self.count < self.capacity,
            "procedural texture block full ({} slots) — raise the capacity passed to \
             AshaRenderPlugin::proc_textures and ProcTextureBridge::new together",
            self.capacity,
        );
        let index = self.count;
        self.count += 1;
        let slot = self.base_slot + index;
        self.pending.push(ProcTextureUpload {
            index,
            slot,
            width,
            height,
            fill,
        });
        ProcTexture(slot)
    }

    pub fn count(&self) -> u32 {
        self.count
    }
}

/// Render-thread owner of reserved slots, textures, and staging lifetimes.
///
/// Call [`Self::ingest`] once before scene recording and [`Self::free`] after
/// the GPU is idle.
pub struct ProcTextureBridge {
    base: SampledSlot,
    reserved: Vec<SampledSlot>,
    textures: Vec<SlotTexture>,
    /// Staging entries retained until their recording frame retires.
    staging: Vec<(u64, usize)>,
    /// Lazily created shared bake pipeline.
    bake: Option<BakeShaders>,
}

/// Texture storage selected by its fill path.
enum SlotTexture {
    Static(StaticTexture),
    Baked {
        texture: OwnedTexture,
        params: gpu::Ptr<BakeData>,
    },
}

/// Shared one-shot fullscreen pipeline for baked textures.
struct BakeShaders {
    vert: gpu::Shader,
    frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl BakeShaders {
    fn new(gpu: &Gpu) -> Self {
        Self {
            vert: gpu.shader_create(
                &asha_assets::load_spv("fullscreen_vert"),
                ShaderTypeGraphics::Vertex,
                "fullscreen_vert",
            ),
            frag: gpu.shader_create(
                &asha_assets::load_spv("bake_frag"),
                ShaderTypeGraphics::Fragment,
                "bake_frag",
            ),
            indices: gpu.fullscreen_triangle_indices(),
        }
    }

    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.vert);
        gpu.shader_destroy(self.frag);
        gpu.free(self.indices);
    }
}

impl ProcTextureBridge {
    /// Reserves a contiguous block and verifies the agreed base slot.
    pub fn new(heap: &mut HeapSlots, base_slot: u32, capacity: u32) -> Self {
        assert!(capacity > 0, "a zero-slot procedural texture block");
        let reserved: Vec<SampledSlot> = (0..capacity).map(|_| heap.alloc_sampled()).collect();
        let base = reserved[0];
        assert!(
            base.index() == base_slot,
            "procedural texture block starts at sampled slot {} but the main thread was \
             told {base_slot} — something registered into the sampled heap first",
            base.index(),
        );
        Self {
            base,
            reserved,
            textures: Vec::new(),
            staging: Vec::new(),
            bake: None,
        }
    }

    /// Ingests ordered fills before geometry and releases retired staging.
    ///
    /// Pixel copies and bake draws are sampleable by later passes this frame;
    /// staging is released only after the GPU retirement gate permits it.
    pub fn ingest(&mut self, ctx: &mut FrameCtx, heap: &HeapSlots) {
        let gpu = ctx.gpu;
        let cb = ctx.cb;
        let frame = ctx.frame;
        let mut baked = false;
        for upload in ctx.proc_texture_uploads() {
            assert!(
                upload.index as usize == self.textures.len(),
                "procedural texture index authority violated: got {}, expected {}",
                upload.index,
                self.textures.len(),
            );
            let slot = self.reserved[upload.index as usize];
            assert!(
                slot.index() == upload.slot,
                "procedural texture {} claims slot {} but reserved {}",
                upload.index,
                upload.slot,
                slot.index(),
            );
            match upload.fill {
                ProcTextureFill::Pixels(pixels) => {
                    let texture = StaticTexture::from_rgba8_in_slot(
                        gpu,
                        heap,
                        slot,
                        upload.width,
                        upload.height,
                        &pixels,
                    );
                    texture.upload(gpu, cb);
                    self.staging.push((frame, self.textures.len()));
                    self.textures.push(SlotTexture::Static(texture));
                }
                ProcTextureFill::Bake(data) => {
                    let texture = gpu.texture_alloc_and_create(
                        TextureDesc {
                            dimensions: [upload.width, upload.height, 1],
                            format: TextureFormat::Rgba8Unorm,
                            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::SAMPLED,
                            ..Default::default()
                        },
                        Queue::Main,
                        None,
                    );
                    heap.write_sampled(
                        gpu,
                        slot,
                        gpu.texture_view_descriptor(texture.texture, TextureViewDesc::default()),
                    );
                    let params: gpu::Ptr<BakeData> = gpu.alloc_slice(1, Memory::Default);
                    // SAFETY: one BakeData in freshly allocated host-visible
                    // memory; nothing has been recorded against it yet.
                    unsafe { *params.cpu = data };
                    let shaders = self.bake.get_or_insert_with(|| BakeShaders::new(gpu));
                    gpu.cmd_begin_render_pass(
                        cb,
                        RenderPassDesc {
                            color_attachments: &[RenderAttachment {
                                texture: texture.texture,
                                load_op: LoadOp::DontCare,
                                store_op: StoreOp::Store,
                                clear_color: [0.0; 4],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    );
                    gpu.cmd_set_shaders(cb, shaders.vert, shaders.frag);
                    gpu.cmd_draw_indexed_instanced(
                        cb,
                        GpuPtr::null(),
                        params.gpu.cast(),
                        shaders.indices.cast(),
                        3,
                        1,
                    );
                    gpu.cmd_end_render_pass(cb);
                    baked = true;
                    self.textures.push(SlotTexture::Baked { texture, params });
                }
            }
        }
        if baked {
            gpu.cmd_barrier(
                cb,
                Stage::RasterColorOut,
                Stage::FragmentShader,
                HazardFlags::COLOR_ATTACHMENT,
            );
        }

        let retired = (frame + 1).saturating_sub(FRAMES_IN_FLIGHT);
        let textures = &mut self.textures;
        self.staging.retain(|&(recorded, index)| {
            if recorded > retired {
                return true;
            }
            match &mut textures[index] {
                SlotTexture::Static(texture) => texture.upload_finish(gpu),
                SlotTexture::Baked { .. } => unreachable!("baked texture in staging lane"),
            }
            false
        });
    }

    /// Returns the reserved slot for a registration index.
    pub fn slot(&self, index: u32) -> SampledSlot {
        self.reserved[index as usize]
    }

    /// The block's base slot, as reserved.
    pub fn base(&self) -> SampledSlot {
        self.base
    }

    pub fn free(mut self, gpu: &Gpu) {
        for &(_, index) in &self.staging {
            match &mut self.textures[index] {
                SlotTexture::Static(texture) => texture.upload_finish(gpu),
                SlotTexture::Baked { .. } => unreachable!("baked texture in staging lane"),
            }
        }
        for entry in self.textures {
            match entry {
                SlotTexture::Static(texture) => texture.free(gpu),
                SlotTexture::Baked { texture, params } => {
                    gpu.texture_free_and_destroy(texture);
                    gpu.free(params);
                }
            }
        }
        if let Some(bake) = self.bake {
            bake.free(gpu);
        }
    }
}

/// Appends this frame's registrations in registration order.
pub(crate) fn make_extract() -> crate::ExtractFn {
    Box::new(move |world, frame| {
        let mut textures = world.resource_mut::<ProcTextures>();
        frame.proc_texture_uploads.append(&mut textures.pending);
    })
}
