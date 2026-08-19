//! Shared headless verification and input helpers for render examples.

use bevy::prelude::*;
use gpu::{
    Gpu, HazardFlags, Memory, OwnedTexture, Queue, Stage, TextureDesc, TextureFormat, UsageFlags,
};
use render::FrameCtx;

pub fn esc_to_exit(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

/// Applies a radial deadzone while preserving remaining travel.
#[allow(dead_code)]
pub fn radial_deadzone(v: Vec2, dz: f32) -> Vec2 {
    let len = v.length();
    if len < dz {
        return Vec2::ZERO;
    }
    v * ((len - dz) / (1.0 - dz) / len)
}

/// Frame used to record the offscreen verification copy.
#[allow(dead_code)]
pub const VERIFY_COPY_FRAME: u64 = 30;
#[allow(dead_code)]
pub const VERIFY_CHECK_FRAME: u64 = 36;

/// Optional offscreen target and readback for example verification.
#[allow(dead_code)]
pub struct Verify {
    pub target: OwnedTexture,
    pub readback: gpu::Ptr<u8>,
    width: u32,
}

#[allow(dead_code)]
impl Verify {
    /// `Some` iff `ASHA_VERIFY` is set in the environment.
    pub fn from_env(gpu: &Gpu, width: u32, height: u32) -> Option<Self> {
        std::env::var_os("ASHA_VERIFY").map(|_| Self {
            target: gpu.texture_alloc_and_create(
                TextureDesc {
                    dimensions: [width, height, 1],
                    format: TextureFormat::Rgba8Unorm,
                    usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
                    ..Default::default()
                },
                Queue::Main,
                None,
            ),
            readback: gpu.alloc_slice::<u8>((width * height * 4) as u64, Memory::Readback),
            width,
        })
    }

    /// Records a transfer copy after publishing the finished color pass.
    pub fn copy_frame(&self, ctx: &FrameCtx) {
        ctx.gpu.cmd_barrier(
            ctx.cb,
            Stage::RasterColorOut,
            Stage::Transfer,
            HazardFlags::empty(),
        );
        ctx.gpu
            .cmd_copy_texture_to_buffer(ctx.cb, self.target.texture, self.readback.cast());
    }

    /// Reads a copied pixel after its frame has retired.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        unsafe {
            let p = self.readback.cpu.add(((y * self.width + x) * 4) as usize);
            [*p, *p.add(1), *p.add(2), *p.add(3)]
        }
    }

    pub fn teardown(self, gpu: &Gpu) {
        gpu.texture_free_and_destroy(self.target);
        gpu.free(self.readback);
    }
}
