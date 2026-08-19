//! Shared fixtures for post-processing GPU tests.
#![allow(dead_code)]

use abi_core::GpuPtr;
use abi_core::glam::{Vec2, Vec3};
use gpu::pass::FrameAlloc;
use gpu::{Gpu, Memory};

pub const SIZE: u32 = 32;

/// Decodes IEEE-754 binary16 RGBA16F readbacks.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let out = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut fraction = fraction;
            let mut exponent = -14i32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            fraction &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exponent + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(out)
}

pub fn texel(readback: gpu::Ptr<[u16; 4]>, x: u32, y: u32) -> Vec3 {
    // SAFETY: the readback contains SIZE² RGBA16F texels.
    let raw = unsafe { *readback.cpu.add((y * SIZE + x) as usize) };
    Vec3::new(f16_to_f32(raw[0]), f16_to_f32(raw[1]), f16_to_f32(raw[2]))
}

/// Test-scoped [`FrameAlloc`] with explicit allocation cleanup.
pub struct TestAlloc<'a> {
    pub gpu: &'a Gpu,
    pub live: Vec<gpu::Ptr<u8>>,
}

impl FrameAlloc for TestAlloc<'_> {
    fn frame_alloc<T: bytemuck::Pod>(&mut self, value: T) -> GpuPtr<T> {
        let p = self.gpu.alloc::<T>(Memory::Default);
        // SAFETY: allocation is fresh and sized for T.
        unsafe { *p.cpu = value };
        self.live.push(p.cast());
        p.gpu
    }

    fn frame_alloc_slice<T: bytemuck::Pod>(&mut self, values: &[T]) -> GpuPtr<T> {
        if values.is_empty() {
            return GpuPtr::null();
        }
        let p = self
            .gpu
            .alloc_slice::<T>(values.len() as u64, Memory::Default);
        // SAFETY: allocation is fresh and covers the complete slice.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), p.cpu, values.len()) };
        self.live.push(p.cast());
        p.gpu
    }
}

impl TestAlloc<'_> {
    pub fn free(self) {
        for p in self.live {
            self.gpu.free(p);
        }
    }
}

/// Bilinear clamp-to-edge sample matching Vulkan.
pub fn bilinear(texel_at: impl Fn(u32, u32) -> Vec3, uv: Vec2) -> Vec3 {
    let t = uv * SIZE as f32 - 0.5;
    let (x0, y0) = (t.x.floor(), t.y.floor());
    let (fx, fy) = (t.x - x0, t.y - y0);
    let at = |x: f32, y: f32| {
        texel_at(
            (x as i32).clamp(0, SIZE as i32 - 1) as u32,
            (y as i32).clamp(0, SIZE as i32 - 1) as u32,
        )
    };
    let top = at(x0, y0).lerp(at(x0 + 1.0, y0), fx);
    let bottom = at(x0, y0 + 1.0).lerp(at(x0 + 1.0, y0 + 1.0), fx);
    top.lerp(bottom, fy)
}
