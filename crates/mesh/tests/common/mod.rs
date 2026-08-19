//! Shared fixtures for the mesh GPU test binaries.

// Shared across test binaries; not every binary uses every helper.
#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard};

use abi_core::GpuPtr;
use abi_core::View;
use abi_core::glam::{UVec2, Vec3};
use gpu::{Gpu, Memory};
use mesh::FrameAlloc;

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

pub fn gpu_test_lock() -> MutexGuard<'static, ()> {
    GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub struct TestFrameAlloc<'a> {
    pub gpu: &'a Gpu,
    pub ptrs: Vec<gpu::Ptr<u8>>,
}

impl FrameAlloc for TestFrameAlloc<'_> {
    fn frame_alloc<T: bytemuck::Pod>(&mut self, value: T) -> GpuPtr<T> {
        let ptr = self.gpu.alloc::<T>(Memory::Default);
        // SAFETY: fresh host-visible allocation sized for T.
        unsafe { *ptr.cpu = value };
        self.ptrs.push(ptr.cast());
        ptr.gpu
    }

    fn frame_alloc_slice<T: bytemuck::Pod>(&mut self, values: &[T]) -> GpuPtr<T> {
        if values.is_empty() {
            return GpuPtr::null();
        }
        let ptr = self
            .gpu
            .alloc_slice::<T>(values.len() as u64, Memory::Default);
        // SAFETY: fresh host-visible allocation sized for the complete slice.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), ptr.cpu, values.len()) };
        self.ptrs.push(ptr.cast());
        ptr.gpu
    }
}

impl TestFrameAlloc<'_> {
    pub fn free(self) {
        for ptr in self.ptrs {
            self.gpu.free(ptr);
        }
    }
}

pub fn view(size: UVec2) -> View {
    View {
        camera_position: [0.0, 0.0, -8.0],
        tan_half_fov: 0.55,
        camera_forward: Vec3::Z.to_array(),
        aspect: size.x as f32 / size.y as f32,
        camera_right: Vec3::NEG_X.to_array(),
        depth_near_plane: 0.1,
        camera_up: Vec3::Y.to_array(),
        _pad: 0,
        output_size: size.to_array(),
        _pad2: [0; 2],
    }
}

pub fn mesh_heap(gpu: &Gpu) -> (gpu::HeapSlots, gpu::SamplerSlot) {
    let mut heap = gpu.heap_slots_create(2, 2, 2);
    let sampler = heap.add_sampler(gpu, gpu.sampler_descriptor(gpu::SamplerDesc::default()));
    (heap, sampler)
}
