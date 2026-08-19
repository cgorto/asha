//! Dispatch a compiled fill shader through the GPU abstraction.
//!
//! Requires prebuilt shader assets.

use abi_core::FillData;
use asha_assets::load_spv;
use gpu::{Gpu, HazardFlags, Memory, Queue, Stage};

#[test]
fn dispatch_fill() {
    let gpu = Gpu::new(true).expect("vulkan init");

    let code = load_spv("fill");
    // Workgroup size must match the compiled shader.
    let shader = gpu.shader_create_compute(&code, 64, 1, 1, "fill");

    const COUNT: u32 = 1000;
    let dst = gpu.alloc_slice::<f32>(1024, Memory::Default);
    let data = gpu.alloc::<FillData>(Memory::Default);
    unsafe {
        // ABI data matches the shader's compiled layout.
        *data.cpu = FillData {
            dst: dst.gpu,
            count: COUNT,
            value: 42.5,
        };
        for i in 0..1024 {
            *dst.cpu.add(i) = -1.0;
        }
    }

    let sem = gpu.semaphore_create(0);
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_set_compute_shader(cb, shader);
    gpu.cmd_dispatch(cb, data.gpu, COUNT.div_ceil(64), 1, 1);
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_add_signal_semaphore(cb, sem, 1);
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.semaphore_wait(sem, 1);
    gpu.queue_wait_idle(Queue::Main);

    unsafe {
        for i in 0..COUNT as usize {
            assert_eq!(*dst.cpu.add(i), 42.5, "element {i} not filled");
        }
        // Bounds checking must leave the allocation tail unchanged.
        for i in COUNT as usize..1024 {
            assert_eq!(*dst.cpu.add(i), -1.0, "element {i} written out of bounds");
        }
    }

    gpu.semaphore_destroy(sem);
    gpu.shader_destroy(shader);
    gpu.free(dst);
    gpu.free(data);
}
