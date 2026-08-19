//! Verifies Vulkan initialization, allocation, and memory round trips.

use gpu::{Gpu, Memory};

#[test]
fn init_alloc_roundtrip() {
    let gpu = Gpu::new(true).expect("vulkan init");
    println!("features: {:?}", gpu.features_available());
    println!("limits:   {:?}", gpu.device_limits());

    // Default memory is mapped and addressable.
    let p = gpu.alloc_slice::<u32>(1024, Memory::Default);
    assert!(!p.is_null());
    assert!(!p.cpu.is_null());
    assert_ne!(p.gpu.addr(), 0);
    assert_eq!(p.gpu.addr() % 4, 0);

    unsafe {
        for i in 0..1024 {
            *p.cpu.add(i) = (i as u32) ^ 0xA5A5_5A5A;
        }
        for i in 0..1024 {
            assert_eq!(*p.cpu.add(i), (i as u32) ^ 0xA5A5_5A5A);
        }
    }

    // Suballocation preserves the expected offset.
    let sub = gpu.mem_suballoc(p.cast(), 256, 4, 16);
    assert_eq!(sub.gpu.addr(), p.gpu.addr() + 256);
    assert_eq!(sub.cpu as usize, p.cpu as usize + 256);

    // GPU-only memory remains device-addressable.
    let g = gpu.alloc_slice::<f32>(4096, Memory::Gpu);
    assert!(g.cpu.is_null());
    assert_ne!(g.gpu.addr(), 0);

    // Readback memory is mapped.
    let r = gpu.alloc_slice::<u8>(64, Memory::Readback);
    assert!(!r.cpu.is_null());

    // Zero-size allocations return null.
    let z = gpu.alloc_slice::<u8>(0, Memory::Default);
    assert!(z.is_null());
    gpu.free(z); // Null frees are harmless.

    gpu.free(p);
    gpu.free(g);
    gpu.free(r);
    gpu.wait_idle();
}

#[test]
fn submit_copy_chain() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let sem = gpu.semaphore_create(0);

    const COUNT: usize = 256;
    let src = gpu.alloc_slice::<u32>(COUNT as u64, Memory::Default);
    let mid = gpu.alloc_slice::<u32>(COUNT as u64, Memory::Gpu);
    let dst = gpu.alloc_slice::<u32>(COUNT as u64, Memory::Readback);
    unsafe {
        for i in 0..COUNT {
            *src.cpu.add(i) = (i as u32).wrapping_mul(2654435761);
        }
    }
    let bytes = (COUNT * 4) as u64;

    // Submission A uploads and signals value one.
    let a = gpu.commands_begin(gpu::Queue::Main);
    gpu.cmd_begin_debug_label(a, c"upload");
    gpu.cmd_checkpoint(a, c"copy src->mid");
    gpu.cmd_mem_copy_raw(a, mid.cast(), src.cast(), bytes);
    gpu.cmd_end_debug_label(a);
    gpu.cmd_add_signal_semaphore(a, sem, 1);
    gpu.queue_submit(gpu::Queue::Main, &[a]);

    // Submission B waits, then copies to readback.
    let b = gpu.commands_begin(gpu::Queue::Main);
    gpu.cmd_add_wait_semaphore(b, sem, 1);
    gpu.cmd_mem_copy_raw(b, dst.cast(), mid.cast(), bytes);
    gpu.cmd_add_signal_semaphore(b, sem, 2);
    gpu.queue_submit(gpu::Queue::Main, &[b]);

    // Wait before reading completed GPU output.
    gpu.semaphore_wait(sem, 2);
    assert_eq!(gpu.semaphore_get_value(sem), 2);
    unsafe {
        for i in 0..COUNT {
            assert_eq!(
                *dst.cpu.add(i),
                (i as u32).wrapping_mul(2654435761),
                "mismatch at {i}"
            );
        }
    }

    // Retired command buffers should be reusable.
    gpu.queue_wait_idle(gpu::Queue::Main);
    for _ in 0..8 {
        let cb = gpu.commands_begin(gpu::Queue::Main);
        gpu.queue_submit(gpu::Queue::Main, &[cb]);
        gpu.queue_wait_idle(gpu::Queue::Main);
    }

    gpu.semaphore_destroy(sem);
    gpu.free(src);
    gpu.free(mid);
    gpu.free(dst);
}
