//! Verifies the last-workgroup two-stage reduction in one dispatch, with
//! device-scope atomic release/acquire handoff under Vulkan's memory model.
//! This workload is deliberately a coherence falsification: 2048 workgroups
//! publish full-range hashed values, so stale partial loads make the wrapping
//! sum fail rather than allowing a low-entropy test to pass by luck.

use abi_core::ReduceSingleDispatchData;
use asha_assets::load_spv;
use gpu::{Gpu, Memory, Queue};

/// Generate full-range deterministic values for the coherence workload.
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

#[test]
fn single_dispatch_reduce_matches_cpu() {
    const GROUPS: u32 = 2048;
    const COUNT: u32 = GROUPS * 64;

    let gpu = Gpu::new(true).expect("vulkan init");
    let shader = gpu.shader_create_compute(
        &load_spv("reduce_single_dispatch"),
        64,
        1,
        1,
        "reduce_single_dispatch",
    );

    let values = gpu.alloc_slice::<u32>(COUNT as u64, Memory::Default);
    let partials = gpu.alloc_slice::<u32>(GROUPS as u64, Memory::Gpu);
    let counter = gpu.alloc_slice::<u32>(1, Memory::Default);
    let result = gpu.alloc_slice::<u32>(1, Memory::Default);
    let data = gpu.alloc::<ReduceSingleDispatchData>(Memory::Default);

    let mut want = 0u32;
    // SAFETY: fresh host-visible allocations, sized above.
    unsafe {
        for i in 0..COUNT {
            let v = hash_u32(i);
            *values.cpu.add(i as usize) = v;
            want = want.wrapping_add(v);
        }
        *counter.cpu = 0;
        *result.cpu = 0xdead_beef; // Detects a missing elected writer.
        *data.cpu = ReduceSingleDispatchData {
            values: values.gpu,
            partials: partials.gpu,
            counter: counter.gpu,
            result: result.gpu,
            group_count: GROUPS,
            _pad: 0,
        };
    }

    // One dispatch, with no inter-stage global barrier, is the contract under test.
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_set_compute_shader(cb, shader);
    gpu.cmd_dispatch(cb, data.gpu, GROUPS, 1, 1);
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    // SAFETY: submit completed (wait idle above).
    let (got, elections) = unsafe { (*result.cpu, *counter.cpu) };
    assert_eq!(elections, GROUPS, "every workgroup must vote exactly once");
    assert_eq!(got, want, "elected workgroup saw stale partials");
    println!("single-dispatch reduce: {COUNT} values -> {got:#010x} == cpu, one dispatch");

    gpu.shader_destroy(shader);
    gpu.free(values);
    gpu.free(partials);
    gpu.free(counter);
    gpu.free(result);
    gpu.free(data);
}
