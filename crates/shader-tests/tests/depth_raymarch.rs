//! Classifies projected segments against reverse-Z depth.

use abi_light::{
    DEPTH_MARCH_MAX_STEPS, DepthMarchConfig, DepthMarchData, DepthMarchQuery, DepthMarchResult,
    depth_march_config_valid,
};
use asha_assets::load_spv;
use gpu::{
    AllocationType, Gpu, GpuPtr, HazardFlags, Memory, Queue, Stage, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};

const W: u32 = 16;
const H: u32 = 8;
const DEPTH_TEX_IDX: u32 = 3;
const QUERY_COUNT: u32 = 5; // Exercises the partial workgroup tail.

fn query(uv: [f32; 2], depth: f32) -> DepthMarchQuery {
    let ndc = [uv[0] * 2.0 - 1.0, uv[1] * 2.0 - 1.0, depth];
    DepthMarchQuery {
        start_ndc: ndc,
        _pad0: 0.0,
        end_ndc: ndc,
        _pad1: 0.0,
    }
}

#[test]
fn reverse_z_depth_march_classifies_controlled_segments() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let shader = gpu.shader_create_compute(
        &load_spv("depth_raymarch_queries"),
        64,
        1,
        1,
        "depth_raymarch_queries",
    );

    // Include a depth discontinuity to test point sampling.
    let depth_desc = TextureDesc {
        dimensions: [W, H, 1],
        format: TextureFormat::R32Float,
        usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
        ..Default::default()
    };
    let (depth_bytes, depth_align) = gpu.texture_size_and_align(depth_desc);
    let depth_mem = gpu.mem_alloc_raw(
        depth_bytes,
        1,
        depth_align,
        Memory::Gpu,
        AllocationType::Default,
    );
    let depth_tex = gpu.texture_create(depth_desc, depth_mem, Queue::Main, None);
    let depth_upload = gpu.alloc_slice::<f32>((W * H) as u64, Memory::Default);
    unsafe {
        for y in 0..H {
            for x in 0..W {
                *depth_upload.cpu.add((y * W + x) as usize) =
                    if y < H / 2 || x < W / 2 { 0.5 } else { 0.1 };
            }
        }
    }

    let descriptor_bytes = gpu.texture_view_descriptor_size() as u64;
    let texture_heap = gpu.mem_alloc_raw(
        descriptor_bytes * 8,
        64,
        256,
        Memory::Default,
        AllocationType::Descriptors,
    );
    gpu.set_texture_desc(
        texture_heap,
        DEPTH_TEX_IDX,
        gpu.texture_view_descriptor(depth_tex, TextureViewDesc::default()),
    );

    let queries = gpu.alloc_slice::<DepthMarchQuery>(QUERY_COUNT as u64, Memory::Default);
    let results = gpu.alloc_slice::<DepthMarchResult>(QUERY_COUNT as u64, Memory::Default);
    let data = gpu.alloc::<DepthMarchData>(Memory::Default);
    let config = DepthMarchConfig {
        linear_steps: 16,
        continue_after_deep_penetration: 1,
        jitter: 1.0,
        depth_thickness: 1.0,
        near_plane: 1.0,
        _pad: [0; 3],
    };
    assert!(depth_march_config_valid(&config));
    assert!(!depth_march_config_valid(&DepthMarchConfig {
        jitter: -0.01,
        ..config
    }));
    assert!(!depth_march_config_valid(&DepthMarchConfig {
        linear_steps: DEPTH_MARCH_MAX_STEPS + 1,
        ..config
    }));

    unsafe {
        // 0: front segment remains unresolved.
        *queries.cpu.add(0) = query([0.25, 0.25], 0.75);
        // 1: crossing segment produces a hit.
        *queries.cpu.add(1) = DepthMarchQuery {
            start_ndc: [-0.75, -0.5, 0.75],
            _pad0: 0.0,
            end_ndc: [0.75, -0.5, 0.25],
            _pad1: 0.0,
        };
        // 2: reject a bilinear false hit at the discontinuity.
        *queries.cpu.add(2) = query([0.5, 0.75], 0.2);
        // 3: reject excessive penetration.
        *queries.cpu.add(3) = query([0.25, 0.25], 0.05);
        // 4: accept penetration within thickness.
        *queries.cpu.add(4) = query([0.25, 0.25], 0.4);

        for i in 0..QUERY_COUNT as usize {
            *results.cpu.add(i) = DepthMarchResult {
                hit: 0xDEAD_BEEF,
                _pad0: 0xDEAD_BEEF,
                hit_t: -1.0,
                hit_penetration: -1.0,
                hit_uv: [-1.0; 2],
                _pad1: [0xDEAD_BEEF; 2],
            };
        }
        *data.cpu = DepthMarchData {
            queries: queries.gpu,
            results: results.gpu,
            depth_texture_id: DEPTH_TEX_IDX,
            query_count: QUERY_COUNT,
            depth_size: [W, H],
            config,
        };
    }

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, depth_tex, depth_upload.cast());
    gpu.cmd_barrier(cb, Stage::Transfer, Stage::Compute, HazardFlags::empty());
    gpu.cmd_set_desc_heap(cb, texture_heap.gpu, GpuPtr::null(), GpuPtr::null());
    gpu.cmd_set_compute_shader(cb, shader);
    gpu.cmd_dispatch(cb, data.gpu, QUERY_COUNT.div_ceil(64), 1, 1);
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let got: Vec<DepthMarchResult> = (0..QUERY_COUNT as usize)
        .map(|i| unsafe { *results.cpu.add(i) })
        .collect();
    for (i, result) in got.iter().enumerate() {
        assert_eq!(result._pad0, 0, "query {i} was not fully written");
        assert_eq!(result._pad1, [0; 2], "query {i} was not fully written");
        assert!((0.0..=1.0).contains(&result.hit_t), "query {i} hit_t");
        assert!(
            (0.0..=1.0).contains(&result.hit_uv[0]) && (0.0..=1.0).contains(&result.hit_uv[1]),
            "query {i} hit_uv"
        );
    }
    assert_eq!(got[0].hit, 0, "front segment must remain unresolved");
    assert_eq!(got[1].hit, 1, "crossing segment must hit");
    assert!((0.4..0.9).contains(&got[1].hit_t));
    assert_eq!(got[2].hit, 0, "bilinear shrink-wrap false hit");
    assert_eq!(got[3].hit, 0, "excess penetration must be rejected");
    assert_eq!(got[4].hit, 1, "near-behind segment must hit");
    assert!(got[4].hit_penetration > 0.0 && got[4].hit_penetration < 1.0);

    gpu.texture_destroy(depth_tex);
    gpu.shader_destroy(shader);
    gpu.free(data);
    gpu.free(queries);
    gpu.free(results);
    gpu.free(depth_upload);
    gpu.mem_free_raw(texture_heap);
    gpu.mem_free_raw(depth_mem);
}
