//! Verifies per-tile reverse-Z min/max reduction using workgroup shared
//! memory, barriers, and the bindless sampled heap (set 0). The 100×70
//! texture deliberately creates partial right/bottom tiles; out-of-bounds
//! lanes contribute 0.0, the infinite-far sentinel, and results match an
//! exact CPU reference.

use abi_light::{DepthReduceData, TILE_SIZE, TileDepthBounds};
use asha_assets::load_spv;
use gpu::{
    AllocationType, Gpu, GpuPtr, HazardFlags, Memory, Queue, Stage, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};

const W: u32 = 100; // Partial right and bottom tiles exercise OOB lanes.
const H: u32 = 70;
const TILES_X: u32 = W.div_ceil(TILE_SIZE);
const TILES_Y: u32 = H.div_ceil(TILE_SIZE);
const SCENE_TEX_IDX: u32 = 3; // Slot zero is the disabled sentinel.

/// Values stay in (0, 1] so real pixels are distinct from OOB reverse-Z 0.0.
fn scene_depth(x: u32, y: u32) -> f32 {
    ((x * 31 + y * 17) % 89 + 1) as f32 / 90.0
}

/// Exact CPU reference, including OOB lanes contributing reverse-Z zero.
fn cpu_reference() -> Vec<TileDepthBounds> {
    let mut out = Vec::new();
    for ty in 0..TILES_Y {
        for tx in 0..TILES_X {
            let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
            for ly in 0..TILE_SIZE {
                for lx in 0..TILE_SIZE {
                    let (x, y) = (tx * TILE_SIZE + lx, ty * TILE_SIZE + ly);
                    let d = if x < W && y < H {
                        scene_depth(x, y)
                    } else {
                        0.0
                    };
                    mn = mn.min(d);
                    mx = mx.max(d);
                }
            }
            out.push(TileDepthBounds {
                min_depth: mn,
                max_depth: mx,
            });
        }
    }
    out
}

#[test]
fn depth_reduce_matches_cpu() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let shader = gpu.shader_create_compute(
        &load_spv("depth_reduce"),
        TILE_SIZE,
        TILE_SIZE,
        1,
        "depth_reduce",
    );

    let desc = TextureDesc {
        dimensions: [W, H, 1],
        format: TextureFormat::R32Float,
        usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
        ..Default::default()
    };
    let (size, align) = gpu.texture_size_and_align(desc);
    let scene_mem = gpu.mem_alloc_raw(size, 1, align, Memory::Gpu, AllocationType::Default);
    let scene_tex = gpu.texture_create(desc, scene_mem, Queue::Main, None);

    let scene_up = gpu.alloc_slice::<f32>((W * H) as u64, Memory::Default);
    unsafe {
        for y in 0..H {
            for x in 0..W {
                *scene_up.cpu.add((y * W + x) as usize) = scene_depth(x, y);
            }
        }
    }

    // Set 0 sampled heap: slot zero stays reserved; use an arbitrary nonzero ID.
    let desc_size = gpu.texture_view_descriptor_size() as u64;
    let heap = gpu.mem_alloc_raw(
        desc_size * 8,
        64,
        256,
        Memory::Default,
        AllocationType::Descriptors,
    );
    gpu.set_texture_desc(
        heap,
        SCENE_TEX_IDX,
        gpu.texture_view_descriptor(scene_tex, TextureViewDesc::default()),
    );

    let tiles = (TILES_X * TILES_Y) as u64;
    let bounds = gpu.alloc_slice::<TileDepthBounds>(tiles, Memory::Default);
    let data = gpu.alloc::<DepthReduceData>(Memory::Default);
    unsafe {
        for i in 0..tiles as usize {
            // Sentinel detects unwritten tile results.
            *bounds.cpu.add(i) = TileDepthBounds {
                min_depth: -9.0,
                max_depth: -9.0,
            };
        }
        *data.cpu = DepthReduceData {
            tile_depth_bounds: bounds.gpu,
            depth_texture_id: SCENE_TEX_IDX,
            screen_size: [W, H],
            tile_count: [TILES_X, TILES_Y],
        };
    }

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, scene_tex, scene_up.cast());
    gpu.cmd_barrier(cb, Stage::Transfer, Stage::Compute, HazardFlags::empty());
    gpu.cmd_set_desc_heap(cb, heap.gpu, GpuPtr::null(), GpuPtr::null());
    gpu.cmd_set_compute_shader(cb, shader);
    gpu.cmd_dispatch(cb, data.gpu, TILES_X, TILES_Y, 1);
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let expected = cpu_reference();
    for (i, exp) in expected.iter().enumerate() {
        let got = unsafe { *bounds.cpu.add(i) };
        assert_eq!(got.min_depth, exp.min_depth, "tile {i} min");
        assert_eq!(got.max_depth, exp.max_depth, "tile {i} max");
    }

    gpu.texture_destroy(scene_tex);
    gpu.shader_destroy(shader);
    gpu.free(data);
    gpu.free(bounds);
    gpu.free(scene_up);
    gpu.mem_free_raw(heap);
    gpu.mem_free_raw(scene_mem);
}
