//! Verifies the 256-lane Blelloch exclusive scan over per-tile light counts.
//! It runs both a 2040-tile 1080p-shaped case, exercising the zero-filled
//! tail, and exactly MAX_TILES. Exact integer CPU comparisons cover every
//! offset and the grand total.

use abi_light::{MAX_TILES, PrefixSumData, TileHeader};
use asha_assets::load_spv;
use gpu::{Gpu, HazardFlags, Memory, Queue, Stage};

fn light_count(i: u32, salt: u32) -> u32 {
    i.wrapping_mul(2654435761).wrapping_add(salt) % 38 // 0..=37 lights per tile
}

struct Case {
    tile_count: u32,
    salt: u32,
    headers: gpu::Ptr<TileHeader>,
    total: gpu::Ptr<u32>,
    data: gpu::Ptr<PrefixSumData>,
}

#[test]
fn prefix_sum_matches_cpu() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let shader = gpu.shader_create_compute(&load_spv("prefix_sum"), 256, 1, 1, "prefix_sum");

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_set_compute_shader(cb, shader);

    let mut cases = Vec::new();
    for (tile_count, salt) in [(2040u32, 7u32), (MAX_TILES, 1312u32)] {
        let headers = gpu.alloc_slice::<TileHeader>(tile_count as u64, Memory::Default);
        let total = gpu.alloc::<u32>(Memory::Default);
        let data = gpu.alloc::<PrefixSumData>(Memory::Default);
        unsafe {
            for i in 0..tile_count {
                *headers.cpu.add(i as usize) = TileHeader {
                    light_count: light_count(i, salt),
                    light_offset: 0xDEAD_BEEF, // Detects unwritten offsets.
                };
            }
            *total.cpu = 0xDEAD_BEEF;
            *data.cpu = PrefixSumData {
                tile_headers: headers.gpu,
                total_light_count: total.gpu,
                tile_count,
            };
        }
        gpu.cmd_dispatch(cb, data.gpu, 1, 1, 1);
        cases.push(Case {
            tile_count,
            salt,
            headers,
            total,
            data,
        });
    }

    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    for case in &cases {
        let mut running = 0u64;
        for i in 0..case.tile_count {
            let got = unsafe { *case.headers.cpu.add(i as usize) };
            assert_eq!(
                got.light_offset, running as u32,
                "tile_count={} tile {i} offset",
                case.tile_count
            );
            assert_eq!(
                got.light_count,
                light_count(i, case.salt),
                "input clobbered at {i}"
            );
            running += got.light_count as u64;
        }
        let got_total = unsafe { *case.total.cpu };
        assert_eq!(
            got_total as u64, running,
            "tile_count={} grand total",
            case.tile_count
        );
    }

    gpu.shader_destroy(shader);
    for case in cases {
        gpu.free(case.headers);
        gpu.free(case.total);
        gpu.free(case.data);
    }
}
