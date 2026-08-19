//! Smoke harness for the Slug text pass.
//!
//! Uses synthetic ABI buffers and verifies covered and clear pixels.

use abi_ui::{
    TextBandHeader, TextCamera, TextCurve, TextDraw, TextGlyphDescriptor, TextGlyphInstance,
};
use gpu::{
    Gpu, HazardFlags, LoadOp, Memory, Queue, Stage, StoreOp, TextureDesc, TextureFormat, UsageFlags,
};
use text::{TextBatch, TextPass, TextPassTarget};

const W: u32 = 256;
const H: u32 = 128;
const CLEAR: [f32; 4] = [0.05, 0.10, 0.15, 1.0];

fn main() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let pass = TextPass::new(&gpu);

    let target = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba8Unorm,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    let instances = gpu.alloc_slice::<TextGlyphInstance>(1, Memory::Default);
    let descriptors = gpu.alloc_slice::<TextGlyphDescriptor>(1, Memory::Default);
    let curves = gpu.alloc_slice::<TextCurve>(4, Memory::Default);
    let bands = gpu.alloc_slice::<TextBandHeader>(2, Memory::Default);
    let band_curve_indices = gpu.alloc_slice::<u32>(4, Memory::Default);
    let draw = gpu.alloc::<TextDraw>(Memory::Default);
    let readback = gpu.alloc_slice::<u8>((W * H * 4) as u64, Memory::Readback);

    unsafe {
        *instances.cpu = TextGlyphInstance {
            pen_doc: [48.0, 88.0],
            glyph_id: 0,
            color: 0xffff_ffff,
        };
        *descriptors.cpu = TextGlyphDescriptor {
            bbox_em: [0.0, 0.0, 1.0, 1.0],
            band_scale: [1.0, 1.0],
            band_offset: [0.0, 0.0],
            hband_base: 0,
            vband_base: 1,
            band_max: 0,
            _pad0: 0,
        };
        // Unit square; degenerate quadratics represent line segments.
        *curves.cpu.add(0) = TextCurve {
            p1: [0.0, 0.0],
            p2: [0.0, 1.0],
            p3: [0.0, 1.0],
        };
        *curves.cpu.add(1) = TextCurve {
            p1: [0.0, 1.0],
            p2: [1.0, 1.0],
            p3: [1.0, 1.0],
        };
        *curves.cpu.add(2) = TextCurve {
            p1: [1.0, 1.0],
            p2: [1.0, 0.0],
            p3: [1.0, 0.0],
        };
        *curves.cpu.add(3) = TextCurve {
            p1: [1.0, 0.0],
            p2: [0.0, 0.0],
            p3: [0.0, 0.0],
        };
        // Horizontal band: vertical curves, descending maximum x.
        *bands.cpu.add(0) = TextBandHeader { first: 0, count: 2 };
        *band_curve_indices.cpu.add(0) = 2;
        *band_curve_indices.cpu.add(1) = 0;
        // Vertical band: horizontal curves, descending maximum y.
        *bands.cpu.add(1) = TextBandHeader { first: 2, count: 2 };
        *band_curve_indices.cpu.add(2) = 1;
        *band_curve_indices.cpu.add(3) = 3;
        *draw.cpu = TextDraw {
            instances: instances.gpu,
            descriptors: descriptors.gpu,
            curves: curves.gpu,
            bands: bands.gpu,
            band_curve_indices: band_curve_indices.gpu,
            camera: TextCamera {
                xform: [2.0 / W as f32, -2.0 / H as f32, -1.0, 1.0],
                zoom: 1.0,
                font_px_per_em: 48.0,
                _pad0: [0.0; 2],
            },
            glyph_count: 1,
            flags: 0,
            _pad0: [0; 2],
        };
    }

    let cb = gpu.commands_begin(Queue::Main);
    pass.record(
        &gpu,
        cb,
        TextPassTarget {
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color: CLEAR,
            ..TextPassTarget::overlay(target.texture, [W, H])
        },
        TextBatch {
            draw: draw.gpu.cast(),
            glyph_count: 1,
        },
    );
    gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_copy_texture_to_buffer(cb, target.texture, readback.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let pixel = |x: u32, y: u32| -> [u8; 4] {
        unsafe {
            let p = readback.cpu.add(((y * W + x) * 4) as usize);
            [*p, *p.add(1), *p.add(2), *p.add(3)]
        }
    };
    let center = pixel(72, 64);
    assert!(
        center[0] > 180 && center[1] > 180 && center[2] > 180 && center[3] > 240,
        "text center pixel should be covered white, got {center:?}",
    );
    let corner = pixel(8, 8);
    assert!(
        (corner[0] as i32 - (CLEAR[0] * 255.0) as i32).abs() <= 2
            && (corner[1] as i32 - (CLEAR[1] * 255.0) as i32).abs() <= 2
            && (corner[2] as i32 - (CLEAR[2] * 255.0) as i32).abs() <= 2,
        "outside glyph should stay clear, got {corner:?}",
    );
    println!("TEXT VERIFY OK center={center:?} corner={corner:?}");

    pass.free(&gpu);
    gpu.texture_free_and_destroy(target);
    for ptr in [
        instances.cast::<u8>(),
        descriptors.cast(),
        curves.cast(),
        bands.cast(),
        band_curve_indices.cast(),
        draw.cast(),
        readback.cast(),
    ] {
        gpu.free(ptr);
    }
}
