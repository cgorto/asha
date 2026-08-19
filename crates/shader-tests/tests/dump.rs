use std::fs::File;
use std::io::BufReader;

use abi_core::glam::Vec3;
use gpu::{
    Gpu, HazardFlags, Memory, Queue, Stage, TextureDesc, TextureFormat, UsageFlags,
    dump_texture_png,
};

const W: u32 = 4;
const H: u32 = 4;

fn linear_texel(i: usize) -> [f32; 4] {
    let a = i as f32 / 16.0;
    let b = ((i * 5) % 16) as f32 / 16.0;
    let c = if i % 5 == 0 {
        2.0
    } else {
        (15 - i) as f32 / 16.0
    };
    let r = if i % 7 == 0 { -0.25 } else { a };
    [r, b, c, 0.5]
}

fn f32_to_f16(value: f32) -> u16 {
    assert!(value.is_finite());
    if value == 0.0 {
        return ((value.to_bits() >> 16) & 0x8000) as u16;
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x7f_ffff;
    let half_exp = exp + 15;
    assert!(
        (1..0x1f).contains(&half_exp),
        "test value must be normal f16"
    );
    assert_eq!(
        mant & 0x1fff,
        0,
        "test value must be exactly f16-representable"
    );
    sign | ((half_exp as u16) << 10) | (mant >> 13) as u16
}

fn expected_rgb(linear: [f32; 4]) -> [u8; 3] {
    let encoded = abi_post::srgb_encode(Vec3::new(
        linear[0].clamp(0.0, 1.0),
        linear[1].clamp(0.0, 1.0),
        linear[2].clamp(0.0, 1.0),
    ));
    [
        (encoded.x * 255.0).round() as u8,
        (encoded.y * 255.0).round() as u8,
        (encoded.z * 255.0).round() as u8,
    ]
}

#[test]
fn dumps_rgba16_float_texture_to_rgb_png() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let target = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba16Float,
            usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_SRC | UsageFlags::TRANSFER_DST,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let upload = gpu.alloc_slice::<[u16; 4]>((W * H) as u64, Memory::Default);
    unsafe {
        for i in 0..(W * H) as usize {
            let t = linear_texel(i);
            *upload.cpu.add(i) = [
                f32_to_f16(t[0]),
                f32_to_f16(t[1]),
                f32_to_f16(t[2]),
                f32_to_f16(t[3]),
            ];
        }
    }

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, target.texture, upload.cast());
    gpu.cmd_barrier(cb, Stage::Transfer, Stage::All, HazardFlags::empty());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let path = std::env::temp_dir().join(format!("asha_gpu_dump_{}.png", std::process::id()));
    dump_texture_png(&gpu, target.texture, &path);

    let file = File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .unwrap_or_else(|e| panic!("read info {}: {e}", path.display()));
    let mut decoded = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut decoded)
        .unwrap_or_else(|e| panic!("read frame {}: {e}", path.display()));
    let bytes = &decoded[..info.buffer_size()];

    assert_eq!(info.width, W);
    assert_eq!(info.height, H);
    assert_eq!(info.color_type, png::ColorType::Rgb);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);

    let mut expected = Vec::with_capacity((W * H * 3) as usize);
    for i in 0..(W * H) as usize {
        expected.extend_from_slice(&expected_rgb(linear_texel(i)));
    }
    assert_eq!(bytes, expected.as_slice());

    let _ = std::fs::remove_file(&path);
    gpu.free(upload);
    gpu.texture_free_and_destroy(target);
}
