//! Shared UI GPU fixtures, compositing, and readback helpers.
//!
//! Shader evaluation remains the color oracle.

#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard};

use abi_core::glam::Vec4;
use abi_ui::{UiDraw, UiVertex};
use gpu::{
    Gpu, HazardFlags, Memory, Queue, SamplerDesc, Stage, TextureDesc, TextureFormat, UsageFlags,
};
use ui::{UiBatch, UiPass, UiPassTarget, UiScissor};

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Serializes GPU device creation across test threads.
pub fn gpu_test_lock() -> MutexGuard<'static, ()> {
    GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Builds TL, TR, BR, BL vertices for an axis-aligned quad.
/// Flat fields are duplicated across all corners.
pub fn quad(
    rect: [f32; 4],
    color: Vec4,
    color2: Vec4,
    radius: Vec4,
    border: Vec4,
    flags: u32,
    uv: [[f32; 2]; 4],
) -> [UiVertex; 4] {
    let [x, y, w, h] = rect;
    let positions = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
    let points = [
        [-w / 2.0, -h / 2.0],
        [w / 2.0, -h / 2.0],
        [w / 2.0, h / 2.0],
        [-w / 2.0, h / 2.0],
    ];
    quad_raw(
        positions,
        uv,
        points,
        [w, h],
        color,
        color2,
        radius,
        border,
        flags,
    )
}

/// Builds vertices with independently supplied corner geometry.
#[allow(clippy::too_many_arguments)]
pub fn quad_raw(
    positions: [[f32; 2]; 4],
    uv: [[f32; 2]; 4],
    point: [[f32; 2]; 4],
    size: [f32; 2],
    color: Vec4,
    color2: Vec4,
    radius: Vec4,
    border: Vec4,
    flags: u32,
) -> [UiVertex; 4] {
    core::array::from_fn(|i| UiVertex {
        pos: positions[i],
        uv: uv[i],
        color: color.to_array(),
        color2: color2.to_array(),
        radius: radius.to_array(),
        border: border.to_array(),
        size,
        point: point[i],
        flags,
        tex_slot: 0,
    })
}

/// CPU equivalent of `UiPass` straight-alpha compositing.
pub fn composite(fg: Vec4, bg: Vec4) -> Vec4 {
    let a = fg.w;
    Vec4::new(
        fg.x * a + bg.x * (1.0 - a),
        fg.y * a + bg.y * (1.0 - a),
        fg.z * a + bg.z * (1.0 - a),
        a + bg.w * (1.0 - a),
    )
}

/// Renders and reads back an exact `Rgba32Float` UI target.
pub fn render_ui(
    gpu: &Gpu,
    pass: &UiPass,
    size: [u32; 2],
    clear_color: [f32; 4],
    batches: &[UiBatch],
) -> Vec<[f32; 4]> {
    let [w, h] = size;
    let color = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [w, h, 1],
            format: TextureFormat::Rgba32Float,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    // RuntimeArray descriptors must remain valid for untextured draws.
    let mut heap = gpu.heap_slots_create(2, 2, 2);
    let _sampler = heap.add_sampler(gpu, gpu.sampler_descriptor(SamplerDesc::default()));

    let read = gpu.alloc_slice::<[f32; 4]>((w * h) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    heap.bind(gpu, cb);
    pass.record(
        gpu,
        cb,
        UiPassTarget::clear(color.texture, size, clear_color),
        batches,
    );
    // Barrier all stages before transfer readback.
    gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_copy_texture_to_buffer(cb, color.texture, read.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let mut out = vec![[0.0f32; 4]; (w * h) as usize];
    // SAFETY: readback matches the copied target dimensions.
    unsafe { std::ptr::copy_nonoverlapping(read.cpu, out.as_mut_ptr(), out.len()) };

    gpu.free(read);
    heap.free(gpu);
    gpu.texture_free_and_destroy(color);
    out
}

pub fn pixel(readback: &[[f32; 4]], w: u32, x: u32, y: u32) -> Vec4 {
    Vec4::from_array(readback[(y * w + x) as usize])
}

/// Uploads quads as one draw; callers must free returned allocations.
pub struct UploadedDraw {
    pub vertices: gpu::Ptr<UiVertex>,
    pub draw: gpu::Ptr<UiDraw>,
}

impl UploadedDraw {
    pub fn free(self, gpu: &Gpu) {
        gpu.free(self.vertices);
        gpu.free(self.draw);
    }
}

pub fn upload_quads(gpu: &Gpu, quads: &[[UiVertex; 4]], view: [f32; 4]) -> UploadedDraw {
    let vertex_count = quads.len() * 4;
    let vertices = gpu.alloc_slice::<UiVertex>(vertex_count as u64, Memory::Default);
    // SAFETY: allocation matches the vertex count.
    unsafe {
        for (i, v) in quads.iter().flatten().enumerate() {
            *vertices.cpu.add(i) = *v;
        }
    }
    let draw = gpu.alloc::<UiDraw>(Memory::Default);
    // SAFETY: allocation holds exactly one UiDraw.
    unsafe {
        *draw.cpu = UiDraw {
            vertices: vertices.gpu,
            view,
            quad_count: quads.len() as u32,
            sampler_slot: 0,
        };
    }
    UploadedDraw { vertices, draw }
}

/// `clip = px * view.xy + view.zw`, top-left origin, +y down.
pub fn view_for(size: [u32; 2]) -> [f32; 4] {
    [2.0 / size[0] as f32, 2.0 / size[1] as f32, -1.0, -1.0]
}

pub fn one_batch(
    draw: gpu::Ptr<UiDraw>,
    quad_count: u32,
    scissor: Option<UiScissor>,
) -> [UiBatch; 1] {
    [UiBatch {
        draw: draw.gpu,
        quad_count,
        scissor,
    }]
}
