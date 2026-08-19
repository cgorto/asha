use std::sync::{Mutex, MutexGuard};

use abi_core::GpuPtr;
use abi_core::glam::{UVec2, Vec3, Vec4};
use abi_core::{View, ray_direction};
use abi_light::PointLight;
use abi_light::{
    EXT_MAX, FOG_LIGHT_TILE, FogCurve, FroxelParams, OitParticle, extinction_decode,
    extinction_encode, extinction_to_u8, fog_light_tile_bounds, fog_point_light_radiance,
    froxel_params_from, height_fog_optical_depth, height_gradient, hg_phase, integrate_step,
    interleaved_gradient_noise, oit_resolve, splat_weights, transmittance, warped_slice_of_z,
    z_of_warped_slice,
};
use gpu::pass::{FrameAlloc, Pass as _};
use gpu::{
    Gpu, HazardFlags, LoadOp, Queue, RenderAttachment, RenderPassDesc, SamplerDesc, Stage, StoreOp,
    TextureDesc, TextureFormat, TextureViewDesc, UsageFlags,
};
use post::volumetrics::{
    FROXEL_DEPTH, FROXEL_HEIGHT, FROXEL_WIDTH, FogDials, FogLightInputs, OccluderVolume,
    VolumetricPasses,
};

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

fn gpu_test_lock() -> MutexGuard<'static, ()> {
    GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct TestFrameAlloc<'a> {
    gpu: &'a Gpu,
    ptrs: Vec<gpu::Ptr<u8>>,
}

impl FrameAlloc for TestFrameAlloc<'_> {
    fn frame_alloc<T: bytemuck::Pod>(&mut self, value: T) -> abi_core::GpuPtr<T> {
        let ptr = self.gpu.alloc::<T>(gpu::Memory::Default);
        // SAFETY: fresh allocation sized for T.
        unsafe { std::ptr::write(ptr.cpu, value) };
        self.ptrs.push(ptr.cast());
        ptr.gpu
    }

    fn frame_alloc_slice<T: bytemuck::Pod>(&mut self, values: &[T]) -> abi_core::GpuPtr<T> {
        if values.is_empty() {
            return abi_core::GpuPtr::null();
        }
        let ptr = self
            .gpu
            .alloc_slice::<T>(values.len() as u64, gpu::Memory::Default);
        // SAFETY: fresh allocation covers the complete slice.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), ptr.cpu, values.len()) };
        self.ptrs.push(ptr.cast());
        ptr.gpu
    }
}

impl TestFrameAlloc<'_> {
    fn free(self) {
        for ptr in self.ptrs {
            self.gpu.free(ptr);
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let f = match exp {
        0 => (mant as f32) * 2.0f32.powi(-24),
        31 => {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15),
    };
    if sign != 0 { -f } else { f }
}

fn view(size: UVec2) -> View {
    View {
        camera_position: [0.0, 2.5, -12.0],
        tan_half_fov: 0.58,
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

/// Shared sun fixture for GPU inputs and CPU references.
///
/// Normalization differences remain within pixel tolerances.
fn test_sun_dir() -> Vec3 {
    Vec3::new(0.3, 0.9, 0.25).normalize()
}
const TEST_SUN_COLOR: [f32; 3] = [0.55, 0.48, 0.40];

fn test_light_inputs() -> FogLightInputs {
    FogLightInputs {
        sun_dir: test_sun_dir().to_array(),
        sun_color: TEST_SUN_COLOR,
        occluder: None,
        local_lights: GpuPtr::null(),
        local_light_count: 0,
    }
}

/// CPU reference for the sampled occluder volume.
struct CpuOccluder {
    dims: UVec2,
    depth: u32,
    data: Vec<u8>,
    world_min: Vec3,
    world_inv_extent: Vec3,
}

impl CpuOccluder {
    fn value(&self, x: u32, y: u32, z: u32) -> f32 {
        self.data[(x + y * self.dims.x + z * self.dims.x * self.dims.y) as usize] as f32 / 255.0
    }

    fn sample_trilinear(&self, uvw: Vec3) -> f32 {
        let (x0, x1, fx) = axis_lerp(uvw.x, self.dims.x);
        let (y0, y1, fy) = axis_lerp(uvw.y, self.dims.y);
        let (z0, z1, fz) = axis_lerp(uvw.z, self.depth);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let x00 = lerp(self.value(x0, y0, z0), self.value(x1, y0, z0), fx);
        let x10 = lerp(self.value(x0, y1, z0), self.value(x1, y1, z0), fx);
        let x01 = lerp(self.value(x0, y0, z1), self.value(x1, y0, z1), fx);
        let x11 = lerp(self.value(x0, y1, z1), self.value(x1, y1, z1), fx);
        lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz)
    }
}

/// CPU lighting inputs matching the GPU's sampled data.
struct TwinLight<'a> {
    sun_dir: Vec3,
    sun_color: Vec3,
    occluder: Option<&'a CpuOccluder>,
    steps: u32,
    local_lights: &'a [PointLight],
}

fn twin_from<'a>(
    inputs: &FogLightInputs,
    occluder: Option<&'a CpuOccluder>,
    steps: u32,
) -> TwinLight<'a> {
    twin_from_local(inputs, occluder, steps, &[])
}

fn twin_from_local<'a>(
    inputs: &FogLightInputs,
    occluder: Option<&'a CpuOccluder>,
    steps: u32,
    local_lights: &'a [PointLight],
) -> TwinLight<'a> {
    TwinLight {
        sun_dir: Vec3::from_array(inputs.sun_dir).normalize(),
        sun_color: Vec3::from_array(inputs.sun_color),
        occluder,
        steps,
        local_lights,
    }
}

/// CPU reference for the occluder march.
///
/// Tap spacing doubles; out-of-bounds taps contribute nothing.
fn cpu_occluder_visibility(occ: &CpuOccluder, steps: u32, pos: Vec3, sun_dir: Vec3) -> f32 {
    let inv_abs = occ.world_inv_extent.abs();
    let extent = Vec3::new(
        1.0 / inv_abs.x.max(1.0e-6),
        1.0 / inv_abs.y.max(1.0e-6),
        1.0 / inv_abs.z.max(1.0e-6),
    );
    let max_extent = extent.x.max(extent.y).max(extent.z);
    let mut denom = 0.0f32;
    let mut spacing = 1.0f32;
    for _ in 0..steps {
        denom += spacing;
        spacing *= 2.0;
    }
    let mut visibility = 1.0f32;
    let mut tap_spacing = max_extent / denom.max(1.0);
    let mut t = 0.0f32;
    for _ in 0..steps {
        t += tap_spacing;
        let uvw = (pos + sun_dir * t - occ.world_min) * occ.world_inv_extent;
        let inside = uvw.x >= 0.0
            && uvw.y >= 0.0
            && uvw.z >= 0.0
            && uvw.x <= 1.0
            && uvw.y <= 1.0
            && uvw.z <= 1.0;
        if inside {
            visibility = (visibility * (1.0 - occ.sample_trilinear(uvw).clamp(0.0, 1.0))).max(0.0);
            if visibility <= 0.0 {
                break;
            }
        }
        tap_spacing *= 2.0;
    }
    visibility
}

/// CPU reference for `fog_light` per-slice evaluation.
fn cpu_slice_light(
    params: &FogCurve,
    dials: &FogDials,
    light: &TwinLight,
    camera: Vec3,
    dir: Vec3,
    view_to_ray: f32,
    i: u32,
) -> ([f32; 3], f32) {
    let z_mid = z_of_warped_slice(params, i as f32 + 0.5);
    let pos = camera + dir * (z_mid * view_to_ray);
    let h = pos.y;
    let density = dials.density.max(0.0);
    let falloff = dials.height_falloff.max(0.0);
    let sigma_t = density * (-(falloff * (h - dials.height_offset).max(0.0))).exp();
    let self_shadow = transmittance(height_fog_optical_depth(
        h,
        light.sun_dir.y,
        0.0,
        f32::INFINITY,
        density,
        falloff,
        dials.height_offset,
    ));
    let occ_vis = match light.occluder {
        Some(occ) if light.steps > 0 => {
            cpu_occluder_visibility(occ, light.steps, pos, light.sun_dir)
        }
        _ => 1.0,
    };
    let phase = hg_phase(dir.dot(light.sun_dir), dials.anisotropy);
    let tint = Vec3::from_array(height_gradient(
        h,
        dials.gradient_bottom,
        dials.gradient_top,
        dials.gradient_offset,
        dials.gradient_length,
    ));
    let mut local = Vec3::ZERO;
    for point in light.local_lights {
        local += fog_point_light_radiance(
            dir,
            pos,
            point,
            dials.anisotropy,
            density,
            falloff,
            dials.height_offset,
        );
    }
    let lighting = (light.sun_color * (phase * self_shadow * occ_vis)
        + Vec3::from_array(dials.ambient_color)
        + local)
        * tint;
    ((lighting * sigma_t).to_array(), sigma_t)
}

fn setup_targets(
    gpu: &Gpu,
    heap: &mut gpu::HeapSlots,
    size: UVec2,
) -> (
    gpu::OwnedTexture,
    gpu::OwnedTexture,
    gpu::SampledSlot,
    gpu::StorageSlot,
) {
    let hdr = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [size.x, size.y, 1],
            format: TextureFormat::Rgba16Float,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::STORAGE | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let depth = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [size.x, size.y, 1],
            format: TextureFormat::D32Float,
            usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT
                | UsageFlags::SAMPLED
                | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let depth_slot = heap.add_sampled(
        gpu,
        gpu.texture_view_descriptor(depth.texture, TextureViewDesc::default()),
    );
    let hdr_rw = heap.add_storage(
        gpu,
        gpu.texture_rw_view_descriptor(hdr.texture, TextureViewDesc::default()),
    );
    (hdr, depth, depth_slot, hdr_rw)
}

fn clamp_sampler(gpu: &Gpu, heap: &mut gpu::HeapSlots) -> gpu::SamplerSlot {
    heap.add_sampler(
        gpu,
        gpu.sampler_descriptor(SamplerDesc {
            address_mode_u: gpu::AddressMode::ClampToEdge,
            address_mode_v: gpu::AddressMode::ClampToEdge,
            address_mode_w: gpu::AddressMode::ClampToEdge,
            ..Default::default()
        }),
    )
}

fn upload_occluder_texture(
    gpu: &Gpu,
    heap: &mut gpu::HeapSlots,
    dims: UVec2,
    depth: u32,
    data: &[u8],
) -> (gpu::OwnedTexture, gpu::SampledSlot) {
    assert_eq!(data.len(), (dims.x * dims.y * depth) as usize);
    let texture = gpu.texture_alloc_and_create(
        TextureDesc {
            ty: gpu::TextureType::D3,
            dimensions: [dims.x, dims.y, depth],
            mip_count: 1,
            format: TextureFormat::R8Unorm,
            usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let slot = heap.add_sampled(
        gpu,
        gpu.texture_view_descriptor(
            texture.texture,
            TextureViewDesc {
                ty: gpu::TextureType::D3,
                ..Default::default()
            },
        ),
    );
    let upload = gpu.alloc_slice::<u8>(data.len() as u64, gpu::Memory::Default);
    // SAFETY: staging allocation matches the source slice.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), upload.cpu, data.len());
    }

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, texture.texture, upload.cast());
    gpu.cmd_barrier(cb, Stage::Transfer, Stage::Compute, HazardFlags::empty());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);
    gpu.free(upload);

    (texture, slot)
}

fn axis_lerp(coord: f32, dim: u32) -> (u32, u32, f32) {
    let t = coord * dim as f32 - 0.5;
    let base = t.floor();
    let frac = t - base;
    let i0 = (base as i32).clamp(0, dim as i32 - 1) as u32;
    let i1 = (base as i32 + 1).clamp(0, dim as i32 - 1) as u32;
    (i0, i1, frac)
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

fn cpu_v_int_texel(
    params: &FogCurve,
    view: &View,
    dials: &FogDials,
    light: &TwinLight,
    x: u32,
    y: u32,
    z: u32,
) -> [f32; 4] {
    let dir = ray_direction(view, UVec2::new(x, y));
    let forward = Vec3::from_array(view.camera_forward);
    let view_to_ray = 1.0 / forward.dot(dir).max(1.0e-6);
    let camera = Vec3::from_array(view.camera_position);
    let mut luminance = Vec3::ZERO;
    let mut throughput = 1.0f32;
    for i in 0..=z {
        let z0 = z_of_warped_slice(params, i as f32);
        let z1 = z_of_warped_slice(params, i as f32 + 1.0);
        let (scatter, sigma_t) = cpu_slice_light(params, dials, light, camera, dir, view_to_ray, i);
        let (added, step_t) =
            integrate_step(scatter, sigma_t, 0.0, (z1 - z0) * view_to_ray, throughput);
        luminance += Vec3::from_array(added);
        throughput *= step_t;
    }
    [luminance.x, luminance.y, luminance.z, throughput]
}

fn cpu_v_int_sample(
    params: &FogCurve,
    view: &View,
    dials: &FogDials,
    light: &TwinLight,
    uvw: Vec3,
) -> [f32; 4] {
    let (x0, x1, fx) = axis_lerp(uvw.x, FROXEL_WIDTH);
    let (y0, y1, fy) = axis_lerp(uvw.y, FROXEL_HEIGHT);
    let (z0, z1, fz) = axis_lerp(uvw.z, FROXEL_DEPTH);
    let sample = |x, y, z| cpu_v_int_texel(params, view, dials, light, x, y, z);

    let c000 = sample(x0, y0, z0);
    let c100 = sample(x1, y0, z0);
    let c010 = sample(x0, y1, z0);
    let c110 = sample(x1, y1, z0);
    let c001 = sample(x0, y0, z1);
    let c101 = sample(x1, y0, z1);
    let c011 = sample(x0, y1, z1);
    let c111 = sample(x1, y1, z1);

    let x00 = lerp4(c000, c100, fx);
    let x10 = lerp4(c010, c110, fx);
    let x01 = lerp4(c001, c101, fx);
    let x11 = lerp4(c011, c111, fx);
    let y0 = lerp4(x00, x10, fy);
    let y1 = lerp4(x01, x11, fy);
    lerp4(y0, y1, fz)
}

fn particle_view_z(view: &View, particle: OitParticle) -> f32 {
    let center = Vec3::from_array(particle.pos);
    let camera = Vec3::from_array(view.camera_position);
    let forward = Vec3::from_array(view.camera_forward);
    forward.dot(center - camera)
}

fn cpu_ext_column(params: &FogCurve, view: &View, particles: &[OitParticle]) -> ([u32; 16], u32) {
    let mut dwords = [0u32; 16];
    let mut overflow = u32::MAX;
    for particle in particles.iter().copied() {
        let encoded = extinction_encode(particle.alpha);
        let slice = warped_slice_of_z(params, particle_view_z(view, particle));
        let (s0, w0, s1, w1) = splat_weights(slice, params.params.slice_count_u32);
        for (slice, u) in [
            (s0, extinction_to_u8(encoded * w0)),
            (s1, extinction_to_u8(encoded * w1)),
        ] {
            if u == 0 {
                continue;
            }
            let word = (slice / 4) as usize;
            let lane = slice % 4;
            let shift = lane * 8;
            let prev = dwords[word];
            dwords[word] = prev.wrapping_add(u << shift);
            let lane_byte = (prev >> shift) & 0xff;
            if lane_byte + u > 255 {
                overflow = overflow.min(slice);
            }
        }
    }
    (dwords, overflow)
}

fn cpu_v_int_texel_oit(
    params: &FogCurve,
    view: &View,
    dials: &FogDials,
    light: &TwinLight,
    ext: &[u32; 16],
    overflow: u32,
    x: u32,
    y: u32,
    z: u32,
) -> [f32; 4] {
    let dir = ray_direction(view, UVec2::new(x, y));
    let forward = Vec3::from_array(view.camera_forward);
    let view_to_ray = 1.0 / forward.dot(dir).max(1.0e-6);
    let camera = Vec3::from_array(view.camera_position);
    let mut luminance = Vec3::ZERO;
    let mut throughput = 1.0f32;
    for i in 0..=z {
        if i >= overflow {
            return [luminance.x, luminance.y, luminance.z, 0.0];
        }
        let z0 = z_of_warped_slice(params, i as f32);
        let z1 = z_of_warped_slice(params, i as f32 + 1.0);
        let (scatter, sigma_t) = cpu_slice_light(params, dials, light, camera, dir, view_to_ray, i);
        let packed = ext[(i / 4) as usize];
        let oit_od = extinction_decode((packed >> ((i % 4) * 8)) & 0xff);
        let (added, step_t) = integrate_step(
            scatter,
            sigma_t,
            oit_od,
            (z1 - z0) * view_to_ray,
            throughput,
        );
        luminance += Vec3::from_array(added);
        throughput *= step_t;
    }
    [luminance.x, luminance.y, luminance.z, throughput]
}

fn cpu_v_int_sample_oit(
    params: &FogCurve,
    view: &View,
    dials: &FogDials,
    light: &TwinLight,
    ext: &[u32; 16],
    overflow: u32,
    uvw: Vec3,
) -> [f32; 4] {
    let (x0, x1, fx) = axis_lerp(uvw.x, FROXEL_WIDTH);
    let (y0, y1, fy) = axis_lerp(uvw.y, FROXEL_HEIGHT);
    let (z0, z1, fz) = axis_lerp(uvw.z, FROXEL_DEPTH);
    let sample = |x, y, z| cpu_v_int_texel_oit(params, view, dials, light, ext, overflow, x, y, z);

    let c000 = sample(x0, y0, z0);
    let c100 = sample(x1, y0, z0);
    let c010 = sample(x0, y1, z0);
    let c110 = sample(x1, y1, z0);
    let c001 = sample(x0, y0, z1);
    let c101 = sample(x1, y0, z1);
    let c011 = sample(x0, y1, z1);
    let c111 = sample(x1, y1, z1);

    let x00 = lerp4(c000, c100, fx);
    let x10 = lerp4(c010, c110, fx);
    let x01 = lerp4(c001, c101, fx);
    let x11 = lerp4(c011, c111, fx);
    let y0 = lerp4(x00, x10, fy);
    let y1 = lerp4(x01, x11, fy);
    lerp4(y0, y1, fz)
}

fn cpu_composite_pixel(
    params: &FogCurve,
    screen_view: &View,
    dials: &FogDials,
    light: &TwinLight,
    pixel: UVec2,
    depth: f32,
    base: Vec4,
) -> Vec4 {
    let scene_z = if depth > 0.0 {
        screen_view.depth_near_plane / depth
    } else {
        f32::INFINITY
    };
    let sample_z = scene_z.min(params.params.f);
    let dither = interleaved_gradient_noise(pixel.x, pixel.y);
    let w = ((warped_slice_of_z(params, sample_z) - dials.fog_sample_bias + dither)
        / params.params.slice_count)
        .clamp(0.0, 1.0);
    let uv = (pixel.as_vec2() + 0.5) / UVec2::from_array(screen_view.output_size).as_vec2();
    let froxel_view = View {
        output_size: [FROXEL_WIDTH, FROXEL_HEIGHT],
        ..*screen_view
    };
    let v_int = cpu_v_int_sample(params, &froxel_view, dials, light, Vec3::new(uv.x, uv.y, w));
    let mut rgb = base.truncate() * v_int[3] + Vec3::new(v_int[0], v_int[1], v_int[2]);

    if scene_z > params.params.f {
        let dir = ray_direction(screen_view, pixel);
        let forward = Vec3::from_array(screen_view.camera_forward);
        let view_to_ray = 1.0 / forward.dot(dir).max(1.0e-6);
        let t1 = if scene_z.is_finite() {
            scene_z * view_to_ray
        } else {
            f32::INFINITY
        };
        let od = height_fog_optical_depth(
            screen_view.camera_position[1],
            dir.y,
            params.params.f * view_to_ray,
            t1,
            dials.density,
            dials.height_falloff,
            dials.height_offset,
        );
        let beyond_t = transmittance(od);
        // Beyond-far inscatter uses sun phase and flat ambient light.
        let phase = hg_phase(dir.dot(light.sun_dir), dials.anisotropy);
        let beyond_light = light.sun_color * phase + Vec3::from_array(dials.ambient_color);
        rgb = rgb * beyond_t + beyond_light * ((1.0 - beyond_t) * v_int[3]);
    }

    Vec4::new(rgb.x, rgb.y, rgb.z, base.w)
}

fn cpu_composite_pixel_oit(
    params: &FogCurve,
    screen_view: &View,
    dials: &FogDials,
    light: &TwinLight,
    ext: &[u32; 16],
    overflow: u32,
    pixel: UVec2,
    depth: f32,
    base: Vec4,
) -> Vec4 {
    let scene_z = if depth > 0.0 {
        screen_view.depth_near_plane / depth
    } else {
        f32::INFINITY
    };
    let sample_z = scene_z.min(params.params.f);
    let dither = interleaved_gradient_noise(pixel.x, pixel.y);
    let w = ((warped_slice_of_z(params, sample_z) - dials.fog_sample_bias + dither)
        / params.params.slice_count)
        .clamp(0.0, 1.0);
    let uv = (pixel.as_vec2() + 0.5) / UVec2::from_array(screen_view.output_size).as_vec2();
    let froxel_view = View {
        output_size: [FROXEL_WIDTH, FROXEL_HEIGHT],
        ..*screen_view
    };
    let v_int = cpu_v_int_sample_oit(
        params,
        &froxel_view,
        dials,
        light,
        ext,
        overflow,
        Vec3::new(uv.x, uv.y, w),
    );
    let mut rgb = base.truncate() * v_int[3] + Vec3::new(v_int[0], v_int[1], v_int[2]);

    if scene_z > params.params.f {
        let dir = ray_direction(screen_view, pixel);
        let forward = Vec3::from_array(screen_view.camera_forward);
        let view_to_ray = 1.0 / forward.dot(dir).max(1.0e-6);
        let t1 = if scene_z.is_finite() {
            scene_z * view_to_ray
        } else {
            f32::INFINITY
        };
        let od = height_fog_optical_depth(
            screen_view.camera_position[1],
            dir.y,
            params.params.f * view_to_ray,
            t1,
            dials.density,
            dials.height_falloff,
            dials.height_offset,
        );
        let beyond_t = transmittance(od);
        // Beyond-far inscatter uses sun phase and flat ambient light.
        let phase = hg_phase(dir.dot(light.sun_dir), dials.anisotropy);
        let beyond_light = light.sun_color * phase + Vec3::from_array(dials.ambient_color);
        rgb = rgb * beyond_t + beyond_light * ((1.0 - beyond_t) * v_int[3]);
    }

    Vec4::new(rgb.x, rgb.y, rgb.z, base.w)
}

fn cpu_oit_pixel(
    params: &FogCurve,
    screen_view: &View,
    dials: &FogDials,
    light: &TwinLight,
    pixel: UVec2,
    depth: f32,
    base: Vec4,
    particles: &[OitParticle],
) -> Vec4 {
    let (ext, overflow) = cpu_ext_column(params, screen_view, particles);
    let hdr = cpu_composite_pixel_oit(
        params,
        screen_view,
        dials,
        light,
        &ext,
        overflow,
        pixel,
        depth,
        base,
    );
    let froxel_view = View {
        output_size: [FROXEL_WIDTH, FROXEL_HEIGHT],
        ..*screen_view
    };
    let uv = (pixel.as_vec2() + 0.5) / UVec2::from_array(screen_view.output_size).as_vec2();
    let dither = interleaved_gradient_noise(pixel.x, pixel.y);

    let mut accum_rgb = Vec3::ZERO;
    let mut accum_alpha_w = 0.0;
    let mut accum_neg_log = 0.0;
    for particle in particles.iter().copied() {
        let view_z = particle_view_z(screen_view, particle);
        let sample_z = view_z.min(params.params.f);
        let w = ((warped_slice_of_z(params, sample_z) - dials.fog_sample_bias + dither)
            / params.params.slice_count)
            .clamp(0.0, 1.0);
        let v_int = cpu_v_int_sample_oit(
            params,
            &froxel_view,
            dials,
            light,
            &ext,
            overflow,
            Vec3::new(uv.x, uv.y, w),
        );
        let alpha = particle.alpha.clamp(0.0, 1.0);
        let fog_dim = v_int[3];
        let alpha_w = alpha * fog_dim;
        accum_rgb += Vec3::from_array(particle.color) * fog_dim * alpha_w;
        accum_alpha_w += alpha_w;
        accum_neg_log += extinction_encode(alpha) * EXT_MAX;
    }

    let rgb = Vec3::from_array(oit_resolve(
        accum_rgb.to_array(),
        accum_alpha_w,
        accum_neg_log,
        hdr.truncate().to_array(),
    ));
    Vec4::new(rgb.x, rgb.y, rgb.z, base.w)
}

fn read_hdr(readback: gpu::Ptr<u16>, size: UVec2) -> Vec<Vec4> {
    let mut out = Vec::with_capacity((size.x * size.y) as usize);
    for i in 0..(size.x * size.y) as usize {
        let offset = i * 4;
        let px = unsafe {
            Vec4::new(
                f16_to_f32(*readback.cpu.add(offset)),
                f16_to_f32(*readback.cpu.add(offset + 1)),
                f16_to_f32(*readback.cpu.add(offset + 2)),
                f16_to_f32(*readback.cpu.add(offset + 3)),
            )
        };
        out.push(px);
    }
    out
}

/// Copies a D32 target after GPU idleness.
fn read_depth(gpu: &Gpu, depth: gpu::Texture, readback: gpu::Ptr<f32>) {
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_copy_texture_to_buffer(cb, depth, readback.cast());
    gpu.cmd_barrier(cb, Stage::Transfer, Stage::All, HazardFlags::empty());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);
}

fn particle_at(view: &View, view_z: f32, color: [f32; 3], alpha: f32) -> OitParticle {
    let camera = Vec3::from_array(view.camera_position);
    let forward = Vec3::from_array(view.camera_forward);
    OitParticle {
        pos: (camera + forward * view_z).to_array(),
        size: 9.0,
        color,
        alpha,
        ..Default::default()
    }
}

fn upload_particles(gpu: &Gpu, particles: &[OitParticle]) -> gpu::Ptr<OitParticle> {
    let ptr = gpu.alloc_slice::<OitParticle>(particles.len() as u64, gpu::Memory::Default);
    // SAFETY: allocation matches the source slice.
    unsafe {
        std::ptr::copy_nonoverlapping(particles.as_ptr(), ptr.cpu, particles.len());
    }
    ptr
}

#[allow(clippy::too_many_arguments)]
fn render_volumetric_frame(
    gpu: &Gpu,
    heap: &gpu::HeapSlots,
    fog: &VolumetricPasses,
    hdr: gpu::Texture,
    depth: gpu::Texture,
    depth_slot: gpu::SampledSlot,
    hdr_rw: gpu::StorageSlot,
    clamp_sampler: gpu::SamplerSlot,
    view: &View,
    dials: &FogDials,
    light_inputs: &FogLightInputs,
    depth_value: f32,
    base: Vec4,
    particles: GpuPtr<OitParticle>,
    particle_count: u32,
    particles_tinted: bool,
    readback: gpu::Ptr<u16>,
) {
    let mut frame = TestFrameAlloc {
        gpu,
        ptrs: Vec::new(),
    };
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            color_attachments: &[RenderAttachment {
                texture: hdr,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: base.to_array(),
                ..Default::default()
            }],
            depth_attachment: Some(RenderAttachment {
                texture: depth,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [depth_value, 0.0, 0.0, 0.0],
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    gpu.cmd_end_render_pass(cb);
    gpu.cmd_barrier(cb, Stage::All, Stage::All, HazardFlags::empty());
    heap.bind(gpu, cb);
    fog.record(
        gpu,
        cb,
        &mut frame,
        depth,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        view,
        dials,
        light_inputs,
        particles,
        particle_count,
        particles_tinted,
    );
    gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_copy_texture_to_buffer(cb, hdr, readback.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);
    frame.free();
}

// GPU division may differ from CPU division by a few ULPs. Compare curve
// fields by ULP, then use GPU parameters for pixel-reference evaluation.
fn assert_params_close(got: FroxelParams, want: FroxelParams) {
    fn ulp_close(got: f32, want: f32, name: &str) {
        let diff = (got.to_bits() as i64 - want.to_bits() as i64).unsigned_abs();
        assert!(diff <= 8, "{name}: gpu {got} vs cpu {want} ({diff} ulp)");
    }
    ulp_close(got.f, want.f, "f");
    ulp_close(got.a, want.a, "a");
    ulp_close(got.inv_a, want.inv_a, "inv_a");
    ulp_close(got.slice_count, want.slice_count, "slice_count");
    ulp_close(got.slice_scale, want.slice_scale, "slice_scale");
    ulp_close(got.z_scale, want.z_scale, "z_scale");
    assert_eq!(got.slice_count_u32, want.slice_count_u32, "slice_count_u32");
}

#[test]
fn oit_order_independent_transparency_matches_cpu_twin() {
    const W: u32 = 80;
    const H: u32 = 50;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 80.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.08, 0.12, 0.18, 1.0);
    let dials = FogDials {
        density: 0.05,
        height_falloff: 0.0,
        f_min: 96.0,
        f_max: 96.0,
        ..Default::default()
    };
    let light_inputs = test_light_inputs();
    let twin = twin_from(&light_inputs, None, 0);
    let particles = [
        particle_at(&view, 10.0, [0.95, 0.20, 0.12], 0.5),
        particle_at(&view, 20.0, [0.20, 0.85, 0.28], 0.5),
        particle_at(&view, 40.0, [0.25, 0.35, 1.00], 0.5),
    ];
    let mut reversed = particles;
    reversed.reverse();

    let mut heap = gpu.heap_slots_create(16, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let mut fog = VolumetricPasses::new(&gpu, &mut heap);
    fog.resize(&gpu, &mut heap, size);
    let forward_ptr = upload_particles(&gpu, &particles);
    let reversed_ptr = upload_particles(&gpu, &reversed);
    let readback_a = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);
    let readback_b = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        forward_ptr.gpu,
        particles.len() as u32,
        false,
        readback_a,
    );
    let got_params = fog.curve_after_idle();
    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        reversed_ptr.gpu,
        reversed.len() as u32,
        false,
        readback_b,
    );

    let img_a = read_hdr(readback_a, size);
    let img_b = read_hdr(readback_b, size);
    for (i, (a, b)) in img_a.iter().zip(&img_b).enumerate() {
        for c in 0..4 {
            let err = (a[c] - b[c]).abs();
            assert!(
                err <= 2.0e-3,
                "pixel {i} channel {c}: forward {a} vs reversed {b}"
            );
        }
    }

    for pixel in [
        UVec2::new(W / 2, H / 2),
        UVec2::new(W / 2 - 2, H / 2 + 1),
        UVec2::new(W / 2 + 3, H / 2 - 2),
    ] {
        let got = img_a[(pixel.y * W + pixel.x) as usize];
        let want = cpu_oit_pixel(
            &got_params,
            &view,
            &dials,
            &twin,
            pixel,
            depth_value,
            base,
            &particles,
        );
        // OIT quantizes extinction, slices, filtering, and FP16 blending.
        // The tolerance covers these losses while checking ordering and depth.
        for c in 0..4 {
            let err = (got[c] - want[c]).abs();
            assert!(
                err <= 2.0e-2,
                "pixel {pixel:?} channel {c}: gpu {} vs cpu {} (err {err})",
                got[c],
                want[c]
            );
        }
    }

    gpu.free(readback_a);
    gpu.free(readback_b);
    gpu.free(forward_ptr);
    gpu.free(reversed_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

#[test]
fn oit_particles_are_dimmed_by_fog() {
    const W: u32 = 80;
    const H: u32 = 50;
    const OIT_TOL: f32 = 2.0e-2;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 80.0f32;
    let particle_z = 24.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.20, 0.20, 0.20, 1.0);
    let no_fog_dials = FogDials {
        density: 0.0,
        height_falloff: 0.0,
        ambient_color: [0.0; 3],
        f_min: 96.0,
        f_max: 96.0,
        ..Default::default()
    };
    let fog_dials = FogDials {
        density: 0.04,
        ..no_fog_dials
    };
    let light_inputs = FogLightInputs {
        sun_dir: test_sun_dir().to_array(),
        sun_color: [0.0; 3],
        occluder: None,
        local_lights: GpuPtr::null(),
        local_light_count: 0,
    };
    let twin = twin_from(&light_inputs, None, 0);
    let particles = [particle_at(&view, particle_z, [1.0, 0.92, 0.75], 0.5)];

    let mut heap = gpu.heap_slots_create(16, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let mut fog = VolumetricPasses::new(&gpu, &mut heap);
    fog.resize(&gpu, &mut heap, size);
    let particle_ptr = upload_particles(&gpu, &particles);
    let readback_no_fog = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);
    let readback_fog = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &no_fog_dials,
        &light_inputs,
        depth_value,
        base,
        particle_ptr.gpu,
        particles.len() as u32,
        false,
        readback_no_fog,
    );
    let no_fog_params = fog.curve_after_idle();
    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &fog_dials,
        &light_inputs,
        depth_value,
        base,
        particle_ptr.gpu,
        particles.len() as u32,
        false,
        readback_fog,
    );
    let fog_params = fog.curve_after_idle();

    let no_fog_img = read_hdr(readback_no_fog, size);
    let fog_img = read_hdr(readback_fog, size);
    // Probe centers remain inside the size-9 particle quad.
    let probes = [
        UVec2::new(W / 2, H / 2),
        UVec2::new(W / 2 - 2, H / 2 + 1),
        UVec2::new(W / 2 + 3, H / 2 - 2),
    ];
    let mut checks = Vec::new();
    for pixel in probes {
        let no_fog_gpu = no_fog_img[(pixel.y * W + pixel.x) as usize];
        let fog_gpu = fog_img[(pixel.y * W + pixel.x) as usize];
        let no_fog_cpu = cpu_oit_pixel(
            &no_fog_params,
            &view,
            &no_fog_dials,
            &twin,
            pixel,
            depth_value,
            base,
            &particles,
        );
        let fog_cpu = cpu_oit_pixel(
            &fog_params,
            &view,
            &fog_dials,
            &twin,
            pixel,
            depth_value,
            base,
            &particles,
        );
        checks.push((pixel, no_fog_gpu, no_fog_cpu, fog_gpu, fog_cpu));
    }

    gpu.free(readback_no_fog);
    gpu.free(readback_fog);
    gpu.free(particle_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);

    assert!(
        particle_z < no_fog_params.params.f && particle_z < fog_params.params.f,
        "particle z {particle_z} must stay inside froxel far bounds {:?} / {:?}",
        no_fog_params.params.f,
        fog_params.params.f
    );
    for (pixel, no_fog_gpu, no_fog_cpu, fog_gpu, fog_cpu) in checks {
        let luminance = |px: Vec4| px.x * 0.2126 + px.y * 0.7152 + px.z * 0.0722;
        let no_fog_luma = luminance(no_fog_gpu);
        let fog_luma = luminance(fog_gpu);
        assert!(
            fog_luma < no_fog_luma * 0.75 && no_fog_luma - fog_luma > 0.12,
            "pixel {pixel:?}: fogged particle must be meaningfully darker, no-fog {} vs fog {}",
            no_fog_gpu,
            fog_gpu
        );

        // OIT quantization and FP16 blending determine this tolerance.
        for (label, got, want) in [
            ("no fog", no_fog_gpu, no_fog_cpu),
            ("fog", fog_gpu, fog_cpu),
        ] {
            for c in 0..4 {
                let err = (got[c] - want[c]).abs();
                assert!(
                    err <= OIT_TOL,
                    "{label} pixel {pixel:?} channel {c}: gpu {} vs cpu {} (err {err})",
                    got,
                    want
                );
            }
        }
    }
}

#[test]
fn oit_resolve_preserves_foreground_fog_inscatter() {
    const W: u32 = 80;
    const H: u32 = 50;
    const OIT_TOL: f32 = 2.0e-2;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 80.0f32;
    let particle_z = 24.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.0, 0.0, 0.0, 1.0);
    let dials = FogDials {
        density: 0.04,
        height_falloff: 0.0,
        ambient_color: [1.0; 3],
        f_min: 96.0,
        f_max: 96.0,
        ..Default::default()
    };
    let light_inputs = FogLightInputs {
        sun_dir: test_sun_dir().to_array(),
        sun_color: [0.0; 3],
        occluder: None,
        local_lights: GpuPtr::null(),
        local_light_count: 0,
    };
    let twin = twin_from(&light_inputs, None, 0);
    // Black emission verifies resolve preserves the pre-extinguished medium.
    let particles = [particle_at(&view, particle_z, [0.0; 3], 0.8)];

    let mut heap = gpu.heap_slots_create(16, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let mut fog = VolumetricPasses::new(&gpu, &mut heap);
    fog.resize(&gpu, &mut heap, size);
    let particle_ptr = upload_particles(&gpu, &particles);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        particle_ptr.gpu,
        particles.len() as u32,
        false,
        readback,
    );
    let params = fog.curve_after_idle();
    let image = read_hdr(readback, size);
    let (ext, overflow) = cpu_ext_column(&params, &view, &particles);
    let probes = [
        UVec2::new(W / 2, H / 2),
        UVec2::new(W / 2 - 2, H / 2 + 1),
        UVec2::new(W / 2 + 3, H / 2 - 2),
    ];
    let mut checks = Vec::new();
    for pixel in probes {
        let got = image[(pixel.y * W + pixel.x) as usize];
        let pre_resolve = cpu_composite_pixel_oit(
            &params,
            &view,
            &dials,
            &twin,
            &ext,
            overflow,
            pixel,
            depth_value,
            base,
        );
        let want = cpu_oit_pixel(
            &params,
            &view,
            &dials,
            &twin,
            pixel,
            depth_value,
            base,
            &particles,
        );
        checks.push((pixel, got, pre_resolve, want));
    }

    gpu.free(readback);
    gpu.free(particle_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);

    for (pixel, got, pre_resolve, want) in checks {
        let pre_luma = pre_resolve
            .truncate()
            .dot(Vec3::new(0.2126, 0.7152, 0.0722));
        let got_luma = got.truncate().dot(Vec3::new(0.2126, 0.7152, 0.0722));
        let old_double_attenuated = pre_luma * (1.0 - particles[0].alpha);
        assert!(
            pre_luma > 0.2 && got_luma > old_double_attenuated + 0.15,
            "pixel {pixel:?}: foreground fog was double-attenuated: pre {pre_resolve}, got {got}"
        );
        for c in 0..4 {
            let cpu_err = (got[c] - want[c]).abs();
            let preservation_err = (got[c] - pre_resolve[c]).abs();
            assert!(
                cpu_err <= OIT_TOL && preservation_err <= OIT_TOL,
                "pixel {pixel:?} channel {c}: gpu {got}, cpu {want}, pre-resolve {pre_resolve}"
            );
        }
    }
}

#[test]
fn oit_overflow_saturates_instead_of_corrupting() {
    const W: u32 = 80;
    const H: u32 = 50;
    const COUNT: usize = 60;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 80.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.25, 0.35, 0.45, 1.0);
    let dials = FogDials {
        density: 0.0,
        f_min: 96.0,
        f_max: 96.0,
        ..Default::default()
    };
    let light_inputs = test_light_inputs();
    let twin = twin_from(&light_inputs, None, 0);
    let particle = particle_at(&view, 20.0, [0.0, 0.0, 0.0], 0.5);
    let particles = [particle; COUNT];

    let mut heap = gpu.heap_slots_create(16, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let mut fog = VolumetricPasses::new(&gpu, &mut heap);
    fog.resize(&gpu, &mut heap, size);
    let particle_ptr = upload_particles(&gpu, &particles);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        particle_ptr.gpu,
        COUNT as u32,
        false,
        readback,
    );
    let got_params = fog.curve_after_idle();
    let img = read_hdr(readback, size);
    let pixel = UVec2::new(W / 2, H / 2);
    let got = img[(pixel.y * W + pixel.x) as usize];
    let want = cpu_oit_pixel(
        &got_params,
        &view,
        &dials,
        &twin,
        pixel,
        depth_value,
        base,
        &particles,
    );

    for c in 0..3 {
        assert!(got[c].is_finite(), "channel {c} is not finite: {}", got[c]);
        assert!(
            got[c] <= 1.0e-3,
            "black overflow stack leaked channel {c}: {}",
            got[c]
        );
        assert!(
            got[c] <= 1.0,
            "channel {c} exceeds physical max: {}",
            got[c]
        );
        let err = (got[c] - want[c]).abs();
        assert!(
            err <= 2.0e-2,
            "overflow channel {c}: gpu {} vs cpu {} (err {err})",
            got[c],
            want[c]
        );
    }
    assert_eq!(got.w, base.w, "resolve must preserve HDR alpha");

    gpu.free(readback);
    gpu.free(particle_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

/// Tests conservative depth priming for opaque fog:
/// 1. Saturated columns receive predicted boundary depths.
/// 2. Hidden particles match the wall-only image.
/// 3. Nearer geometry preserves its depth.
#[test]
fn oit_zero_transmittance_prime_culls_hidden_particles() {
    const W: u32 = 80;
    const H: u32 = 50;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 80.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.30, 0.25, 0.20, 1.0);
    let dials = FogDials {
        density: 0.0,
        f_min: 96.0,
        f_max: 96.0,
        ..Default::default()
    };
    let light_inputs = test_light_inputs();

    // Two frustum-covering α = 0.95 layers at z = 10 contribute
    // 2 · 3.0 > EXT_MAX, so every froxel column saturates there.
    let camera = Vec3::from_array(view.camera_position);
    let forward = Vec3::from_array(view.camera_forward);
    let wall = OitParticle {
        pos: (camera + forward * 10.0).to_array(),
        size: 60.0,
        color: [0.6, 0.6, 0.6],
        alpha: 0.95,
        ..Default::default()
    };
    let hidden = OitParticle {
        pos: (camera + forward * 40.0).to_array(),
        size: 30.0,
        color: [10.0, 0.0, 0.0],
        alpha: 0.9,
        ..Default::default()
    };
    let with_hidden = [wall, wall, hidden];
    let wall_only = [wall, wall];

    let mut heap = gpu.heap_slots_create(16, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let mut fog = VolumetricPasses::new(&gpu, &mut heap);
    fog.resize(&gpu, &mut heap, size);
    let with_hidden_ptr = upload_particles(&gpu, &with_hidden);
    let wall_only_ptr = upload_particles(&gpu, &wall_only);
    let readback_a = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);
    let readback_b = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        with_hidden_ptr.gpu,
        with_hidden.len() as u32,
        false,
        readback_a,
    );
    let quad_count = fog.prime_quad_count_after_idle();
    let tiles = W.div_ceil(abi_light::PRIME_TILE) * H.div_ceil(abi_light::PRIME_TILE);
    assert_eq!(
        quad_count, tiles,
        "frustum-covering wall must saturate every tile"
    );

    // Property 1: every column carries the reference boundary depth.
    let got_params = fog.curve_after_idle();
    let (ext, overflow) = cpu_ext_column(&got_params, &view, &with_hidden);
    let mut zero_slice = overflow.min(got_params.params.slice_count_u32);
    let mut throughput = 1.0f32;
    for i in 0..zero_slice {
        let od = extinction_decode((ext[(i / 4) as usize] >> ((i % 4) * 8)) & 0xff);
        throughput *= transmittance(od);
        if throughput <= abi_light::ZERO_TRANS_EPS {
            zero_slice = i;
            break;
        }
    }
    assert!(
        zero_slice < got_params.params.slice_count_u32,
        "twin must saturate: zero_slice {zero_slice}"
    );
    let expected_depth = abi_light::prime_quad_depth(
        &got_params,
        zero_slice as f32 + 1.0 + dials.fog_sample_bias + 1.5,
        view.depth_near_plane,
    );
    let wall_depth = view.depth_near_plane / 10.0;
    assert!(
        expected_depth < wall_depth,
        "margin must not cull the wall itself: {expected_depth} vs {wall_depth}"
    );
    let depth_rb = gpu.alloc_slice::<f32>((W * H) as u64, gpu::Memory::Readback);
    read_depth(&gpu, depth.texture, depth_rb);
    for pixel in [UVec2::new(W / 2, H / 2), UVec2::new(3, 3)] {
        let got = unsafe { *depth_rb.cpu.add((pixel.y * W + pixel.x) as usize) };
        let err = (got - expected_depth).abs();
        // GPU and CPU exp2 differ by a few ULPs.
        assert!(
            err <= expected_depth * 1.0e-4,
            "primed depth at {pixel:?}: gpu {got} vs cpu {expected_depth}"
        );
    }

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        wall_only_ptr.gpu,
        wall_only.len() as u32,
        false,
        readback_b,
    );

    // Property 2: culling matches the wall-only image within tolerance.
    let img_a = read_hdr(readback_a, size);
    let img_b = read_hdr(readback_b, size);
    for (i, (a, b)) in img_a.iter().zip(&img_b).enumerate() {
        for c in 0..4 {
            let err = (a[c] - b[c]).abs();
            assert!(
                err <= 2.0e-3,
                "pixel {i} channel {c}: hidden {a} vs wall-only {b} (err {err})"
            );
        }
    }

    // Property 3: nearer geometry leaves cleared depth unchanged.
    let near_scene_depth = view.depth_near_plane / 5.0;
    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        near_scene_depth,
        base,
        wall_only_ptr.gpu,
        wall_only.len() as u32,
        false,
        readback_b,
    );
    read_depth(&gpu, depth.texture, depth_rb);
    for pixel in [UVec2::new(W / 2, H / 2), UVec2::new(3, 3)] {
        let got = unsafe { *depth_rb.cpu.add((pixel.y * W + pixel.x) as usize) };
        assert_eq!(
            got, near_scene_depth,
            "prime must never overwrite nearer opaque depth at {pixel:?}"
        );
    }

    gpu.free(depth_rb);
    gpu.free(readback_a);
    gpu.free(readback_b);
    gpu.free(with_hidden_ptr);
    gpu.free(wall_only_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

#[test]
fn fog_probe_pixels_match_cpu_twin() {
    const W: u32 = 37;
    const H: u32 = 23;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 48.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.25, 0.5, 0.75, 1.0);
    let dials = FogDials {
        density: 0.05,
        height_falloff: 0.2,
        height_offset: -3.0,
        ..Default::default()
    };
    let light_inputs = test_light_inputs();
    let twin = twin_from(&light_inputs, None, 0);

    let mut heap = gpu.heap_slots_create(8, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let fog = VolumetricPasses::new(&gpu, &mut heap);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        GpuPtr::null(),
        0,
        false,
        readback,
    );

    let max_depth = view.depth_near_plane / depth_value;
    let want_params = froxel_params_from(
        max_depth,
        dials.slice_count,
        dials.a,
        dials.f_min,
        dials.f_max,
    );
    let got_params = fog.curve_after_idle();
    assert_params_close(got_params.params, want_params);

    let img = read_hdr(readback, size);
    for pixel in [
        UVec2::new(3, 2),
        UVec2::new(18, 11),
        UVec2::new(34, 20),
        UVec2::new(9, 17),
        UVec2::new(27, 5),
    ] {
        let got = img[(pixel.y * W + pixel.x) as usize];
        let want = cpu_composite_pixel(&got_params, &view, &dials, &twin, pixel, depth_value, base);
        // FP16 volume and HDR storage justify the absolute tolerance.
        for c in 0..4 {
            let err = (got[c] - want[c]).abs();
            assert!(
                err <= 4.0e-3,
                "pixel {pixel:?} channel {c}: gpu {} vs cpu {} (err {err})",
                got[c],
                want[c]
            );
        }
    }

    gpu.free(readback);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

/// Tests demand-driven depth warping across two frames.
///
/// The GPU curve and image must match CPU references.
#[test]
fn fog_warp_lut_matches_cpu_twin_and_concentrates() {
    const W: u32 = 64;
    const H: u32 = 40;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 90.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.02, 0.03, 0.05, 1.0);
    let dials = FogDials {
        density: 0.02,
        height_falloff: 0.0,
        f_min: 96.0,
        f_max: 96.0,
        warp_gain: 6.0,
        warp_bound: 4.0,
        ..Default::default()
    };
    let light_inputs = test_light_inputs();
    let twin = twin_from(&light_inputs, None, 0);
    let camera = Vec3::from_array(view.camera_position);
    let forward = Vec3::from_array(view.camera_forward);
    let stack_z = 60.0f32;
    let particle = OitParticle {
        pos: (camera + forward * stack_z).to_array(),
        size: 50.0,
        color: [0.9, 0.8, 0.6],
        alpha: 0.6,
        ..Default::default()
    };
    let particles = [particle, particle, particle];

    let mut heap = gpu.heap_slots_create(8, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let mut fog = VolumetricPasses::new(&gpu, &mut heap);
    fog.resize(&gpu, &mut heap, size);
    let particle_ptr = upload_particles(&gpu, &particles);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        particle_ptr.gpu,
        particles.len() as u32,
        false,
        readback,
    );
    let curve1 = fog.curve_after_idle();
    let hist = fog.hist_after_idle();

    // Frame 1 has no prior demand and uses the identity curve.
    for i in 0..=64u32 {
        assert_eq!(
            curve1.warp[i as usize].to_bits(),
            (i as f32).to_bits(),
            "frame-1 warp edge {i}"
        );
    }
    // The demand histogram peaks at the particle depth.
    let bin = warped_slice_of_z(&curve1, stack_z) as usize;
    let total: u32 = hist.iter().sum();
    let near: u32 = hist[bin.saturating_sub(1)..=(bin + 1).min(63)].iter().sum();
    assert!(total > 0, "stack splatted no demand");
    assert!(
        near as f32 >= 0.99 * total as f32,
        "demand mass {near}/{total} strayed from bin {bin}: {hist:?}"
    );

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        particle_ptr.gpu,
        particles.len() as u32,
        false,
        readback,
    );
    let curve2 = fog.curve_after_idle();

    // GPU LUT build matches the CPU reference histogram.
    let want = abi_light::fog_curve_from(curve1.params, &hist, dials.warp_gain, dials.warp_bound);
    for i in 0..=64usize {
        let we = (curve2.warp[i] - want.warp[i]).abs();
        let ue = (curve2.unwarp[i] - want.unwarp[i]).abs();
        assert!(
            we <= 1.0e-4 && ue <= 1.0e-4,
            "LUT edge {i}: gpu ({}, {}) vs cpu ({}, {})",
            curve2.warp[i],
            curve2.unwarp[i],
            want.warp[i],
            want.unwarp[i]
        );
    }
    // Warping concentrates resolution near the active slice.
    let hot = curve2.warp[bin + 1] - curve2.warp[bin];
    assert!(hot > 2.0, "hot raw slice widened to only {hot} slices");

    // The warped frame matches the CPU reference through curve2.
    let img = read_hdr(readback, size);
    for pixel in [UVec2::new(W / 2, H / 2), UVec2::new(W / 2 - 5, H / 2 + 3)] {
        let got = img[(pixel.y * W + pixel.x) as usize];
        let want_px = cpu_oit_pixel(
            &curve2,
            &view,
            &dials,
            &twin,
            pixel,
            depth_value,
            base,
            &particles,
        );
        for c in 0..4 {
            let err = (got[c] - want_px[c]).abs();
            assert!(
                err <= 2.0e-2,
                "warped pixel {pixel:?} channel {c}: gpu {got} vs cpu {want_px}"
            );
        }
    }

    gpu.free(readback);
    gpu.free(particle_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

/// Tests depth warping for two nearby translucent layers.
///
/// Without separation, both layers share one slice and average. Warping
/// separates them; the reference uses exact painter-order transmittance.
#[test]
fn fog_warp_recovers_thin_stack_ordering() {
    const W: u32 = 64;
    const H: u32 = 40;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 90.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.05, 0.05, 0.05, 1.0);
    // The default sample bias clears each particle's splat texel. Unwarped,
    // it also clears the nearby layer; warped separation leaves it intact.
    let dials_off = FogDials {
        density: 0.0,
        f_min: 96.0,
        f_max: 96.0,
        warp_gain: 0.0,
        warp_bound: 6.0,
        ..Default::default()
    };
    let dials_on = FogDials {
        warp_gain: 10.0,
        ..dials_off
    };
    let light_inputs = test_light_inputs();
    let camera = Vec3::from_array(view.camera_position);
    let forward = Vec3::from_array(view.camera_forward);
    let front = OitParticle {
        pos: (camera + forward * 59.0).to_array(),
        size: 50.0,
        color: [1.0, 0.0, 0.0],
        alpha: 0.75,
        ..Default::default()
    };
    let back = OitParticle {
        pos: (camera + forward * 61.0).to_array(),
        size: 50.0,
        color: [0.0, 1.0, 0.0],
        alpha: 0.75,
        ..Default::default()
    };
    let particles = [front, back];

    let mut heap = gpu.heap_slots_create(8, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let mut fog = VolumetricPasses::new(&gpu, &mut heap);
    fog.resize(&gpu, &mut heap, size);
    let particle_ptr = upload_particles(&gpu, &particles);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    let run = |dials: &FogDials, frames: u32| -> Vec<Vec4> {
        for _ in 0..frames {
            render_volumetric_frame(
                &gpu,
                &heap,
                &fog,
                hdr.texture,
                depth.texture,
                depth_slot,
                hdr_rw,
                clamp_sampler,
                &view,
                dials,
                &light_inputs,
                depth_value,
                base,
                particle_ptr.gpu,
                particles.len() as u32,
                false,
                readback,
            );
        }
        read_hdr(readback, size)
    };
    let img_off = run(&dials_off, 1);
    // The first warped frame consumes prior demand; the second uses the
    // converged, frame-stable warp.
    let img_on = run(&dials_on, 2);

    // Reference resolve law uses V_int for both ordering and emission
    // attenuation. Separate slices give w_front = 1 and w_back = T_front;
    // slice-tied layers collapse their weights and average the colors.
    let (a, t) = (0.75f32, 1.0 - 0.75f32);
    let (w_front, w_back) = (1.0f32, t);
    let coverage = 1.0 - t * t;
    let scale = coverage / (a * (w_front + w_back));
    let bg = base.truncate() * (t * t);
    let reference = Vec4::new(
        a * w_front * w_front * scale + bg.x,
        a * w_back * w_back * scale + bg.y,
        bg.z,
        1.0,
    );
    let mut err_off_max = 0.0f32;
    let mut err_on_max = 0.0f32;
    for pixel in [
        UVec2::new(W / 2, H / 2),
        UVec2::new(W / 2 - 4, H / 2 + 2),
        UVec2::new(W / 2 + 5, H / 2 - 3),
    ] {
        let idx = (pixel.y * W + pixel.x) as usize;
        let err_off = (img_off[idx] - reference).abs().max_element();
        let err_on = (img_on[idx] - reference).abs().max_element();
        err_off_max = err_off_max.max(err_off);
        err_on_max = err_on_max.max(err_on);
    }
    assert!(
        err_off_max > 0.25,
        "fixture lost the artifact: unwarped err only {err_off_max}"
    );
    assert!(
        err_on_max < 0.02,
        "warped stack off the exact-transmittance law: {err_on_max}"
    );

    gpu.free(readback);
    gpu.free(particle_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

/// Tests RGB extinction against exact-transmittance resolve.
///
/// A tinted middle layer affects later emission and background channels.
/// An untinted control remains gray.
#[test]
fn oit_tinted_glass_transmits_per_channel() {
    const W: u32 = 64;
    const H: u32 = 40;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 90.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.35, 0.35, 0.35, 1.0);
    let dials = FogDials {
        density: 0.0,
        f_min: 96.0,
        f_max: 96.0,
        ..Default::default()
    };
    let light_inputs = test_light_inputs();
    let camera = Vec3::from_array(view.camera_position);
    let forward = Vec3::from_array(view.camera_forward);
    let tint = [0.95f32, 0.25, 0.08];
    let smoke = |z: f32| OitParticle {
        pos: (camera + forward * z).to_array(),
        size: 50.0,
        color: [0.5; 3],
        alpha: 0.4,
        ..Default::default()
    };
    let glass = OitParticle {
        pos: (camera + forward * 40.0).to_array(),
        size: 50.0,
        color: [0.1; 3],
        alpha: 0.15,
        tint_od: abi_light::tint_od_from_transmittance(tint),
        _pad: 0,
    };
    let gray_glass = OitParticle {
        tint_od: [0.0; 3],
        ..glass
    };
    let tinted_scene = [smoke(20.0), glass, smoke(60.0)];
    let control_scene = [smoke(20.0), gray_glass, smoke(60.0)];

    let mut heap = gpu.heap_slots_create(16, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let mut fog = VolumetricPasses::new(&gpu, &mut heap);
    fog.resize(&gpu, &mut heap, size);
    let tinted_ptr = upload_particles(&gpu, &tinted_scene);
    let control_ptr = upload_particles(&gpu, &control_scene);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    let run = |ptr: GpuPtr<OitParticle>, tinted: bool| -> Vec<Vec4> {
        render_volumetric_frame(
            &gpu,
            &heap,
            &fog,
            hdr.texture,
            depth.texture,
            depth_slot,
            hdr_rw,
            clamp_sampler,
            &view,
            &dials,
            &light_inputs,
            depth_value,
            base,
            ptr,
            3,
            tinted,
            readback,
        );
        read_hdr(readback, size)
    };
    let img_control = run(control_ptr.gpu, false);
    let img_tinted = run(tinted_ptr.gpu, true);

    // Reference scalar transmittances from coverage.
    let (a_s, a_g) = (0.4f32, 0.15f32);
    let (t_s, t_g) = (1.0 - a_s, 1.0 - a_g);
    // Scalar prefixes order events; chroma affects later radiance.
    let w = [1.0f32, t_s, t_s * t_g];
    let coverage = 1.0 - t_s * t_g * t_s;
    let sum_aw = a_s * w[0] + a_g * w[1] + a_s * w[2];
    let scale = coverage / sum_aw;
    let mut expected = [Vec4::new(0.0, 0.0, 0.0, 1.0); 2];
    for (variant, tint_c) in [([1.0f32; 3], 0), (tint, 1)].map(|(t, i)| (t, i)) {
        for c in 0..3 {
            let emission = 0.5 * a_s * w[0] * w[0]
                + 0.1 * a_g * w[1] * w[1]
                + 0.5 * a_s * w[2] * w[2] * variant[c];
            let bg = base[c] * (t_s * t_g * t_s) * variant[c];
            expected[tint_c][c] = emission * scale + bg;
        }
    }

    for pixel in [
        UVec2::new(W / 2, H / 2),
        UVec2::new(W / 2 - 4, H / 2 + 3),
        UVec2::new(W / 2 + 5, H / 2 - 2),
    ] {
        let idx = (pixel.y * W + pixel.x) as usize;
        for c in 0..3 {
            let err_control = (img_control[idx][c] - expected[0][c]).abs();
            let err_tinted = (img_tinted[idx][c] - expected[1][c]).abs();
            assert!(
                err_control <= 2.5e-2,
                "control pixel {pixel:?} ch {c}: gpu {} vs law {}",
                img_control[idx][c],
                expected[0][c]
            );
            assert!(
                err_tinted <= 2.5e-2,
                "tinted pixel {pixel:?} ch {c}: gpu {} vs law {}",
                img_tinted[idx][c],
                expected[1][c]
            );
        }
        // The tint must actually separate the channels: red survives the
        // glass, blue does not.
        assert!(
            img_tinted[idx].x > img_tinted[idx].z + 0.08,
            "no chromatic separation at {pixel:?}: {:?}",
            img_tinted[idx]
        );
        // And the control stayed gray.
        assert!(
            (img_control[idx].x - img_control[idx].z).abs() < 5.0e-3,
            "control drifted off gray at {pixel:?}: {:?}",
            img_control[idx]
        );
    }

    gpu.free(readback);
    gpu.free(tinted_ptr);
    gpu.free(control_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

/// Tests independent per-channel extinction overflow.
///
/// Blue saturates while red remains transmissive.
#[test]
fn oit_rgb_overflow_saturates_per_channel() {
    const W: u32 = 48;
    const H: u32 = 30;
    const PANES: usize = 6;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 90.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.4, 0.4, 0.4, 1.0);
    let dials = FogDials {
        density: 0.0,
        f_min: 96.0,
        f_max: 96.0,
        ..Default::default()
    };
    let light_inputs = test_light_inputs();
    let camera = Vec3::from_array(view.camera_position);
    let forward = Vec3::from_array(view.camera_forward);
    // Six panes exceed the blue channel's 10-bit capacity.
    let pane = OitParticle {
        pos: (camera + forward * 30.0).to_array(),
        size: 50.0,
        color: [0.0; 3],
        alpha: 0.02,
        tint_od: abi_light::tint_od_from_transmittance([1.0, 1.0, 0.02]),
        _pad: 0,
    };
    let panes = [pane; PANES];

    let mut heap = gpu.heap_slots_create(16, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let mut fog = VolumetricPasses::new(&gpu, &mut heap);
    fog.resize(&gpu, &mut heap, size);
    let ptr = upload_particles(&gpu, &panes);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);
    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        ptr.gpu,
        PANES as u32,
        true,
        readback,
    );
    let img = read_hdr(readback, size);

    let t_scalar = (1.0f32 - 0.02).powi(PANES as i32);
    let coverage = 1.0 - t_scalar;
    let pixel = UVec2::new(W / 2, H / 2);
    let got = img[(pixel.y * W + pixel.x) as usize];
    // Red remains at scalar background transmittance.
    let want_r = base.x * t_scalar;
    assert!(
        (got.x - want_r).abs() <= 2.5e-2,
        "red should ride through: got {} want {want_r} (coverage {coverage})",
        got.x
    );
    // Blue overflow saturates the background.
    assert!(
        got.z <= 1.0e-2,
        "blue must saturate to zero transmission: got {}",
        got.z
    );
    // Green remains close to red after carry quantization.
    assert!(
        (got.y - got.x).abs() <= 4.0e-2,
        "carry corrupted green: {got:?}"
    );

    gpu.free(readback);
    gpu.free(ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

/// CPU-only coverage fixture at the shipping 160×100 froxel grid. Its 13
/// y-tiles cover 12.5 froxel rows, exposing the old NDC-to-tile split that
/// starved rows; divisible GPU fixtures do not exercise that tail.
///
/// Every froxel center must reach the tile sampled by `fog_light`.
#[test]
fn fog_local_light_grid_covers_every_froxel_at_shipping_resolution() {
    let froxel_view = View {
        output_size: [FROXEL_WIDTH, FROXEL_HEIGHT],
        ..view(UVec2::new(1280, 800))
    };
    let tiles = [
        FROXEL_WIDTH.div_ceil(FOG_LIGHT_TILE),
        FROXEL_HEIGHT.div_ceil(FOG_LIGHT_TILE),
    ];
    let camera = Vec3::from_array(froxel_view.camera_position);
    let forward = Vec3::from_array(froxel_view.camera_forward);
    for fy in 0..FROXEL_HEIGHT {
        for fx in 0..FROXEL_WIDTH {
            let dir = ray_direction(&froxel_view, UVec2::new(fx, fy));
            let pos = camera + dir * (20.0 / forward.dot(dir));
            let light = PointLight {
                position: pos.to_array(),
                radius: 0.05,
                color: [1.0; 3],
                intensity: 1.0,
            };
            let b = fog_light_tile_bounds(&froxel_view, &light, tiles);
            let (tx, ty) = (fx / FOG_LIGHT_TILE, fy / FOG_LIGHT_TILE);
            assert!(
                b[0] <= tx && tx <= b[2] && b[1] <= ty && ty <= b[3],
                "froxel ({fx}, {fy}): consumer tile ({tx}, {ty}) outside culled {b:?}"
            );
        }
    }
}

/// Compares local point-light froxel lighting with a CPU reference.
///
/// Sun and ambient are zero, isolating local contributions.
#[test]
fn fog_local_point_light_matches_cpu_twin() {
    const W: u32 = 48;
    const H: u32 = 30;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 40.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.015, 0.02, 0.03, 1.0);
    let dials = FogDials {
        density: 0.035,
        height_falloff: 0.0,
        anisotropy: 0.25,
        ambient_color: [0.0; 3],
        f_min: 48.0,
        f_max: 48.0,
        ..Default::default()
    };
    let camera = Vec3::from_array(view.camera_position);
    let light = PointLight {
        position: (camera + Vec3::new(2.0, 1.0, 18.0)).to_array(),
        radius: 11.0,
        color: [1.0, 0.35, 0.12],
        intensity: 90.0,
    };
    let light_ptr = gpu.alloc::<PointLight>(gpu::Memory::Default);
    // SAFETY: fresh host-visible allocation sized for one PointLight.
    unsafe { std::ptr::write(light_ptr.cpu, light) };
    let light_inputs = FogLightInputs {
        sun_dir: Vec3::Y.to_array(),
        sun_color: [0.0; 3],
        occluder: None,
        local_lights: light_ptr.gpu,
        local_light_count: 1,
    };
    let local_lights = [light];
    let twin = twin_from_local(&light_inputs, None, 0, &local_lights);

    let mut heap = gpu.heap_slots_create(8, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let fog = VolumetricPasses::new(&gpu, &mut heap);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        GpuPtr::null(),
        0,
        false,
        readback,
    );

    let got_params = fog.curve_after_idle();
    let img = read_hdr(readback, size);
    let mut local_energy = 0.0f32;
    for pixel in [
        UVec2::new(W / 2, H / 2),
        UVec2::new(W / 2 + 5, H / 2 - 2),
        UVec2::new(W / 2 - 8, H / 2 + 4),
    ] {
        let got = img[(pixel.y * W + pixel.x) as usize];
        let want = cpu_composite_pixel(&got_params, &view, &dials, &twin, pixel, depth_value, base);
        local_energy += want.truncate().length();
        for c in 0..4 {
            let err = (got[c] - want[c]).abs();
            assert!(
                err <= 8.0e-3,
                "local-light pixel {pixel:?} channel {c}: gpu {} vs cpu {} (err {err})",
                got[c],
                want[c]
            );
        }
    }
    assert!(
        local_energy > 0.1,
        "weak local-light fixture: {local_energy}"
    );

    gpu.free(readback);
    gpu.free(light_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

/// Overflowed light tiles must use the complete bounded array.
///
/// Forty overlapping lights must match the CPU reference.
#[test]
fn fog_local_light_grid_overflow_keeps_every_light() {
    const W: u32 = 32;
    const H: u32 = 20;
    const LIGHT_COUNT: usize = 40;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let scene_z = 36.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.01, 0.01, 0.01, 1.0);
    let dials = FogDials {
        density: 0.03,
        height_falloff: 0.0,
        anisotropy: 0.0,
        ambient_color: [0.0; 3],
        f_min: 40.0,
        f_max: 40.0,
        ..Default::default()
    };
    let camera = Vec3::from_array(view.camera_position);
    let lights = [PointLight {
        position: (camera + Vec3::new(0.0, 0.0, 16.0)).to_array(),
        radius: 9.0,
        color: [0.7, 0.4, 0.2],
        intensity: 2.0,
    }; LIGHT_COUNT];
    let light_ptr = gpu.alloc_slice::<PointLight>(LIGHT_COUNT as u64, gpu::Memory::Default);
    // SAFETY: fresh host-visible allocation sized for the complete array.
    unsafe { std::ptr::copy_nonoverlapping(lights.as_ptr(), light_ptr.cpu, LIGHT_COUNT) };
    let light_inputs = FogLightInputs {
        sun_dir: Vec3::Y.to_array(),
        sun_color: [0.0; 3],
        occluder: None,
        local_lights: light_ptr.gpu,
        local_light_count: LIGHT_COUNT as u32,
    };
    let twin = twin_from_local(&light_inputs, None, 0, &lights);

    let mut heap = gpu.heap_slots_create(8, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let fog = VolumetricPasses::new(&gpu, &mut heap);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);
    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        GpuPtr::null(),
        0,
        false,
        readback,
    );

    let params = fog.curve_after_idle();
    let pixel = UVec2::new(W / 2, H / 2);
    let got = read_hdr(readback, size)[(pixel.y * W + pixel.x) as usize];
    let want = cpu_composite_pixel(&params, &view, &dials, &twin, pixel, depth_value, base);
    assert!(want.x > 0.1, "weak overflow fixture: {want}");
    for c in 0..4 {
        let err = (got[c] - want[c]).abs();
        assert!(
            err <= 1.2e-2,
            "overflow local-light channel {c}: gpu {} vs all-40 cpu {} (err {err})",
            got[c],
            want[c]
        );
    }

    gpu.free(readback);
    gpu.free(light_ptr);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

#[test]
fn fog_occluder_god_ray_column() {
    const W: u32 = 24;
    const H: u32 = 16;
    const OCC_DIM: u32 = 16;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    // The camera views a fixed occluder box from just outside.
    let view = View {
        camera_position: [8.0, 12.0, -4.0],
        ..view(size)
    };
    let scene_z = 20.0f32;
    let depth_value = view.depth_near_plane / scene_z;
    let base = Vec4::new(0.01, 0.012, 0.014, 1.0);

    let mut occ_data = vec![0u8; (OCC_DIM * OCC_DIM * OCC_DIM) as usize];
    for z in 0..OCC_DIM {
        for y in 10..14 {
            for x in 0..OCC_DIM {
                occ_data[(x + y * OCC_DIM + z * OCC_DIM * OCC_DIM) as usize] = 255;
            }
        }
    }
    let cpu_occluder = CpuOccluder {
        dims: UVec2::new(OCC_DIM, OCC_DIM),
        depth: OCC_DIM,
        data: occ_data.clone(),
        world_min: Vec3::ZERO,
        world_inv_extent: Vec3::splat(1.0 / OCC_DIM as f32),
    };

    let dials = FogDials {
        density: 0.04,
        // Uniform fog extinguishes the sun before the occluder appears.
        height_falloff: 0.04,
        anisotropy: 0.0,
        ambient_color: [0.01, 0.01, 0.012],
        gradient_bottom: [1.0; 3],
        gradient_top: [1.0; 3],
        sun_steps: 4,
        sun_lod_ramp: 0.0,
        f_max: 32.0,
        ..Default::default()
    };

    let mut heap = gpu.heap_slots_create(12, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (occluder_texture, occluder_slot) = upload_occluder_texture(
        &gpu,
        &mut heap,
        UVec2::new(OCC_DIM, OCC_DIM),
        OCC_DIM,
        &occ_data,
    );
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let fog = VolumetricPasses::new(&gpu, &mut heap);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);
    let light_inputs = FogLightInputs {
        sun_dir: Vec3::Y.to_array(),
        sun_color: [2.4, 1.8, 1.1],
        occluder: Some(OccluderVolume {
            texture: occluder_slot,
            sampler: clamp_sampler,
            world_min: [0.0; 3],
            world_inv_extent: [1.0 / OCC_DIM as f32; 3],
        }),
        local_lights: GpuPtr::null(),
        local_light_count: 0,
    };

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        depth_value,
        base,
        GpuPtr::null(),
        0,
        false,
        readback,
    );

    let got_params = fog.curve_after_idle();
    let want_params = froxel_params_from(
        scene_z,
        dials.slice_count,
        dials.a,
        dials.f_min,
        dials.f_max,
    );
    assert_params_close(got_params.params, want_params);

    let shadow_pixel = UVec2::new(W / 2, H - 1);
    let lit_pixel = UVec2::new(W / 2, 0);
    let probe_z = z_of_warped_slice(&got_params, 30.5);
    let probe_pos = |pixel| {
        let dir = ray_direction(&view, pixel);
        let forward = Vec3::from_array(view.camera_forward);
        let view_to_ray = 1.0 / forward.dot(dir).max(1.0e-6);
        Vec3::from_array(view.camera_position) + dir * (probe_z * view_to_ray)
    };
    let shadow_probe = probe_pos(shadow_pixel);
    let lit_probe = probe_pos(lit_pixel);
    let in_box = |p: Vec3| {
        p.x >= 0.0 && p.x <= 16.0 && p.y >= 0.0 && p.y <= 16.0 && p.z >= 0.0 && p.z <= 16.0
    };
    assert!(
        in_box(shadow_probe),
        "shadow probe outside occluder box: {shadow_probe:?}"
    );
    assert!(
        in_box(lit_probe),
        "lit probe outside occluder box: {lit_probe:?}"
    );
    assert!(
        shadow_probe.y < 10.0,
        "shadow probe must sit below the opaque slab: {shadow_probe:?}"
    );
    assert!(
        lit_probe.y > 14.0,
        "lit probe must sit above the opaque slab: {lit_probe:?}"
    );
    let shadow_vis = cpu_occluder_visibility(&cpu_occluder, dials.sun_steps, shadow_probe, Vec3::Y);
    let lit_vis = cpu_occluder_visibility(&cpu_occluder, dials.sun_steps, lit_probe, Vec3::Y);
    assert!(shadow_vis < 0.1, "shadow probe visibility {shadow_vis}");
    assert!(lit_vis > 0.9, "lit probe visibility {lit_vis}");

    let twin = twin_from(&light_inputs, Some(&cpu_occluder), dials.sun_steps);
    let img = read_hdr(readback, size);
    let shadow_gpu = img[(shadow_pixel.y * W + shadow_pixel.x) as usize];
    let lit_gpu = img[(lit_pixel.y * W + lit_pixel.x) as usize];
    let shadow_cpu = cpu_composite_pixel(
        &got_params,
        &view,
        &dials,
        &twin,
        shadow_pixel,
        depth_value,
        base,
    );
    let lit_cpu = cpu_composite_pixel(
        &got_params,
        &view,
        &dials,
        &twin,
        lit_pixel,
        depth_value,
        base,
    );

    for (label, got, want) in [
        ("shadow", shadow_gpu, shadow_cpu),
        ("lit", lit_gpu, lit_cpu),
    ] {
        // R8 sampling, slice quantization, and FP16 storage set this tolerance.
        for c in 0..4 {
            let err = (got[c] - want[c]).abs();
            assert!(
                err <= 2.0e-2,
                "{label} pixel channel {c}: gpu {} vs cpu {} (err {err})",
                got[c],
                want[c]
            );
        }
    }

    let expected_gap = lit_cpu.x - shadow_cpu.x;
    let got_gap = lit_gpu.x - shadow_gpu.x;
    assert!(
        shadow_cpu.x < lit_cpu.x * 0.5,
        "CPU twin red gap: shadow {shadow_cpu} lit {lit_cpu}"
    );
    assert!(
        shadow_gpu.x < lit_gpu.x * 0.5,
        "GPU red ordering: shadow {shadow_gpu} lit {lit_gpu}"
    );
    assert!(
        got_gap >= expected_gap - 4.0e-2,
        "GPU gap {got_gap} should track CPU gap {expected_gap}: shadow {shadow_gpu} lit {lit_gpu}"
    );

    gpu.free(readback);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(occluder_texture);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

/// Checks the all-sky analytic continuation.
///
/// Reverse-Z zero depth clamps `f` to `f_min`; downward rays converge to fog.
#[test]
fn fog_sky_pixels_use_analytic_beyond_f() {
    const W: u32 = 31;
    const H: u32 = 17;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let base = Vec4::new(0.1, 0.2, 0.3, 1.0);
    let dials = FogDials {
        density: 0.05,
        height_falloff: 0.2,
        height_offset: -3.0,
        ..Default::default()
    };
    let light_inputs = test_light_inputs();
    let twin = twin_from(&light_inputs, None, 0);

    let mut heap = gpu.heap_slots_create(8, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let fog = VolumetricPasses::new(&gpu, &mut heap);
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);

    render_volumetric_frame(
        &gpu,
        &heap,
        &fog,
        hdr.texture,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        0.0,
        base,
        GpuPtr::null(),
        0,
        false,
        readback,
    );

    let got_params = fog.curve_after_idle();
    assert_eq!(
        got_params.params.f.to_bits(),
        dials.f_min.to_bits(),
        "all-sky frame must clamp f to f_min"
    );

    let img = read_hdr(readback, size);
    for pixel in [
        UVec2::new(15, 1),  // Upward ray.
        UVec2::new(15, 8),  // Near-horizontal ray.
        UVec2::new(2, 15),  // Downward ray.
        UVec2::new(28, 15), // Downward opposite corner.
    ] {
        let got = img[(pixel.y * W + pixel.x) as usize];
        let want = cpu_composite_pixel(&got_params, &view, &dials, &twin, pixel, 0.0, base);
        for c in 0..4 {
            let err = (got[c] - want[c]).abs();
            assert!(
                err <= 4.0e-3,
                "pixel {pixel:?} channel {c}: gpu {} vs cpu {} (err {err})",
                got[c],
                want[c]
            );
        }
    }

    gpu.free(readback);
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}

#[test]
fn fog_zero_density_and_zero_particles_leaves_hdr_untouched() {
    const W: u32 = 19;
    const H: u32 = 13;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let view = view(size);
    let depth_value = view.depth_near_plane / 32.0;
    let base = Vec4::new(0.25, 0.5, 0.75, 1.0);

    let mut heap = gpu.heap_slots_create(8, 8, 4);
    let clamp_sampler = clamp_sampler(&gpu, &mut heap);
    let (hdr, depth, depth_slot, hdr_rw) = setup_targets(&gpu, &mut heap, size);
    let fog = VolumetricPasses::new(&gpu, &mut heap);
    let mut frame = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let readback = gpu.alloc_slice::<u16>((W * H * 4) as u64, gpu::Memory::Readback);
    let dials = FogDials {
        density: 0.0,
        ..Default::default()
    };
    let light_inputs = test_light_inputs();
    let _twin = twin_from(&light_inputs, None, 0);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            color_attachments: &[RenderAttachment {
                texture: hdr.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: base.to_array(),
                ..Default::default()
            }],
            depth_attachment: Some(RenderAttachment {
                texture: depth.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [depth_value, 0.0, 0.0, 0.0],
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    gpu.cmd_end_render_pass(cb);
    gpu.cmd_barrier(cb, Stage::All, Stage::Compute, HazardFlags::empty());
    heap.bind(&gpu, cb);
    fog.record(
        &gpu,
        cb,
        &mut frame,
        depth.texture,
        depth_slot,
        hdr_rw,
        clamp_sampler,
        &view,
        &dials,
        &light_inputs,
        GpuPtr::null(),
        0,
        false,
    );
    assert!(
        frame.ptrs.is_empty(),
        "zero-density/zero-particle record must allocate no dispatch data"
    );
    gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_copy_texture_to_buffer(cb, hdr.texture, readback.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let img = read_hdr(readback, size);
    for y in 0..H {
        for x in 0..W {
            let got = img[(y * W + x) as usize];
            assert_eq!(got, base, "pixel ({x},{y})");
        }
    }

    gpu.free(readback);
    frame.free();
    fog.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(depth);
    heap.free(&gpu);
}
