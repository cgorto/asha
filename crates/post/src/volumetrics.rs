//! Froxel volumetrics with depth reduction, lighting, integration, and OIT.
//!
//! Callers must bind descriptor heaps before recording.
//! With zero density and no particles, [`VolumetricPasses::record`] records no
//! GPU work. Otherwise, HDR writes become visible to fragment consumers.
//!
//! Zero-transmittance priming marks columns whose throughput reaches
//! `ZERO_TRANS_EPS`, the extinction saturation threshold. It writes
//! conservative cover depths into
//! the caller's reverse-Z buffer before OIT accumulation; GREATER can move only
//! nearer, so opaque geometry is not lost. Read pristine depth first.

use abi_core::GpuPtr;
use abi_core::View;
use abi_core::glam::UVec2;
use abi_light::PointLight;
use abi_light::{
    FOG_LIGHT_TILE, FOG_LIGHTS_PER_TILE, FOG_LOCAL_LIGHT_MAX, FOG_SLICE_MAX, FogCompositeData,
    FogCurve, FogDepthMaxData, FogIntegrateData, FogLightData, FogLightGridData, FogParamsData,
    FogPrimeQuad, FogPrimeSpawnData, FogPrimeVertData, OitAccumFragData, OitAccumVertData,
    OitParticle, OitResolveData, OitSplatFragData, OitSplatVertData, PRIME_TILE,
};
use gpu::{
    BlendFactor, BlendOp, BlendState, ColorComponentFlags, CommandBuffer, CompareOp, DepthFlags,
    DepthState, Gpu, HazardFlags, HeapSlots, LoadOp, Memory, OwnedTexture, Queue, RenderAttachment,
    RenderPassDesc, SampledSlot, SamplerSlot, ShaderTypeGraphics, Stage, StorageSlot, StoreOp,
    Texture, TextureDesc, TextureFormat, TextureType, TextureViewDesc, UsageFlags,
};
use std::mem::size_of;

use gpu::pass::{FrameAlloc, Pass};

pub const FROXEL_WIDTH: u32 = 160;
pub const FROXEL_HEIGHT: u32 = 100;
pub const FROXEL_DEPTH: u32 = 64;
/// Fixed stage order emitted by `record_profiled`.
///
/// Hosts can assign timestamps by name.
pub const FOG_PROFILE_NAMES: [&str; 9] = [
    "fog_prepare",
    "fog_light_grid",
    "fog_light",
    "fog_splat",
    "fog_integrate",
    "fog_composite",
    "fog_prime",
    "fog_accum",
    "fog_resolve",
];
const FROXEL_LIGHT_TILES_X: u32 = FROXEL_WIDTH.div_ceil(FOG_LIGHT_TILE);
const FROXEL_LIGHT_TILES_Y: u32 = FROXEL_HEIGHT.div_ceil(FOG_LIGHT_TILE);
const FROXEL_LIGHT_TILE_COUNT: u32 = FROXEL_LIGHT_TILES_X * FROXEL_LIGHT_TILES_Y;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogDials {
    pub density: f32,
    pub height_falloff: f32,
    pub height_offset: f32,
    pub fog_sample_bias: f32,
    /// Henyey-Greenstein g for the sun term; 0 = isotropic.
    pub anisotropy: f32,
    pub ambient_color: [f32; 3],
    pub gradient_bottom: [f32; 3],
    pub gradient_top: [f32; 3],
    pub gradient_offset: f32,
    pub gradient_length: f32,
    /// Occluder-march taps toward the sun; zero disables the march.
    pub sun_steps: u32,
    pub sun_lod_ramp: f32,
    pub slice_count: u32,
    pub a: f32,
    pub f_min: f32,
    pub f_max: f32,
    /// Depth-warp strength toward OIT-event depths.
    ///
    /// Zero preserves the unwarped logarithmic curve.
    pub warp_gain: f32,
    /// Bounded-warp law: every slice's width stays within
    /// `[1/warp_bound, warp_bound]` of its log-curve width, so distant
    /// event clusters can never starve near-camera fog. <= 1 disables.
    pub warp_bound: f32,
}

impl Default for FogDials {
    fn default() -> Self {
        Self {
            density: 0.02,
            height_falloff: 0.15,
            height_offset: 0.0,
            fog_sample_bias: 2.0,
            anisotropy: 0.35,
            ambient_color: [0.38, 0.40, 0.45],
            gradient_bottom: [1.0; 3],
            gradient_top: [1.0; 3],
            gradient_offset: 30.0,
            gradient_length: 25.0,
            sun_steps: 6,
            sun_lod_ramp: 0.75,
            slice_count: FROXEL_DEPTH,
            a: 4.0,
            f_min: 4.0,
            f_max: 512.0,
            warp_gain: 0.0,
            warp_bound: 4.0,
        }
    }
}

/// Generalized occluder volume for the sun march — any producer that can
/// present occupancy as a sampled 3D opacity texture over an AABB: mesh
/// voxelization, SDF-derived opacity, or baked
/// occupancy. Value in [0, 1] is opacity at
/// `uvw = (pos - world_min) * world_inv_extent`.
#[derive(Clone, Copy)]
pub struct OccluderVolume {
    pub texture: SampledSlot,
    pub sampler: SamplerSlot,
    pub world_min: [f32; 3],
    pub world_inv_extent: [f32; 3],
}

/// Per-frame lighting inputs the host owns: the fog's sun must be the same
/// sun the sky/shading uses.
#[derive(Clone, Copy)]
pub struct FogLightInputs {
    pub sun_dir: [f32; 3],
    pub sun_color: [f32; 3],
    pub occluder: Option<OccluderVolume>,
    /// Local point lights sharing surface lighting's finite-radius law.
    /// Zero count disables local-light processing.
    pub local_lights: GpuPtr<PointLight>,
    pub local_light_count: u32,
}

struct OitTargets {
    size: UVec2,
    accum_rgb: OwnedTexture,
    accum_moments: OwnedTexture,
    /// Zero-transmittance cover quads, one slot per screen tile.
    prime_quads: gpu::Ptr<FogPrimeQuad>,
}

pub struct VolumetricPasses {
    v_ext: gpu::Ptr<u32>,
    overflow: gpu::Ptr<u32>,
    /// Per froxel column: first slice where throughput hit the zero floor.
    zero_slice: gpu::Ptr<u32>,
    /// Event occupancy bits for each column and warped slice.
    occupancy: gpu::Ptr<u32>,
    /// Packed RGB extinction words, one per column and slice, written
    /// only by tinted draws; cleared only on tinted frames.
    v_ext_rgb: gpu::Ptr<u32>,
    occupancy_rgb: gpu::Ptr<u32>,
    /// Per-channel first-overflow min slice, channel-major.
    overflow_rgb: gpu::Ptr<u32>,
    /// Chromatic transmittance multiplier volume: V_int.a per channel on
    /// top of the scalar integral. Written/read only on tinted frames.
    v_tint: OwnedTexture,
    v_tint_slot: SampledSlot,
    v_tint_rw: StorageSlot,
    /// Tiled local-light grid with bounded retained lists.
    /// Overflow tiles fall back to all lights.
    local_tile_counts: gpu::Ptr<u32>,
    local_tile_indices: gpu::Ptr<u32>,
    local_tile_overflow: gpu::Ptr<u32>,
    /// `DrawIndexedIndirectCommand` as 5 dwords; dword 1 (instance_count)
    /// is bumped by `fog_prime_spawn`. Host-visible for test readback.
    prime_draw_args: gpu::Ptr<u32>,
    splat_target: OwnedTexture,
    accum: Option<OitTargets>,
    accum_rgb_slot: SampledSlot,
    accum_moments_slot: SampledSlot,
    v_int: OwnedTexture,
    v_int_slot: SampledSlot,
    v_int_rw: StorageSlot,
    v_scatter: OwnedTexture,
    v_scatter_slot: SampledSlot,
    v_scatter_rw: StorageSlot,
    max_depth_bits: gpu::Ptr<u32>,
    curve: gpu::Ptr<FogCurve>,
    /// Raw-slice OIT histogram consumed by the next frame's warp build.
    /// Host-visible for test readback; zeroed at creation.
    hist: gpu::Ptr<u32>,
    quad_indices: gpu::Ptr<u32>,
    depth_max_shader: gpu::Shader,
    params_shader: gpu::Shader,
    prime_spawn_shader: gpu::Shader,
    light_grid_shader: gpu::Shader,
    prime_vert_shader: gpu::Shader,
    prime_frag_shader: gpu::Shader,
    oit_splat_vert_shader: gpu::Shader,
    oit_splat_frag_shader: gpu::Shader,
    light_shader: gpu::Shader,
    integrate_shader: gpu::Shader,
    composite_shader: gpu::Shader,
    oit_accum_vert_shader: gpu::Shader,
    oit_accum_frag_shader: gpu::Shader,
    oit_resolve_shader: gpu::Shader,
}

impl VolumetricPasses {
    pub fn new(gpu: &Gpu, heap: &mut HeapSlots) -> Self {
        let splat_target = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [FROXEL_WIDTH, FROXEL_HEIGHT, 1],
                format: TextureFormat::R8Unorm,
                usage: UsageFlags::COLOR_ATTACHMENT,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let v_int = gpu.texture_alloc_and_create(
            TextureDesc {
                ty: TextureType::D3,
                dimensions: [FROXEL_WIDTH, FROXEL_HEIGHT, FROXEL_DEPTH],
                format: TextureFormat::Rgba16Float,
                usage: UsageFlags::SAMPLED | UsageFlags::STORAGE,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let view_desc = TextureViewDesc {
            ty: TextureType::D3,
            ..Default::default()
        };
        let v_int_slot =
            heap.add_sampled(gpu, gpu.texture_view_descriptor(v_int.texture, view_desc));
        let v_int_rw = heap.add_storage(
            gpu,
            gpu.texture_rw_view_descriptor(v_int.texture, view_desc),
        );
        let v_tint = gpu.texture_alloc_and_create(
            TextureDesc {
                ty: TextureType::D3,
                dimensions: [FROXEL_WIDTH, FROXEL_HEIGHT, FROXEL_DEPTH],
                format: TextureFormat::Rgba16Float,
                usage: UsageFlags::SAMPLED | UsageFlags::STORAGE,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let v_tint_slot =
            heap.add_sampled(gpu, gpu.texture_view_descriptor(v_tint.texture, view_desc));
        let v_tint_rw = heap.add_storage(
            gpu,
            gpu.texture_rw_view_descriptor(v_tint.texture, view_desc),
        );
        // Per-froxel scattering and extinction from fog lighting.
        let v_scatter = gpu.texture_alloc_and_create(
            TextureDesc {
                ty: TextureType::D3,
                dimensions: [FROXEL_WIDTH, FROXEL_HEIGHT, FROXEL_DEPTH],
                format: TextureFormat::Rgba16Float,
                usage: UsageFlags::SAMPLED | UsageFlags::STORAGE,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let v_scatter_slot = heap.add_sampled(
            gpu,
            gpu.texture_view_descriptor(v_scatter.texture, view_desc),
        );
        let v_scatter_rw = heap.add_storage(
            gpu,
            gpu.texture_rw_view_descriptor(v_scatter.texture, view_desc),
        );
        let quad_indices = gpu.alloc_slice::<u32>(6, Memory::Default);
        // SAFETY: fresh host-visible allocation, six indices for the two
        // billboard triangles. The shaders use vertex_index 0..5.
        unsafe {
            std::ptr::copy_nonoverlapping([0u32, 1, 2, 3, 4, 5].as_ptr(), quad_indices.cpu, 6);
        }
        let hist = gpu.alloc_slice::<u32>(FOG_SLICE_MAX as u64, Memory::Default);
        // SAFETY: fresh host-visible allocation; fog_params reads the
        // histogram before any splat has run, so it must start empty.
        unsafe { std::ptr::write_bytes(hist.cpu, 0, FOG_SLICE_MAX as usize) };

        Self {
            v_ext: gpu.alloc_slice::<u32>(v_ext_dwords() as u64, Memory::Gpu),
            overflow: gpu.alloc_slice::<u32>((FROXEL_WIDTH * FROXEL_HEIGHT) as u64, Memory::Gpu),
            zero_slice: gpu.alloc_slice::<u32>((FROXEL_WIDTH * FROXEL_HEIGHT) as u64, Memory::Gpu),
            occupancy: gpu
                .alloc_slice::<u32>((FROXEL_WIDTH * FROXEL_HEIGHT * 2) as u64, Memory::Gpu),
            v_ext_rgb: gpu.alloc_slice::<u32>(
                (FROXEL_WIDTH * FROXEL_HEIGHT * FROXEL_DEPTH) as u64,
                Memory::Gpu,
            ),
            occupancy_rgb: gpu
                .alloc_slice::<u32>((FROXEL_WIDTH * FROXEL_HEIGHT * 2) as u64, Memory::Gpu),
            overflow_rgb: gpu
                .alloc_slice::<u32>((FROXEL_WIDTH * FROXEL_HEIGHT * 3) as u64, Memory::Gpu),
            v_tint,
            v_tint_slot,
            v_tint_rw,
            local_tile_counts: gpu.alloc_slice::<u32>(FROXEL_LIGHT_TILE_COUNT as u64, Memory::Gpu),
            local_tile_indices: gpu.alloc_slice::<u32>(
                (FROXEL_LIGHT_TILE_COUNT * FOG_LIGHTS_PER_TILE) as u64,
                Memory::Gpu,
            ),
            local_tile_overflow: gpu
                .alloc_slice::<u32>(FROXEL_LIGHT_TILE_COUNT as u64, Memory::Gpu),
            prime_draw_args: gpu.alloc_slice::<u32>(5, Memory::Default),
            splat_target,
            accum: None,
            accum_rgb_slot: heap.alloc_sampled(),
            accum_moments_slot: heap.alloc_sampled(),
            v_int,
            v_int_slot,
            v_int_rw,
            v_scatter,
            v_scatter_slot,
            v_scatter_rw,
            max_depth_bits: gpu.alloc::<u32>(Memory::Gpu),
            curve: gpu.alloc::<FogCurve>(Memory::Default),
            hist,
            quad_indices,
            depth_max_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("fog_depth_max"),
                32,
                32,
                1,
                "fog_depth_max",
            ),
            params_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("fog_params"),
                1,
                1,
                1,
                "fog_params",
            ),
            prime_spawn_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("fog_prime_spawn"),
                8,
                8,
                1,
                "fog_prime_spawn",
            ),
            light_grid_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("fog_light_grid"),
                64,
                1,
                1,
                "fog_light_grid",
            ),
            prime_vert_shader: gpu.shader_create(
                &asha_assets::load_spv("fog_prime_vert"),
                ShaderTypeGraphics::Vertex,
                "fog_prime_vert",
            ),
            prime_frag_shader: gpu.shader_create(
                &asha_assets::load_spv("fog_prime_frag"),
                ShaderTypeGraphics::Fragment,
                "fog_prime_frag",
            ),
            oit_splat_vert_shader: gpu.shader_create(
                &asha_assets::load_spv("oit_splat_vert"),
                ShaderTypeGraphics::Vertex,
                "oit_splat_vert",
            ),
            oit_splat_frag_shader: gpu.shader_create(
                &asha_assets::load_spv("oit_splat_frag"),
                ShaderTypeGraphics::Fragment,
                "oit_splat_frag",
            ),
            light_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("fog_light"),
                8,
                8,
                1,
                "fog_light",
            ),
            integrate_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("fog_integrate"),
                8,
                8,
                1,
                "fog_integrate",
            ),
            composite_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("fog_composite"),
                8,
                8,
                1,
                "fog_composite",
            ),
            oit_accum_vert_shader: gpu.shader_create(
                &asha_assets::load_spv("oit_accum_vert"),
                ShaderTypeGraphics::Vertex,
                "oit_accum_vert",
            ),
            oit_accum_frag_shader: gpu.shader_create(
                &asha_assets::load_spv("oit_accum_frag"),
                ShaderTypeGraphics::Fragment,
                "oit_accum_frag",
            ),
            oit_resolve_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("oit_resolve"),
                8,
                8,
                1,
                "oit_resolve",
            ),
        }
    }

    /// Resize private OIT accumulation targets.
    ///
    /// Waits before freeing in-flight images.
    pub fn resize(&mut self, gpu: &Gpu, heap: &mut HeapSlots, size: UVec2) {
        assert!(size.x > 0 && size.y > 0);
        assert!(
            size.x.div_ceil(PRIME_TILE) <= u16::MAX as u32
                && size.y.div_ceil(PRIME_TILE) <= u16::MAX as u32,
            "fog-prime tile coordinates pack into 16 bits per axis"
        );
        if self.accum.as_ref().is_some_and(|a| a.size == size) {
            return;
        }
        if let Some(old) = self.accum.take() {
            gpu.queue_wait_idle(Queue::Main);
            gpu.texture_free_and_destroy(old.accum_rgb);
            gpu.texture_free_and_destroy(old.accum_moments);
            gpu.free(old.prime_quads);
        }

        let accum_rgb = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [size.x, size.y, 1],
                format: TextureFormat::Rgba16Float,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::SAMPLED,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let accum_moments = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [size.x, size.y, 1],
                format: TextureFormat::Rg16Float,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::SAMPLED,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        heap.write_sampled(
            gpu,
            self.accum_rgb_slot,
            gpu.texture_view_descriptor(accum_rgb.texture, TextureViewDesc::default()),
        );
        heap.write_sampled(
            gpu,
            self.accum_moments_slot,
            gpu.texture_view_descriptor(accum_moments.texture, TextureViewDesc::default()),
        );
        let tiles = size.x.div_ceil(PRIME_TILE) as u64 * size.y.div_ceil(PRIME_TILE) as u64;
        self.accum = Some(OitTargets {
            size,
            accum_rgb,
            accum_moments,
            prime_quads: gpu.alloc_slice::<FogPrimeQuad>(tiles, Memory::Gpu),
        });
    }

    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        depth_texture: Texture,
        depth_slot: SampledSlot,
        hdr_storage_slot: StorageSlot,
        clamp_sampler: SamplerSlot,
        view: &View,
        dials: &FogDials,
        light: &FogLightInputs,
        particles: GpuPtr<OitParticle>,
        particle_count: u32,
        particles_tinted: bool,
    ) {
        self.record_profiled(
            gpu,
            cb,
            fa,
            depth_texture,
            depth_slot,
            hdr_storage_slot,
            clamp_sampler,
            view,
            dials,
            light,
            particles,
            particle_count,
            particles_tinted,
            |_| {},
        );
    }

    /// Record the pass and report stages in [`FOG_PROFILE_NAMES`] order.
    ///
    /// Idle optional stages are reported to preserve timestamp positions.
    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    pub fn record_profiled(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        depth_texture: Texture,
        depth_slot: SampledSlot,
        hdr_storage_slot: StorageSlot,
        clamp_sampler: SamplerSlot,
        view: &View,
        dials: &FogDials,
        light: &FogLightInputs,
        particles: GpuPtr<OitParticle>,
        particle_count: u32,
        particles_tinted: bool,
        mut stage_end: impl FnMut(&'static str),
    ) {
        if dials.density == 0.0 && particle_count == 0 {
            for name in FOG_PROFILE_NAMES {
                stage_end(name);
            }
            return;
        }
        assert!(
            particle_count == 0 || !particles.is_null(),
            "OIT particle count requires a non-null particle pointer"
        );
        assert!(
            light.local_light_count <= FOG_LOCAL_LIGHT_MAX,
            "fog local-light count {} exceeds bounded maximum {FOG_LOCAL_LIGHT_MAX}",
            light.local_light_count
        );
        assert!(
            light.local_light_count == 0 || !light.local_lights.is_null(),
            "fog local-light count requires a non-null light pointer"
        );

        let [w, h] = view.output_size;
        let size = UVec2::new(w, h);
        // OIT-only frames have no local-light scattering.
        let fog_local_light_count = if dials.density > 0.0 {
            light.local_light_count
        } else {
            0
        };
        if particle_count > 0 {
            assert!(
                self.accum.as_ref().is_some_and(|a| a.size == size),
                "VolumetricPasses::resize must match the screen before OIT record"
            );
        }
        // Integration currently writes the full fixed-depth volume.
        let slice_count = FROXEL_DEPTH;

        gpu.cmd_fill_buffer(cb, self.v_ext, 0, v_ext_bytes());
        gpu.cmd_fill_buffer(cb, self.overflow, u32::MAX, overflow_bytes());
        gpu.cmd_fill_buffer(cb, self.max_depth_bits, 0, size_of::<u32>() as u64);
        if fog_local_light_count > 0 {
            gpu.cmd_fill_buffer(
                cb,
                self.local_tile_counts,
                0,
                FROXEL_LIGHT_TILE_COUNT as u64 * size_of::<u32>() as u64,
            );
            gpu.cmd_fill_buffer(
                cb,
                self.local_tile_overflow,
                0,
                FROXEL_LIGHT_TILE_COUNT as u64 * size_of::<u32>() as u64,
            );
        }
        let tinted = particles_tinted && particle_count > 0;
        if particle_count > 0 {
            gpu.cmd_fill_buffer(
                cb,
                self.occupancy,
                0,
                (FROXEL_WIDTH * FROXEL_HEIGHT * 2) as u64 * size_of::<u32>() as u64,
            );
            // RGB extinction storage is used only for tinted frames.
            if tinted {
                gpu.cmd_fill_buffer(
                    cb,
                    self.v_ext_rgb,
                    0,
                    (FROXEL_WIDTH * FROXEL_HEIGHT * FROXEL_DEPTH) as u64 * size_of::<u32>() as u64,
                );
                gpu.cmd_fill_buffer(
                    cb,
                    self.occupancy_rgb,
                    0,
                    (FROXEL_WIDTH * FROXEL_HEIGHT * 2) as u64 * size_of::<u32>() as u64,
                );
                gpu.cmd_fill_buffer(
                    cb,
                    self.overflow_rgb,
                    u32::MAX,
                    (FROXEL_WIDTH * FROXEL_HEIGHT * 3) as u64 * size_of::<u32>() as u64,
                );
            }
            // DrawIndexedIndirectCommand { index_count: 6, instance_count: 0, .. }
            gpu.cmd_fill_buffer(cb, self.prime_draw_args, 6, size_of::<u32>() as u64);
            gpu.cmd_fill_buffer(
                cb,
                self.prime_draw_args.offset(1),
                0,
                4 * size_of::<u32>() as u64,
            );
        }
        // Fill-buffer writes require a generic transfer barrier.
        gpu.cmd_barrier(cb, Stage::Transfer, Stage::Compute, HazardFlags::empty());

        let depth_data = fa.frame_alloc(FogDepthMaxData {
            max_depth_bits: self.max_depth_bits.gpu,
            depth_texture_id: depth_slot.index(),
            _pad0: 0,
            screen_size: view.output_size,
            depth_near_plane: view.depth_near_plane,
            _pad1: 0,
        });
        gpu.cmd_set_compute_shader(cb, self.depth_max_shader);
        gpu.cmd_dispatch(cb, depth_data, w.div_ceil(32), h.div_ceil(32), 1);
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::Compute,
            HazardFlags::SHADER_BUFFER,
        );

        let params_data = fa.frame_alloc(FogParamsData {
            max_depth_bits: self.max_depth_bits.gpu,
            curve_out: self.curve.gpu,
            hist: self.hist.gpu,
            slice_count,
            a: dials.a,
            f_min: dials.f_min,
            f_max: dials.f_max,
            warp_gain: dials.warp_gain,
            warp_bound: dials.warp_bound,
        });
        gpu.cmd_set_compute_shader(cb, self.params_shader);
        gpu.cmd_dispatch(cb, params_data, 1, 1, 1);
        if particle_count > 0 {
            gpu.cmd_barrier(cb, Stage::Compute, Stage::All, HazardFlags::SHADER_BUFFER);
        } else {
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::Compute,
                HazardFlags::SHADER_BUFFER,
            );
        }

        stage_end(FOG_PROFILE_NAMES[0]);

        let froxel_view = View {
            output_size: [FROXEL_WIDTH, FROXEL_HEIGHT],
            ..*view
        };

        // Build bounded local-light lists per froxel workgroup.
        // Overflow falls back to the complete light array.
        if fog_local_light_count > 0 {
            let grid_data = fa.frame_alloc(FogLightGridData {
                view: froxel_view,
                lights: light.local_lights,
                tile_counts: self.local_tile_counts.gpu,
                tile_indices: self.local_tile_indices.gpu,
                tile_overflow: self.local_tile_overflow.gpu,
                light_count: fog_local_light_count,
                tile_count: [FROXEL_LIGHT_TILES_X, FROXEL_LIGHT_TILES_Y],
                lights_per_tile: FOG_LIGHTS_PER_TILE,
            });
            gpu.cmd_set_compute_shader(cb, self.light_grid_shader);
            gpu.cmd_dispatch(cb, grid_data, fog_local_light_count.div_ceil(64), 1, 1);
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::Compute,
                HazardFlags::SHADER_BUFFER,
            );
        }
        stage_end(FOG_PROFILE_NAMES[1]);

        // Compute froxel lighting and closed-form fog self-shadow.
        let (occ_tex, occ_sampler, occ_min, occ_inv, occ_steps) = match light.occluder {
            Some(o) => (
                o.texture.index(),
                o.sampler.index(),
                o.world_min,
                o.world_inv_extent,
                dials.sun_steps,
            ),
            None => (0, 0, [0.0; 3], [0.0; 3], 0),
        };
        let light_data = fa.frame_alloc(FogLightData {
            view: froxel_view,
            params: self.curve.gpu,
            v_scatter_texture: self.v_scatter_rw.index(),
            occluder_texture: occ_tex,
            occluder_sampler: occ_sampler,
            _pad0: 0,
            occluder_world_min: occ_min,
            occluder_world_inv_extent: occ_inv,
            sun_dir: light.sun_dir,
            sun_color: light.sun_color,
            ambient_color: dials.ambient_color,
            anisotropy_g: dials.anisotropy,
            gradient_bottom: dials.gradient_bottom,
            gradient_top: dials.gradient_top,
            gradient_offset: dials.gradient_offset,
            gradient_length: dials.gradient_length,
            density: dials.density,
            height_falloff: dials.height_falloff,
            height_offset: dials.height_offset,
            sun_occlusion_steps: occ_steps,
            sun_occlusion_lod_ramp: dials.sun_lod_ramp,
            _pad1: 0,
            local_lights: light.local_lights,
            local_tile_counts: self.local_tile_counts.gpu,
            local_tile_indices: self.local_tile_indices.gpu,
            local_tile_overflow: self.local_tile_overflow.gpu,
            local_light_count: fog_local_light_count,
            local_tile_count: [FROXEL_LIGHT_TILES_X, FROXEL_LIGHT_TILES_Y],
            local_lights_per_tile: FOG_LIGHTS_PER_TILE,
        });
        gpu.cmd_set_compute_shader(cb, self.light_shader);
        gpu.cmd_dispatch(
            cb,
            light_data,
            FROXEL_WIDTH.div_ceil(8),
            FROXEL_HEIGHT.div_ceil(8),
            1,
        );
        stage_end(FOG_PROFILE_NAMES[2]);

        if particle_count > 0 {
            let splat_vert_data = fa.frame_alloc(OitSplatVertData {
                view: froxel_view,
                params: self.curve.gpu,
                particles,
            });
            let splat_frag_data = fa.frame_alloc(OitSplatFragData {
                view: froxel_view,
                params: self.curve.gpu,
                v_ext: self.v_ext.gpu,
                overflow: self.overflow.gpu,
                occupancy: self.occupancy.gpu,
                hist: self.hist.gpu,
                v_ext_rgb: self.v_ext_rgb.gpu,
                occupancy_rgb: self.occupancy_rgb.gpu,
                overflow_rgb: self.overflow_rgb.gpu,
            });
            gpu.cmd_begin_render_pass(
                cb,
                RenderPassDesc {
                    render_area_size: [FROXEL_WIDTH, FROXEL_HEIGHT],
                    color_attachments: &[RenderAttachment {
                        texture: self.splat_target.texture,
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::DontCare,
                        clear_color: [0.0; 4],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            );
            gpu.cmd_set_shaders(cb, self.oit_splat_vert_shader, self.oit_splat_frag_shader);
            gpu.cmd_set_cull_mode(cb, false);
            gpu.cmd_draw_indexed_instanced(
                cb,
                splat_vert_data.cast(),
                splat_frag_data.cast(),
                self.quad_indices.cast(),
                6,
                particle_count,
            );
            gpu.cmd_end_render_pass(cb);
            gpu.cmd_barrier(
                cb,
                Stage::FragmentShader,
                Stage::Compute,
                HazardFlags::SHADER_BUFFER,
            );
        }
        stage_end(FOG_PROFILE_NAMES[3]);

        // Order scattering writes before integration reads.
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::Compute,
            HazardFlags::SHADER_IMAGE,
        );

        let integrate_data = fa.frame_alloc(FogIntegrateData {
            view: froxel_view,
            params: self.curve.gpu,
            v_scatter_texture: self.v_scatter_slot.index(),
            v_int_texture: self.v_int_rw.index(),
            volume_depth: FROXEL_DEPTH,
            _pad0: 0,
            v_ext: self.v_ext.gpu,
            overflow: self.overflow.gpu,
            zero_slice: self.zero_slice.gpu,
            occupancy: self.occupancy.gpu,
            v_ext_rgb: self.v_ext_rgb.gpu,
            occupancy_rgb: self.occupancy_rgb.gpu,
            overflow_rgb: self.overflow_rgb.gpu,
            oit_enable: u32::from(particle_count > 0),
            tinted_enable: u32::from(tinted),
            v_tint_texture: self.v_tint_rw.index(),
            _pad: 0,
        });
        gpu.cmd_set_compute_shader(cb, self.integrate_shader);
        gpu.cmd_dispatch(
            cb,
            integrate_data,
            FROXEL_WIDTH.div_ceil(8),
            FROXEL_HEIGHT.div_ceil(8),
            1,
        );
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::Compute,
            HazardFlags::SHADER_IMAGE | HazardFlags::SHADER_BUFFER,
        );
        stage_end(FOG_PROFILE_NAMES[4]);

        let composite_data = fa.frame_alloc(FogCompositeData {
            view: *view,
            params: self.curve.gpu,
            sun_dir: light.sun_dir,
            anisotropy_g: dials.anisotropy,
            sun_color: light.sun_color,
            density: dials.density,
            ambient_color: dials.ambient_color,
            height_falloff: dials.height_falloff,
            height_offset: dials.height_offset,
            depth_texture_id: depth_slot.index(),
            v_int_texture: self.v_int_slot.index(),
            v_int_sampler: clamp_sampler.index(),
            hdr_texture: hdr_storage_slot.index(),
            fog_sample_bias: dials.fog_sample_bias,
            v_tint_texture: self.v_tint_slot.index(),
            tinted_enable: u32::from(tinted),
        });
        gpu.cmd_set_compute_shader(cb, self.composite_shader);
        gpu.cmd_dispatch(cb, composite_data, w.div_ceil(8), h.div_ceil(8), 1);
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::FragmentShader,
            HazardFlags::SHADER_IMAGE,
        );
        stage_end(FOG_PROFILE_NAMES[5]);

        if particle_count > 0 {
            let accum = self.accum.as_ref().expect("checked above");

            // Saturated columns write conservative cover depths before OIT.
            // Reverse-Z GREATER accepts only nearer values, preserving opaque
            // geometry while culling sub-threshold particle contributions.
            let spawn_data = fa.frame_alloc(FogPrimeSpawnData {
                params: self.curve.gpu,
                zero_slice: self.zero_slice.gpu,
                quads: accum.prime_quads.gpu,
                draw_args: self.prime_draw_args.gpu,
                froxel_size: [FROXEL_WIDTH, FROXEL_HEIGHT],
                screen_size: view.output_size,
                depth_near_plane: view.depth_near_plane,
                slice_margin: dials.fog_sample_bias + 1.5,
                _pad: [0; 2],
            });
            gpu.cmd_set_compute_shader(cb, self.prime_spawn_shader);
            gpu.cmd_dispatch(
                cb,
                spawn_data,
                w.div_ceil(PRIME_TILE).div_ceil(8),
                h.div_ceil(PRIME_TILE).div_ceil(8),
                1,
            );
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::All,
                HazardFlags::DRAW_ARGUMENTS | HazardFlags::SHADER_BUFFER,
            );

            let prime_vert_data = fa.frame_alloc(FogPrimeVertData {
                quads: accum.prime_quads.gpu,
                screen_size: view.output_size,
            });
            gpu.cmd_begin_render_pass(
                cb,
                RenderPassDesc {
                    render_area_size: [w, h],
                    depth_attachment: Some(RenderAttachment {
                        texture: depth_texture,
                        load_op: LoadOp::Load,
                        store_op: StoreOp::Store,
                        clear_color: [0.0; 4],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            );
            gpu.cmd_set_shaders(cb, self.prime_vert_shader, self.prime_frag_shader);
            gpu.cmd_set_depth_state(
                cb,
                DepthState {
                    mode: DepthFlags::READ | DepthFlags::WRITE,
                    compare: CompareOp::Greater,
                    ..Default::default()
                },
            );
            gpu.cmd_set_cull_mode(cb, false);
            gpu.cmd_draw_indexed_instanced_indirect(
                cb,
                prime_vert_data.cast(),
                GpuPtr::null(),
                self.quad_indices.cast(),
                self.prime_draw_args.cast(),
            );
            gpu.cmd_end_render_pass(cb);
            // Order depth writes before accumulation early-Z reads.
            gpu.cmd_barrier(
                cb,
                Stage::LateFragmentTests,
                Stage::EarlyFragmentTests,
                HazardFlags::empty(),
            );
            stage_end(FOG_PROFILE_NAMES[6]);

            let accum_vert_data = fa.frame_alloc(OitAccumVertData {
                view: *view,
                params: self.curve.gpu,
                particles,
            });
            let accum_frag_data = fa.frame_alloc(OitAccumFragData {
                view: *view,
                params: self.curve.gpu,
                v_int_texture: self.v_int_slot.index(),
                v_int_sampler: clamp_sampler.index(),
                fog_sample_bias: dials.fog_sample_bias,
                v_tint_texture: self.v_tint_slot.index(),
                tinted_enable: u32::from(tinted),
                _pad: 0,
            });
            gpu.cmd_begin_render_pass(
                cb,
                RenderPassDesc {
                    render_area_size: [w, h],
                    color_attachments: &[
                        RenderAttachment {
                            texture: accum.accum_rgb.texture,
                            load_op: LoadOp::Clear,
                            store_op: StoreOp::Store,
                            clear_color: [0.0; 4],
                            ..Default::default()
                        },
                        RenderAttachment {
                            texture: accum.accum_moments.texture,
                            load_op: LoadOp::Clear,
                            store_op: StoreOp::Store,
                            clear_color: [0.0; 4],
                            ..Default::default()
                        },
                    ],
                    depth_attachment: Some(RenderAttachment {
                        texture: depth_texture,
                        load_op: LoadOp::Load,
                        store_op: StoreOp::Store,
                        clear_color: [0.0; 4],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            );
            gpu.cmd_set_shaders(cb, self.oit_accum_vert_shader, self.oit_accum_frag_shader);
            gpu.cmd_set_depth_state(
                cb,
                DepthState {
                    mode: DepthFlags::READ,
                    compare: CompareOp::Greater,
                    ..Default::default()
                },
            );
            gpu.cmd_set_cull_mode(cb, false);
            let additive = BlendState {
                enable: true,
                color_op: BlendOp::Add,
                src_color_factor: BlendFactor::One,
                dst_color_factor: BlendFactor::One,
                alpha_op: BlendOp::Add,
                src_alpha_factor: BlendFactor::One,
                dst_alpha_factor: BlendFactor::One,
                color_write_mask: ColorComponentFlags::all().bits(),
            };
            gpu.cmd_set_blend_states(cb, &[additive, additive]);
            gpu.cmd_draw_indexed_instanced(
                cb,
                accum_vert_data.cast(),
                accum_frag_data.cast(),
                self.quad_indices.cast(),
                6,
                particle_count,
            );
            gpu.cmd_end_render_pass(cb);
            gpu.cmd_barrier(
                cb,
                Stage::RasterColorOut,
                Stage::Compute,
                HazardFlags::COLOR_ATTACHMENT,
            );
            stage_end(FOG_PROFILE_NAMES[7]);

            let resolve_data = fa.frame_alloc(OitResolveData {
                view: *view,
                params: self.curve.gpu,
                accum_rgb_texture: self.accum_rgb_slot.index(),
                accum_moments_texture: self.accum_moments_slot.index(),
                hdr_texture: hdr_storage_slot.index(),
                _pad: 0,
            });
            gpu.cmd_set_compute_shader(cb, self.oit_resolve_shader);
            gpu.cmd_dispatch(cb, resolve_data, w.div_ceil(8), h.div_ceil(8), 1);
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::FragmentShader,
                HazardFlags::SHADER_IMAGE,
            );
            stage_end(FOG_PROFILE_NAMES[8]);
        } else {
            stage_end(FOG_PROFILE_NAMES[6]);
            stage_end(FOG_PROFILE_NAMES[7]);
            stage_end(FOG_PROFILE_NAMES[8]);
        }
    }

    pub fn v_int_slot(&self) -> SampledSlot {
        self.v_int_slot
    }

    /// Read the prime-quad count after the GPU is idle.
    pub fn prime_quad_count_after_idle(&self) -> u32 {
        // SAFETY: caller guarantees GPU idleness.
        unsafe { *self.prime_draw_args.cpu.add(1) }
    }

    /// Read the latest demand histogram after GPU idleness.
    pub fn hist_after_idle(&self) -> [u32; FOG_SLICE_MAX as usize] {
        let mut out = [0u32; FOG_SLICE_MAX as usize];
        // SAFETY: caller guarantees GPU idleness.
        unsafe {
            std::ptr::copy_nonoverlapping(self.hist.cpu, out.as_mut_ptr(), FOG_SLICE_MAX as usize)
        };
        out
    }

    /// Read the complete curve after GPU idleness.
    pub fn curve_after_idle(&self) -> FogCurve {
        // SAFETY: caller guarantees GPU idleness.
        unsafe { *self.curve.cpu }
    }
}

impl Pass for VolumetricPasses {
    const NAME: &'static str = "volumetrics";

    fn free(self, gpu: &Gpu) {
        gpu.free(self.v_ext);
        gpu.free(self.overflow);
        gpu.free(self.zero_slice);
        gpu.free(self.occupancy);
        gpu.free(self.v_ext_rgb);
        gpu.free(self.occupancy_rgb);
        gpu.free(self.overflow_rgb);
        gpu.free(self.local_tile_counts);
        gpu.free(self.local_tile_indices);
        gpu.free(self.local_tile_overflow);
        gpu.free(self.prime_draw_args);
        gpu.texture_free_and_destroy(self.splat_target);
        if let Some(accum) = self.accum {
            gpu.texture_free_and_destroy(accum.accum_rgb);
            gpu.texture_free_and_destroy(accum.accum_moments);
            gpu.free(accum.prime_quads);
        }
        gpu.texture_free_and_destroy(self.v_int);
        gpu.texture_free_and_destroy(self.v_tint);
        gpu.texture_free_and_destroy(self.v_scatter);
        gpu.free(self.max_depth_bits);
        gpu.free(self.curve);
        gpu.free(self.hist);
        gpu.free(self.quad_indices);
        gpu.shader_destroy(self.depth_max_shader);
        gpu.shader_destroy(self.params_shader);
        gpu.shader_destroy(self.prime_spawn_shader);
        gpu.shader_destroy(self.light_grid_shader);
        gpu.shader_destroy(self.prime_vert_shader);
        gpu.shader_destroy(self.prime_frag_shader);
        gpu.shader_destroy(self.oit_splat_vert_shader);
        gpu.shader_destroy(self.oit_splat_frag_shader);
        gpu.shader_destroy(self.light_shader);
        gpu.shader_destroy(self.integrate_shader);
        gpu.shader_destroy(self.composite_shader);
        gpu.shader_destroy(self.oit_accum_vert_shader);
        gpu.shader_destroy(self.oit_accum_frag_shader);
        gpu.shader_destroy(self.oit_resolve_shader);
    }
}

fn v_ext_dwords() -> u32 {
    FROXEL_WIDTH * FROXEL_HEIGHT * (FROXEL_DEPTH / 4)
}

fn v_ext_bytes() -> u64 {
    v_ext_dwords() as u64 * size_of::<u32>() as u64
}

fn overflow_bytes() -> u64 {
    (FROXEL_WIDTH * FROXEL_HEIGHT) as u64 * size_of::<u32>() as u64
}
