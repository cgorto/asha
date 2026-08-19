//! Fixed particle ABI shared by the particles host passes and rust-gpu entry points.
//!
//! This module deliberately contains no authoring vocabulary. Specs, curve
//! baking, and resolver policy are host-only in `particles`.

use abi_core::DrawIndexedIndirectCommand;

use crate::{GpuPtr, gpu_data};

/// A live particle in the fixed ring pool.
#[gpu_data]
pub struct Particle {
    pub position: [f32; 3],
    pub scale: f32,
    pub velocity: [f32; 3],
    pub seed: u32,
    pub rotation: [f32; 4],
    pub lifetime: f32,
    pub max_lifetime: f32,
    pub emitter_index: u32,
    pub material_index: u32,
    pub color: [f32; 4],
    pub angular_velocity: f32,
    pub flags: u32,
    pub initial_scale: f32,
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<Particle>() == 96);
const _: () = assert!(core::mem::align_of::<Particle>() == 4);
const _: () = assert!(core::mem::offset_of!(Particle, position) == 0);
const _: () = assert!(core::mem::offset_of!(Particle, rotation) == 32);
const _: () = assert!(core::mem::offset_of!(Particle, lifetime) == 48);
const _: () = assert!(core::mem::offset_of!(Particle, emitter_index) == 56);
const _: () = assert!(core::mem::offset_of!(Particle, material_index) == 60);
const _: () = assert!(core::mem::offset_of!(Particle, color) == 64);
const _: () = assert!(core::mem::offset_of!(Particle, angular_velocity) == 80);

/// One extracted particle-effect request with column-major transforms.
#[gpu_data(component)]
pub struct ParticleSpawn {
    pub transform: [[f32; 4]; 4],
    pub effect_id: u32,
    pub seed: u32,
    pub _pad: [u32; 2],
    /// Per-spawn modulation multiplied into emitter colors unless disabled.
    pub tint: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<ParticleSpawn>() == 96);
const _: () = assert!(core::mem::align_of::<ParticleSpawn>() == 4);
const _: () = assert!(core::mem::offset_of!(ParticleSpawn, transform) == 0);
const _: () = assert!(core::mem::offset_of!(ParticleSpawn, effect_id) == 64);
const _: () = assert!(core::mem::offset_of!(ParticleSpawn, tint) == 80);

/// Per-emitter state written by the render thread.
#[gpu_data]
pub struct EmitterGpu {
    pub transform: [[f32; 4]; 4],
    pub material_index: u32,
    pub shape_index: u32,
    pub max_particles: u32,
    pub spawn_seed: u32,
    /// Bit 0 enables local coordinates; bit 1 disables tinting.
    pub flags: u32,
    pub emission_scale: [f32; 3],
    /// The owning spawn's tint, copied to every emitter slot of the effect.
    pub tint: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<EmitterGpu>() == 112);
const _: () = assert!(core::mem::align_of::<EmitterGpu>() == 4);
const _: () = assert!(core::mem::offset_of!(EmitterGpu, material_index) == 64);
const _: () = assert!(core::mem::offset_of!(EmitterGpu, shape_index) == 68);
const _: () = assert!(core::mem::offset_of!(EmitterGpu, max_particles) == 72);
const _: () = assert!(core::mem::offset_of!(EmitterGpu, spawn_seed) == 76);
const _: () = assert!(core::mem::offset_of!(EmitterGpu, flags) == 80);
const _: () = assert!(core::mem::offset_of!(EmitterGpu, emission_scale) == 84);
const _: () = assert!(core::mem::offset_of!(EmitterGpu, tint) == 96);

/// Flat emission shape record. The parameter lanes are deliberately generic:
/// their interpretation is selected by `shape_type` in the emit shader.
#[gpu_data]
pub struct ShapeGpu {
    pub shape_type: u32,
    pub _pad0: u32,
    pub params: [f32; 10],
}

const _: () = assert!(core::mem::size_of::<ShapeGpu>() == 48);
const _: () = assert!(core::mem::align_of::<ShapeGpu>() == 4);
const _: () = assert!(core::mem::offset_of!(ShapeGpu, shape_type) == 0);
const _: () = assert!(core::mem::offset_of!(ShapeGpu, params) == 8);

/// Fixed material record. Its LUT address is deliberately two `u32`s through
/// [`GpuPtr`]; never replace it with an `u64` in a GPU-facing layout.
#[gpu_data]
pub struct MaterialGpu {
    pub direction: [f32; 3],
    pub spread: f32,
    pub gravity: [f32; 3],
    pub drag: f32,
    pub speed_min: f32,
    pub speed_max: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    pub lifetime_min: f32,
    pub lifetime_max: f32,
    pub angular_velocity_min: f32,
    pub angular_velocity_max: f32,
    pub primitive: u32,
    pub alignment: u32,
    pub flags: u32,
    pub flatness: f32,
    pub mesh_scale: [f32; 3],
    pub primitive_radius: f32,
    pub curve_lut: GpuPtr<[f32; 4]>,
    pub initial_rotation: [f32; 4],
    pub hue_variation_min: f32,
    pub hue_variation_max: f32,
    pub angle_min_deg: f32,
    pub angle_max_deg: f32,
    pub _pad: [u32; 30],
}

const _: () = assert!(core::mem::size_of::<MaterialGpu>() == 256);
const _: () = assert!(core::mem::align_of::<MaterialGpu>() == 4);
const _: () = assert!(core::mem::offset_of!(MaterialGpu, primitive) == 64);
const _: () = assert!(core::mem::offset_of!(MaterialGpu, alignment) == 68);
const _: () = assert!(core::mem::offset_of!(MaterialGpu, flatness) == 76);
const _: () = assert!(core::mem::offset_of!(MaterialGpu, mesh_scale) == 80);
const _: () = assert!(core::mem::offset_of!(MaterialGpu, curve_lut) == 96);
const _: () = assert!(core::mem::offset_of!(MaterialGpu, initial_rotation) == 104);
const _: () = assert!(core::mem::offset_of!(MaterialGpu, hue_variation_min) == 120);
const _: () = assert!(core::mem::offset_of!(MaterialGpu, angle_min_deg) == 128);

/// Reset-pass data for indirect counts and update dispatch.
#[gpu_data]
pub struct ParticleResetData {
    pub alloc_counter: GpuPtr<u32>,
    pub update_args: GpuPtr<[u32; 3]>,
    pub draw_args: GpuPtr<DrawIndexedIndirectCommand>,
    pub max_particles: u32,
    pub primitive_count: u32,
}

const _: () = assert!(core::mem::size_of::<ParticleResetData>() == 32);
const _: () = assert!(core::mem::offset_of!(ParticleResetData, alloc_counter) == 0);
const _: () = assert!(core::mem::offset_of!(ParticleResetData, draw_args) == 16);
const _: () = assert!(core::mem::offset_of!(ParticleResetData, max_particles) == 24);

/// Emit dispatch data; `max_particles` is the ring modulus.
#[gpu_data]
pub struct ParticleEmitData {
    pub particles: GpuPtr<Particle>,
    pub alloc_counter: GpuPtr<u32>,
    pub emitters: GpuPtr<EmitterGpu>,
    pub spawn_counts: GpuPtr<u32>,
    pub materials: GpuPtr<MaterialGpu>,
    pub shapes: GpuPtr<ShapeGpu>,
    pub emitter_count: u32,
    pub max_particles: u32,
    pub _pad: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<ParticleEmitData>() == 64);
const _: () = assert!(core::mem::offset_of!(ParticleEmitData, particles) == 0);
const _: () = assert!(core::mem::offset_of!(ParticleEmitData, emitters) == 16);
const _: () = assert!(core::mem::offset_of!(ParticleEmitData, materials) == 32);
const _: () = assert!(core::mem::offset_of!(ParticleEmitData, shapes) == 40);
const _: () = assert!(core::mem::offset_of!(ParticleEmitData, emitter_count) == 48);

/// Update dispatch data.
#[gpu_data]
pub struct ParticleUpdateData {
    pub particles: GpuPtr<Particle>,
    pub emitters: GpuPtr<EmitterGpu>,
    pub materials: GpuPtr<MaterialGpu>,
    pub dt: f32,
    pub max_particles: u32,
    pub _pad: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<ParticleUpdateData>() == 40);
const _: () = assert!(core::mem::offset_of!(ParticleUpdateData, particles) == 0);
const _: () = assert!(core::mem::offset_of!(ParticleUpdateData, dt) == 24);
const _: () = assert!(core::mem::offset_of!(ParticleUpdateData, max_particles) == 28);

/// Draw-preparation data with six contiguous visibility bins.
#[gpu_data]
pub struct ParticleDrawPrepareData {
    pub particles: GpuPtr<Particle>,
    pub alloc_counter: GpuPtr<u32>,
    pub emitters: GpuPtr<EmitterGpu>,
    pub materials: GpuPtr<MaterialGpu>,
    pub visible: GpuPtr<u32>,
    pub draw_args: GpuPtr<DrawIndexedIndirectCommand>,
    pub view_proj: [[f32; 4]; 4],
    pub screen_size: [u32; 2],
    pub max_particles: u32,
    pub primitive_count: u32,
}

const _: () = assert!(core::mem::size_of::<ParticleDrawPrepareData>() == 128);
const _: () = assert!(core::mem::offset_of!(ParticleDrawPrepareData, particles) == 0);
const _: () = assert!(core::mem::offset_of!(ParticleDrawPrepareData, visible) == 32);
const _: () = assert!(core::mem::offset_of!(ParticleDrawPrepareData, view_proj) == 48);
const _: () = assert!(core::mem::offset_of!(ParticleDrawPrepareData, max_particles) == 120);

/// Per-primitive vertex pull data. The host writes a different primitive and
/// mesh-pointer tuple for each indirect draw.
#[gpu_data]
pub struct ParticleDrawVertData {
    pub particles: GpuPtr<Particle>,
    pub visible: GpuPtr<u32>,
    pub emitters: GpuPtr<EmitterGpu>,
    pub materials: GpuPtr<MaterialGpu>,
    pub positions: GpuPtr<[f32; 4]>,
    pub normals: GpuPtr<[f32; 4]>,
    pub uvs: GpuPtr<[f32; 2]>,
    pub view_proj: [[f32; 4]; 4],
    pub camera_right: [f32; 3],
    pub _pad0: u32,
    pub camera_up: [f32; 3],
    pub _pad1: u32,
    pub camera_forward: [f32; 3],
    pub _pad_forward: u32,
    pub primitive: u32,
    pub _pad2: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<ParticleDrawVertData>() == 184);
const _: () = assert!(core::mem::offset_of!(ParticleDrawVertData, particles) == 0);
const _: () = assert!(core::mem::offset_of!(ParticleDrawVertData, positions) == 32);
const _: () = assert!(core::mem::offset_of!(ParticleDrawVertData, view_proj) == 56);
const _: () = assert!(core::mem::offset_of!(ParticleDrawVertData, camera_forward) == 152);
const _: () = assert!(core::mem::offset_of!(ParticleDrawVertData, primitive) == 168);

/// Fragment data for particle materials and interpolants.
#[gpu_data]
pub struct ParticleDrawFragData {
    pub materials: GpuPtr<MaterialGpu>,
}

const _: () = assert!(core::mem::size_of::<ParticleDrawFragData>() == 8);
const _: () = assert!(core::mem::offset_of!(ParticleDrawFragData, materials) == 0);
