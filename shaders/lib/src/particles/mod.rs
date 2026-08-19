//! Particle simulation and binned primitive rendering. Authoring is resolved
//! host-side; shaders consume fixed-layout shared records and LUTs.

use crate::core::util::atomic_add_device;
use abi_core::{GpuPtr, GraphicsPush};
use abi_particles::{
    MaterialGpu, Particle, ParticleDrawFragData, ParticleDrawPrepareData, ParticleDrawVertData,
    ParticleEmitData, ParticleResetData, ParticleUpdateData, ShapeGpu,
};
use glam::{Mat4, UVec3, Vec3, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

const LUT_SAMPLES: u32 = 256;
const PARTICLE_ACTIVE: u32 = 1;
const EMITTER_LOCAL_COORDS: u32 = 1;
const EMITTER_IGNORE_TINT: u32 = 2;
const ALIGNMENT_MESH_3D: u32 = 1;
const ALIGNMENT_Y_TO_VELOCITY: u32 = 2;
const ALIGNMENT_BILLBOARD_Y_TO_VELOCITY: u32 = 3;
const ALIGNMENT_BILLBOARD_Y: u32 = 4;

fn hash(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9e37_79b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
    x = (x ^ (x >> 13)).wrapping_mul(0xc2b2_ae35);
    x ^ (x >> 16)
}

fn rand01(seed: u32) -> f32 {
    hash(seed) as f32 * (1.0 / 4_294_967_296.0)
}
fn rand_range(seed: u32, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * rand01(seed)
}

fn safe_normalize(value: Vec3, fallback: Vec3) -> Vec3 {
    let length_squared = value.length_squared();
    if length_squared > 1.0e-8 {
        value / length_squared.sqrt()
    } else {
        fallback
    }
}

fn random_direction(seed: u32) -> Vec3 {
    let z = rand01(seed) * 2.0 - 1.0;
    let angle = rand01(seed ^ 0xa511_e9b3) * core::f32::consts::TAU;
    let radius = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(angle.cos() * radius, z, angle.sin() * radius)
}

/// Samples one of three 256-entry material LUT rows: row 0 is
/// scale/alpha/damping/angular velocity, row 1 is the sRGB-to-linear color
/// ramp, and row 2 is the HDR spawn palette.
fn sample_lut(mat: MaterialGpu, row: u32, t: f32) -> Vec4 {
    let sample = (t.clamp(0.0, 1.0) * (LUT_SAMPLES - 1) as f32 + 0.5) as u32;
    mat.curve_lut[row * LUT_SAMPLES + sample].into()
}

/// Offsets interior bolt nodes while pinning endpoints.
/// Spawn seeds produce one shared polyline per spawn.
fn bolt_node_offset(spawn_seed: u32, node: u32, segments: u32, amplitude: f32) -> Vec3 {
    if node == 0 || node >= segments {
        return Vec3::ZERO;
    }
    let base = hash(spawn_seed ^ node.wrapping_mul(0x9e37_79b9));
    Vec3::new(
        (rand01(base) - 0.5) * 2.0 * amplitude,
        0.0,
        (rand01(base ^ 0x5bd1_e995) - 0.5) * 2.0 * amplitude,
    )
}

/// Samples a shape position. `stratum` is each lane's jittered [0, 1)
/// position across the dispatch's requested spawns; only path-like shapes
/// consume it, while volume shapes remain pure hashed samples.
fn shape_position(shape: ShapeGpu, seed: u32, spawn_seed: u32, stratum: f32) -> Vec3 {
    match shape.shape_type {
        1 => random_direction(seed) * rand01(seed ^ 0x94d0_49bb).cbrt() * shape.params[0],
        2 => random_direction(seed) * shape.params[0],
        3 => {
            Vec3::new(
                rand01(seed ^ 0x12) * 2.0 - 1.0,
                rand01(seed ^ 0x34) * 2.0 - 1.0,
                rand01(seed ^ 0x56) * 2.0 - 1.0,
            ) * Vec3::new(shape.params[0], shape.params[1], shape.params[2])
        }
        // Cone direction is represented by velocity, not position.
        4 => Vec3::ZERO,
        5 => {
            let angle = rand01(seed ^ 0x78) * core::f32::consts::TAU;
            let radius =
                shape.params[0] + (shape.params[1] - shape.params[0]) * rand01(seed ^ 0x9a);
            Vec3::new(
                angle.cos() * radius,
                (rand01(seed ^ 0xbc) * 2.0 - 1.0) * shape.params[2] * 0.5,
                angle.sin() * radius,
            )
        }
        6 => {
            let segments = shape.params[0].max(1.0);
            let amplitude = shape.params[1];
            let t = stratum.clamp(0.0, 1.0);
            let scaled = t * segments;
            let node = scaled as u32;
            let frac = scaled - node as f32;
            let from = bolt_node_offset(spawn_seed, node, segments as u32, amplitude);
            let to = bolt_node_offset(spawn_seed, node + 1, segments as u32, amplitude);
            let lateral = from + (to - from) * frac;
            Vec3::new(lateral.x, t, lateral.z)
        }
        _ => Vec3::ZERO,
    }
}

/// Samples cone velocity with azimuthal flatness.
fn spawn_velocity(mat: MaterialGpu, transform: Mat4, local_coords: bool, seed: u32) -> Vec3 {
    let speed = rand_range(seed ^ 0x01, mat.speed_min, mat.speed_max);
    let mut direction = safe_normalize(Vec3::from_array(mat.direction), Vec3::X);
    let spread = mat.spread.to_radians();
    if spread > 1.0e-4 {
        let phi = rand01(seed ^ 0x02) * core::f32::consts::TAU;
        let theta = spread * rand01(seed ^ 0x03).sqrt();
        let reference = if direction.x.abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let perpendicular_1 = safe_normalize(direction.cross(reference), Vec3::Z);
        let perpendicular_2 = direction.cross(perpendicular_1);
        let flat_angle = (phi.sin() * (1.0 - mat.flatness)).atan2(phi.cos());
        direction = safe_normalize(
            direction * theta.cos()
                + (perpendicular_1 * flat_angle.cos() + perpendicular_2 * flat_angle.sin())
                    * theta.sin(),
            direction,
        );
    }
    let velocity = direction * speed;
    if local_coords {
        velocity
    } else {
        (transform * velocity.extend(0.0)).truncate()
    }
}

fn hue_rotate(color: Vec3, angle: f32) -> Vec3 {
    let axis = Vec3::splat(0.577_350_27);
    color * angle.cos()
        + axis.cross(color) * angle.sin()
        + axis * axis.dot(color) * (1.0 - angle.cos())
}

fn quat_rotate(quaternion: Vec4, vector: Vec3) -> Vec3 {
    let axis = quaternion.truncate();
    2.0 * axis.dot(vector) * axis
        + (quaternion.w * quaternion.w - axis.dot(axis)) * vector
        + 2.0 * quaternion.w * axis.cross(vector)
}

fn rotate_z(vector: Vec3, angle: f32) -> Vec3 {
    Vec3::new(
        vector.x * angle.cos() - vector.y * angle.sin(),
        vector.x * angle.sin() + vector.y * angle.cos(),
        vector.z,
    )
}

fn y_to_velocity(local: Vec3, velocity: Vec3) -> Vec3 {
    if velocity.length_squared() < 1.0e-6 {
        return local;
    }
    let y_axis = safe_normalize(velocity, Vec3::Y);
    let reference = if y_axis.z.abs() > 0.999 {
        Vec3::X
    } else {
        Vec3::Z
    };
    let x_axis = safe_normalize(y_axis.cross(reference), Vec3::X);
    let z_axis = x_axis.cross(y_axis);
    x_axis * local.x + y_axis * local.y + z_axis * local.z
}

fn billboard_y(local: Vec3, camera_forward: Vec3) -> Vec3 {
    let forward_xz = Vec3::new(camera_forward.x, 0.0, camera_forward.z);
    let forward = safe_normalize(forward_xz, Vec3::Z);
    let right = Vec3::Y.cross(forward);
    right * local.x + Vec3::Y * local.y + forward * local.z
}

fn billboard_y_to_velocity(
    local: Vec3,
    velocity: Vec3,
    camera_right: Vec3,
    camera_up: Vec3,
    camera_forward: Vec3,
) -> Vec3 {
    let projected = velocity - camera_forward * velocity.dot(camera_forward);
    let (up, right) = if projected.length_squared() > 1.0e-6 {
        let up = safe_normalize(projected, camera_up);
        (up, camera_forward.cross(up))
    } else {
        (camera_up, camera_right)
    };
    right * local.x + up * local.y + camera_forward * local.z
}

#[spirv(compute(threads(64)))]
pub fn particle_reset(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ParticleResetData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x < data.primitive_count {
        let mut draw_args = data.draw_args;
        draw_args[gid.x].instance_count = 0;
    }
    if gid.x == 0 {
        let active = data.alloc_counter[0u32].min(data.max_particles);
        let mut update_args = data.update_args.cast::<u32>();
        update_args[0u32] = active.div_ceil(256);
        update_args[1u32] = 1;
        update_args[2u32] = 1;
    }
}

/// Emits requested spawns for one emitter workgroup: group X selects the
/// stable emitter slot and each of its 256 lanes owns one requested spawn.
/// The lane's dispatch stratum is used only by path-like emission shapes.
#[spirv(compute(threads(256)))]
pub fn particle_emit(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ParticleEmitData>,
    #[spirv(workgroup_id)] group_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
) {
    let data = &**data_ptr;
    let emitter_index = group_id.x;
    if emitter_index >= data.emitter_count || local_id.x >= data.spawn_counts[emitter_index] {
        return;
    }
    let emitter = data.emitters[emitter_index];
    if local_id.x >= emitter.max_particles {
        return;
    }
    let material = data.materials[emitter.material_index];
    let allocation = atomic_add_device(data.alloc_counter, 1);
    let particle_index = allocation % data.max_particles;
    let seed = hash(emitter.spawn_seed ^ local_id.x ^ allocation.wrapping_mul(65_537));
    let transform = Mat4::from_cols_array_2d(&emitter.transform);
    let local_coords = emitter.flags & EMITTER_LOCAL_COORDS != 0;
    let stratum = (local_id.x as f32 + rand01(seed ^ 0x51ab))
        / data.spawn_counts[emitter_index].max(1) as f32;
    let local_position = shape_position(
        data.shapes[emitter.shape_index],
        seed,
        emitter.spawn_seed,
        stratum,
    ) * Vec3::from_array(emitter.emission_scale);
    let position = if local_coords {
        local_position
    } else {
        (transform * local_position.extend(1.0)).truncate()
    };
    let scale = rand_range(seed ^ 0x1234_5678, material.scale_min, material.scale_max);
    let lifetime = rand_range(
        seed ^ 0x8765_4321,
        material.lifetime_min,
        material.lifetime_max,
    );
    let palette = sample_lut(material, 2, rand01(seed ^ 0x91e1_0da5));
    let hue = rand_range(
        seed ^ 0x418c_5d31,
        material.hue_variation_min,
        material.hue_variation_max,
    );
    let mut color = hue_rotate(palette.truncate(), hue).extend(palette.w);
    if emitter.flags & EMITTER_IGNORE_TINT == 0 {
        color *= Vec4::from_array(emitter.tint);
    }
    let mut particles = data.particles;
    particles[particle_index] = Particle {
        position: position.to_array(),
        scale,
        velocity: spawn_velocity(material, transform, local_coords, seed).to_array(),
        seed,
        rotation: material.initial_rotation,
        lifetime: 0.0,
        max_lifetime: lifetime,
        emitter_index,
        material_index: emitter.material_index,
        color: color.to_array(),
        angular_velocity: rand_range(
            seed ^ 0x73ab_102d,
            material.angular_velocity_min,
            material.angular_velocity_max,
        ),
        flags: PARTICLE_ACTIVE,
        initial_scale: scale,
        _pad: 0,
    };
}

#[spirv(compute(threads(256)))]
pub fn particle_update(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ParticleUpdateData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.max_particles {
        return;
    }
    let mut particles = data.particles;
    let mut particle = particles[gid.x];
    if particle.flags & PARTICLE_ACTIVE == 0 {
        return;
    }
    particle.lifetime += data.dt;
    if particle.lifetime >= particle.max_lifetime {
        particle.flags = 0;
        particles[gid.x] = particle;
        return;
    }
    let material = data.materials[particle.material_index];
    let t = particle.lifetime / particle.max_lifetime.max(1.0e-6);
    let curves = sample_lut(material, 0, t);
    particle.scale = particle.initial_scale * curves.x;
    let velocity =
        Vec3::from_array(particle.velocity) + Vec3::from_array(material.gravity) * data.dt;
    particle.velocity = (velocity * (1.0 - material.drag * curves.z * data.dt).max(0.0)).to_array();
    particle.position = (Vec3::from_array(particle.position)
        + Vec3::from_array(particle.velocity) * data.dt)
        .to_array();
    if material.alignment == ALIGNMENT_MESH_3D && particle.angular_velocity != 0.0 {
        let half_angle = particle.angular_velocity * curves.w * data.dt * 0.5;
        let delta = Vec4::new(0.0, half_angle.sin(), 0.0, half_angle.cos());
        let quaternion = Vec4::from_array(particle.rotation);
        particle.rotation = Vec4::new(
            delta.w * quaternion.x + delta.x * quaternion.w + delta.y * quaternion.z
                - delta.z * quaternion.y,
            delta.w * quaternion.y - delta.x * quaternion.z
                + delta.y * quaternion.w
                + delta.z * quaternion.x,
            delta.w * quaternion.z + delta.x * quaternion.y - delta.y * quaternion.x
                + delta.z * quaternion.w,
            delta.w * quaternion.w
                - delta.x * quaternion.x
                - delta.y * quaternion.y
                - delta.z * quaternion.z,
        )
        .to_array();
    }
    // Fragment shading evaluates ramp and alpha.
    particles[gid.x] = particle;
}

#[spirv(compute(threads(256)))]
pub fn particle_draw_prepare(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ParticleDrawPrepareData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    let active = data.alloc_counter[0u32].min(data.max_particles);
    if gid.x >= active {
        return;
    }
    let mut particle = data.particles[gid.x];
    if particle.flags & PARTICLE_ACTIVE == 0 {
        return;
    }
    let emitter = data.emitters[particle.emitter_index];
    if emitter.flags & EMITTER_LOCAL_COORDS != 0 {
        particle.position = (Mat4::from_cols_array_2d(&emitter.transform)
            * Vec3::from_array(particle.position).extend(1.0))
        .truncate()
        .to_array();
    }
    let material = data.materials[particle.material_index];
    if material.primitive >= data.primitive_count {
        return;
    }
    let clip =
        Mat4::from_cols_array_2d(&data.view_proj) * Vec3::from_array(particle.position).extend(1.0);
    if clip.w <= 0.0 {
        return;
    }
    let ndc = clip.truncate() / clip.w;
    let radius = particle.scale
        * material.primitive_radius
        * Vec3::from_array(material.mesh_scale).abs().max_element();
    let min_dimension = data.screen_size[0].min(data.screen_size[1]) as f32;
    let margin = radius * 2.0 / min_dimension.max(1.0);
    if ndc.x < -1.0 - margin
        || ndc.x > 1.0 + margin
        || ndc.y < -1.0 - margin
        || ndc.y > 1.0 + margin
        || ndc.z < 0.0
        || ndc.z > 1.0
    {
        return;
    }
    if radius / clip.w * min_dimension * 0.5 < 0.5 {
        return;
    }
    let count_words = data.draw_args.cast::<u32>();
    let slot = atomic_add_device(count_words.offset((material.primitive * 5 + 1) as i64), 1);
    if slot >= data.max_particles {
        return;
    }
    let mut visible = data.visible;
    visible[material.primitive * data.max_particles + slot] = gid.x;
}

#[spirv(vertex)]
pub fn particle_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vertex_index: i32,
    #[spirv(instance_index)] instance_index: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_color: &mut Vec4,
    out_lifetime: &mut f32,
    out_material_index: &mut u32,
) {
    let data = push.vert::<ParticleDrawVertData>();
    let mut particle = data.particles[data.visible[instance_index]];
    let material = data.materials[particle.material_index];
    let emitter = data.emitters[particle.emitter_index];
    if emitter.flags & EMITTER_LOCAL_COORDS != 0 {
        let transform = Mat4::from_cols_array_2d(&emitter.transform);
        particle.position = (transform * Vec3::from_array(particle.position).extend(1.0))
            .truncate()
            .to_array();
        particle.velocity = (transform * Vec3::from_array(particle.velocity).extend(0.0))
            .truncate()
            .to_array();
    }
    let source_position = data.positions[vertex_index];
    let source_normal = data.normals[vertex_index];
    let mesh_scale = Vec3::from_array(material.mesh_scale) * particle.scale;
    let mut local =
        Vec3::new(source_position[0], source_position[1], source_position[2]) * mesh_scale;
    let mut normal = Vec3::new(source_normal[0], source_normal[1], source_normal[2])
        / Vec3::from_array(material.mesh_scale);
    if material.alignment != ALIGNMENT_MESH_3D {
        let seeded_angle = rand_range(
            particle.seed ^ 0x9e37_79b9,
            material.angle_min_deg,
            material.angle_max_deg,
        )
        .to_radians();
        let curves = sample_lut(
            material,
            0,
            particle.lifetime / particle.max_lifetime.max(1.0e-6),
        );
        let spin = particle.angular_velocity * curves.w * particle.lifetime;
        local = rotate_z(local, seeded_angle + spin);
        normal = rotate_z(normal, seeded_angle + spin);
    }
    let right = Vec3::from_array(data.camera_right);
    let up = Vec3::from_array(data.camera_up);
    let forward = Vec3::from_array(data.camera_forward);
    let velocity = Vec3::from_array(particle.velocity);
    let (offset, _world_normal) = match material.alignment {
        ALIGNMENT_MESH_3D => (
            quat_rotate(Vec4::from_array(particle.rotation), local),
            quat_rotate(Vec4::from_array(particle.rotation), normal),
        ),
        ALIGNMENT_Y_TO_VELOCITY => (
            y_to_velocity(local, velocity),
            y_to_velocity(normal, velocity),
        ),
        ALIGNMENT_BILLBOARD_Y_TO_VELOCITY => (
            billboard_y_to_velocity(local, velocity, right, up, forward),
            billboard_y_to_velocity(normal, velocity, right, up, forward),
        ),
        ALIGNMENT_BILLBOARD_Y => (billboard_y(local, forward), billboard_y(normal, forward)),
        _ => (
            right * local.x + up * local.y + forward * local.z,
            right * normal.x + up * normal.y + forward * normal.z,
        ),
    };
    let world = Vec3::from_array(particle.position) + offset;
    *out_pos = Mat4::from_cols_array_2d(&data.view_proj) * world.extend(1.0);
    *out_color = Vec4::from_array(particle.color);
    *out_lifetime = particle.lifetime / particle.max_lifetime.max(1.0e-6);
    *out_material_index = particle.material_index;
}

#[spirv(fragment)]
pub fn particle_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    color: Vec4,
    lifetime: f32,
    #[spirv(flat)] material_index: u32,
    out_color: &mut Vec4,
) {
    let data = push.frag::<ParticleDrawFragData>();
    let material = data.materials[material_index];
    let curves = sample_lut(material, 0, lifetime);
    let ramp = sample_lut(material, 1, lifetime);
    *out_color = Vec4::new(
        color.x * ramp.x,
        color.y * ramp.y,
        color.z * ramp.z,
        color.w * curves.y * ramp.w,
    );
}
