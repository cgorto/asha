//! Runtime effect resolution, emitter slots, and particle compute passes.

use core::mem::{size_of, size_of_val};

use abi_core::glam::{Mat4, Quat, Vec3};
use abi_core::{DrawIndexedIndirectCommand, GpuPtr};
use abi_particles::{
    EmitterGpu, MaterialGpu, Particle, ParticleDrawPrepareData, ParticleEmitData,
    ParticleResetData, ParticleSpawn, ParticleUpdateData, ShapeGpu,
};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{CommandBuffer, Gpu, HazardFlags, Memory, Stage};

use crate::spec::{
    Curve, EffectSpec, EmitterSpec, Gradient, MAX_EFFECT_EMITTERS, PRIMITIVE_COUNT, ParticleView,
    Primitive,
};
use crate::verify::{ParticleVerifySnapshot, VERIFY_RING};

pub const MAX_PARTICLES: u32 = 65_536;
pub const MAX_EMITTERS: u32 = 128;
pub const MAX_MATERIALS: u32 = 128;
pub const MAX_SHAPES: u32 = 128;
pub const CURVE_SAMPLES: u32 = 256;
pub const CURVE_ROWS: u32 = 3;

const EMITTER_LOCAL_COORDS: u32 = 1;
const EMITTER_IGNORE_TINT: u32 = 2;
const EXPLOSIVENESS_BURST_THRESHOLD: f32 = 0.95;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EffectHandle(pub u32);

#[derive(Clone, Copy)]
struct ResolvedEmitter {
    gpu: EmitterGpu,
    delay: f32,
    one_shot: bool,
    explosiveness: f32,
    spawn_rate: f32,
    fixed_fps: u32,
    lifetime_max: f32,
    emission_curve: Curve,
    offset: [f32; 3],
}

struct RegisteredEffect {
    name: &'static str,
    emitters: Vec<ResolvedEmitter>,
}

#[derive(Clone, Copy)]
struct EmitterRuntime {
    resolved: ResolvedEmitter,
    transform: [[f32; 4]; 4],
    seed: u32,
    tint: [f32; 4],
    delay_remaining: f32,
    spawn_accumulator: f32,
    accumulated_delta: f32,
    total_spawned: u32,
    has_emitted: bool,
    spawn_time: f32,
}

impl EmitterRuntime {
    fn new(
        resolved: ResolvedEmitter,
        transform: Mat4,
        seed: u32,
        tint: [f32; 4],
        spawn_time: f32,
    ) -> Self {
        Self {
            resolved,
            transform: transform.to_cols_array_2d(),
            seed,
            tint,
            delay_remaining: resolved.delay,
            // Phase-zero emitters start immediately; delayed ones start empty.
            // Delay overshoot contributes phase; `seed` remains the GPU spawn seed.
            spawn_accumulator: if resolved.delay > 0.0 { 0.0 } else { 1.0 },
            accumulated_delta: 0.0,
            total_spawned: 0,
            has_emitted: false,
            spawn_time,
        }
    }

    fn cleanup_at(self) -> f32 {
        self.spawn_time + self.resolved.delay + self.resolved.lifetime_max
    }

    fn gpu(self) -> EmitterGpu {
        EmitterGpu {
            transform: self.transform,
            spawn_seed: self.seed,
            tint: self.tint,
            ..self.resolved.gpu
        }
    }

    fn spawn_count(&mut self, dt: f32, time: f32) -> u32 {
        if self.delay_remaining > 0.0 {
            self.delay_remaining -= dt;
            if self.delay_remaining > 0.0 {
                return 0;
            }
            let overshoot = -self.delay_remaining;
            self.delay_remaining = 0.0;
            self.spawn_accumulator += self.resolved.spawn_rate * overshoot;
        }
        if self.resolved.one_shot && self.has_emitted {
            return 0;
        }
        let mut effective_dt = dt;
        if self.resolved.fixed_fps > 0 {
            let fixed_delta = 1.0 / self.resolved.fixed_fps as f32;
            self.accumulated_delta += dt.min(0.1);
            let mut steps = 0;
            // Allow 1e-6 tolerance at fixed-step boundaries.
            while self.accumulated_delta + 1.0e-6 >= fixed_delta {
                self.accumulated_delta -= fixed_delta;
                steps += 1;
            }
            self.accumulated_delta = self.accumulated_delta.max(0.0);
            effective_dt = steps as f32 * fixed_delta;
        }
        // Explosiveness scales rate; near-one values become full bursts.
        let burst = self.resolved.explosiveness >= EXPLOSIVENESS_BURST_THRESHOLD
            || (self.resolved.one_shot && self.resolved.spawn_rate == 0.0);
        let mut rate = if burst {
            self.resolved.gpu.max_particles as f32 / effective_dt.max(1.0e-6)
        } else if self.resolved.explosiveness > 0.0 {
            self.resolved.spawn_rate / (1.0 - self.resolved.explosiveness)
        } else {
            self.resolved.spawn_rate
        };
        let life_t = ((time - self.spawn_time - self.resolved.delay) / self.resolved.lifetime_max)
            .clamp(0.0, 1.0);
        rate *= self.resolved.emission_curve.mapped(life_t).max(0.0);
        self.spawn_accumulator = (self.spawn_accumulator + rate * effective_dt)
            .min(self.resolved.gpu.max_particles as f32);
        let mut count = self.spawn_accumulator as u32;
        self.spawn_accumulator -= count as f32;
        if self.resolved.one_shot {
            let remaining = self.resolved.gpu.max_particles - self.total_spawned;
            count = count.min(remaining);
            self.total_spawned += count;
            self.has_emitted = self.total_spawned >= self.resolved.gpu.max_particles;
        }
        count
    }
}

fn release_finished_slots(
    slots: &mut [Option<EmitterRuntime>],
    free_slots: &mut Vec<u32>,
    time: f32,
) {
    for (index, runtime) in slots.iter_mut().enumerate() {
        let finished = runtime.as_ref().is_some_and(|emitter| {
            emitter.resolved.one_shot && emitter.has_emitted && time >= emitter.cleanup_at()
        });
        if finished {
            *runtime = None;
            free_slots.push(index as u32);
        }
    }
}

fn zii_curve(curve: Curve, fallback: Curve) -> Curve {
    if curve.count == 0 { fallback } else { curve }
}

fn zii_vec3(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    if value == [0.0; 3] { fallback } else { value }
}

fn zii_range(value: [f32; 2], fallback: [f32; 2]) -> [f32; 2] {
    if value == [0.0; 2] { fallback } else { value }
}

fn initial_rotation(degrees: [f32; 3]) -> [f32; 4] {
    // Match the source transform's negative Euler-ZYX convention.
    let radians = degrees.map(|value| -value.to_radians());
    (Quat::from_rotation_z(radians[2])
        * Quat::from_rotation_y(radians[1])
        * Quat::from_rotation_x(radians[0]))
    .to_array()
}

fn resolve_emitter(
    spec: EmitterSpec,
    material_index: u32,
    shape_index: u32,
) -> (ResolvedEmitter, MaterialGpu, Vec<[f32; 4]>) {
    assert!(
        spec.max_particles <= 256 || spec.max_particles == 0,
        "one emitter is one 256-lane emit workgroup"
    );
    assert!(spec.delay.is_finite() && spec.delay >= 0.0);
    assert!(spec.spawn_rate.is_finite() && spec.spawn_rate >= 0.0);
    assert!(
        spec.explosiveness.is_finite() && spec.explosiveness >= 0.0 && spec.explosiveness <= 1.0
    );
    assert!(spec.drag.is_finite() && spec.drag >= 0.0);
    assert!(spec.spread_deg.is_finite() && spec.spread_deg >= 0.0);
    assert!(spec.flatness.is_finite() && (0.0..=1.0).contains(&spec.flatness));
    for range in [
        spec.lifetime_range,
        spec.speed_range,
        spec.scale_range,
        spec.angle_range_deg,
        spec.angular_velocity_range,
        spec.hue_variation,
    ] {
        assert!(range.iter().all(|value| value.is_finite()) && range[0] <= range[1]);
    }
    assert!(spec.offset.iter().all(|value| value.is_finite()));
    assert!(spec.gravity.iter().all(|value| value.is_finite()));
    assert!(
        spec.initial_rotation_deg
            .iter()
            .all(|value| value.is_finite())
    );

    let lifetime = zii_range(spec.lifetime_range, [1.0, 1.0]);
    assert!(lifetime[0] > 0.0 && lifetime[1] > 0.0);
    let scale = zii_range(spec.scale_range, [0.5, 0.5]);
    let direction = zii_vec3(spec.direction, [0.0, 1.0, 0.0]);
    let mesh_scale = zii_vec3(spec.mesh_scale, [1.0, 1.0, 1.0]);
    let emission_scale = zii_vec3(spec.emission_scale, [1.0, 1.0, 1.0]);
    let max_particles = spec.max_particles.max(1);
    let scale_curve = zii_curve(spec.scale_curve, Curve::constant(1.0));
    let alpha_curve = zii_curve(spec.alpha_curve, Curve::constant(1.0));
    let damping_curve = zii_curve(spec.damping_curve, Curve::constant(1.0));
    let emission_curve = zii_curve(spec.emission_curve, Curve::constant(1.0));
    let material = MaterialGpu {
        direction,
        spread: spec.spread_deg,
        gravity: spec.gravity,
        drag: spec.drag,
        speed_min: spec.speed_range[0],
        speed_max: spec.speed_range[1],
        scale_min: scale[0],
        scale_max: scale[1],
        lifetime_min: lifetime[0],
        lifetime_max: lifetime[1],
        angular_velocity_min: spec.angular_velocity_range[0],
        angular_velocity_max: spec.angular_velocity_range[1],
        primitive: spec.primitive as u32,
        alignment: spec.alignment as u32,
        flags: 0,
        flatness: spec.flatness,
        mesh_scale,
        primitive_radius: spec.primitive.radius(),
        initial_rotation: initial_rotation(spec.initial_rotation_deg),
        hue_variation_min: spec.hue_variation[0],
        hue_variation_max: spec.hue_variation[1],
        angle_min_deg: spec.angle_range_deg[0],
        angle_max_deg: spec.angle_range_deg[1],
        ..Default::default()
    };
    let luts = bake_lut(
        scale_curve,
        alpha_curve,
        damping_curve,
        Gradient { ..spec.color_ramp },
        Gradient {
            ..spec.color_initial
        },
    );
    (
        ResolvedEmitter {
            gpu: EmitterGpu {
                material_index,
                shape_index,
                max_particles,
                flags: (if spec.local_coords {
                    EMITTER_LOCAL_COORDS
                } else {
                    0
                }) | (if spec.ignore_tint {
                    EMITTER_IGNORE_TINT
                } else {
                    0
                }),
                emission_scale,
                ..Default::default()
            },
            delay: spec.delay,
            one_shot: spec.one_shot,
            explosiveness: spec.explosiveness,
            spawn_rate: spec.spawn_rate,
            fixed_fps: spec.fixed_fps,
            lifetime_max: lifetime[1],
            emission_curve,
            offset: spec.offset,
        },
        material,
        luts,
    )
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Three LUT rows: dynamics, sRGB-baked ramp, and linear HDR palette.
fn bake_lut(
    scale: Curve,
    alpha: Curve,
    damping: Curve,
    ramp: Gradient,
    palette: Gradient,
) -> Vec<[f32; 4]> {
    let mut luts = vec![[0.0; 4]; (CURVE_ROWS * CURVE_SAMPLES) as usize];
    for i in 0..CURVE_SAMPLES as usize {
        let t = i as f32 / (CURVE_SAMPLES - 1) as f32;
        luts[i] = [scale.mapped(t), alpha.mapped(t), damping.mapped(t), 1.0];
        let srgb = ramp.evaluate(t);
        luts[CURVE_SAMPLES as usize + i] = [
            srgb_to_linear(srgb[0]),
            srgb_to_linear(srgb[1]),
            srgb_to_linear(srgb[2]),
            srgb[3],
        ];
        luts[(2 * CURVE_SAMPLES) as usize + i] = palette.evaluate(t);
    }
    luts
}

pub(crate) fn upload_slice<T: Copy>(
    gpu: &Gpu,
    cb: CommandBuffer,
    dst: gpu::Ptr<T>,
    values: &[T],
    staging: &mut Vec<gpu::Ptr<u8>>,
) {
    if values.is_empty() {
        return;
    }
    let src = gpu.alloc_slice::<T>(values.len() as u64, Memory::Default);
    // SAFETY: staging matches `values` and remains alive through submission.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), src.cpu, values.len()) };
    gpu.cmd_mem_copy_raw(cb, dst.cast(), src.cast(), size_of_val(values) as u64);
    staging.push(src.cast());
}

/// CPU runtime state and fixed GPU particle pools.
pub struct ParticleSimPass {
    reset_shader: gpu::Shader,
    emit_shader: gpu::Shader,
    update_shader: gpu::Shader,
    draw_prepare_shader: gpu::Shader,
    pub(crate) particles: gpu::Ptr<Particle>,
    pub(crate) alloc_counter: gpu::Ptr<u32>,
    update_args: gpu::Ptr<gpu::DispatchIndirectCommand>,
    visible: gpu::Ptr<u32>,
    pub(crate) draw_args: gpu::Ptr<DrawIndexedIndirectCommand>,
    pub(crate) emitters: gpu::Ptr<EmitterGpu>,
    spawn_counts: gpu::Ptr<u32>,
    pub(crate) materials: gpu::Ptr<MaterialGpu>,
    curve_luts: gpu::Ptr<[f32; 4]>,
    shapes: gpu::Ptr<ShapeGpu>,
    pub(crate) verify_ring: gpu::Ptr<ParticleVerifySnapshot>,
    effects: Vec<RegisteredEffect>,
    registered_materials: Vec<MaterialGpu>,
    registered_luts: Vec<[f32; 4]>,
    registered_shapes: Vec<ShapeGpu>,
    material_count: u32,
    shape_count: u32,
    effects_uploaded: bool,
    slots: Vec<Option<EmitterRuntime>>,
    free_slots: Vec<u32>,
    emitters_cpu: [EmitterGpu; MAX_EMITTERS as usize],
    emitter_high_water: u32,
    last_time: Option<f32>,
}

impl ParticleSimPass {
    pub fn new(gpu: &Gpu) -> Self {
        let particles = gpu.alloc_slice::<Particle>(MAX_PARTICLES as u64, Memory::Gpu);
        let alloc_counter = gpu.alloc::<u32>(Memory::Gpu);
        let update_args = gpu.alloc::<gpu::DispatchIndirectCommand>(Memory::Gpu);
        let visible = gpu.alloc_slice::<u32>((MAX_PARTICLES * PRIMITIVE_COUNT) as u64, Memory::Gpu);
        let draw_args =
            gpu.alloc_slice::<DrawIndexedIndirectCommand>(PRIMITIVE_COUNT as u64, Memory::Gpu);
        let emitters = gpu.alloc_slice::<EmitterGpu>(MAX_EMITTERS as u64, Memory::Default);
        let spawn_counts = gpu.alloc_slice::<u32>(MAX_EMITTERS as u64, Memory::Default);
        let materials = gpu.alloc_slice::<MaterialGpu>(MAX_MATERIALS as u64, Memory::Gpu);
        let curve_luts = gpu.alloc_slice::<[f32; 4]>(
            (MAX_MATERIALS * CURVE_ROWS * CURVE_SAMPLES) as u64,
            Memory::Gpu,
        );
        let shapes = gpu.alloc_slice::<ShapeGpu>(MAX_SHAPES as u64, Memory::Gpu);
        let verify_ring =
            gpu.alloc_slice::<ParticleVerifySnapshot>(VERIFY_RING as u64, Memory::Readback);
        let draw_constants = [
            DrawIndexedIndirectCommand {
                index_count: Primitive::Quad.index_count(),
                ..Default::default()
            },
            DrawIndexedIndirectCommand {
                index_count: Primitive::Disc.index_count(),
                ..Default::default()
            },
            DrawIndexedIndirectCommand {
                index_count: Primitive::Cube.index_count(),
                ..Default::default()
            },
            DrawIndexedIndirectCommand {
                index_count: Primitive::Icosphere.index_count(),
                ..Default::default()
            },
            DrawIndexedIndirectCommand {
                index_count: Primitive::Cone.index_count(),
                ..Default::default()
            },
            DrawIndexedIndirectCommand {
                index_count: Primitive::Prism.index_count(),
                ..Default::default()
            },
        ];
        let cb = gpu.commands_begin(gpu::Queue::Main);
        let mut staging = Vec::with_capacity(3);
        upload_slice(gpu, cb, alloc_counter, &[0], &mut staging);
        upload_slice(
            gpu,
            cb,
            update_args,
            &[gpu::DispatchIndirectCommand {
                num_groups_x: 0,
                num_groups_y: 1,
                num_groups_z: 1,
            }],
            &mut staging,
        );
        upload_slice(gpu, cb, draw_args, &draw_constants, &mut staging);
        // Transfer sources use empty hazards; shader-write is invalid there.
        gpu.cmd_barrier(cb, Stage::Transfer, Stage::All, HazardFlags::empty());
        gpu.queue_submit(gpu::Queue::Main, &[cb]);
        gpu.queue_wait_idle(gpu::Queue::Main);
        for src in staging {
            gpu.free(src);
        }
        Self {
            reset_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("particle_reset"),
                64,
                1,
                1,
                "particle_reset",
            ),
            emit_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("particle_emit"),
                256,
                1,
                1,
                "particle_emit",
            ),
            update_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("particle_update"),
                256,
                1,
                1,
                "particle_update",
            ),
            draw_prepare_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("particle_draw_prepare"),
                256,
                1,
                1,
                "particle_draw_prepare",
            ),
            particles,
            alloc_counter,
            update_args,
            visible,
            draw_args,
            emitters,
            spawn_counts,
            materials,
            curve_luts,
            shapes,
            verify_ring,
            effects: Vec::with_capacity(32),
            registered_materials: Vec::with_capacity(MAX_MATERIALS as usize),
            registered_luts: Vec::with_capacity(
                (MAX_MATERIALS * CURVE_ROWS * CURVE_SAMPLES) as usize,
            ),
            registered_shapes: Vec::with_capacity(MAX_SHAPES as usize),
            material_count: 0,
            shape_count: 0,
            effects_uploaded: false,
            slots: (0..MAX_EMITTERS).map(|_| None).collect(),
            free_slots: (0..MAX_EMITTERS).rev().collect(),
            emitters_cpu: [EmitterGpu::default(); MAX_EMITTERS as usize],
            emitter_high_water: 0,
            last_time: None,
        }
    }

    /// Registers an effect during construction and resolves its fixed pools.
    /// Complete registration before calling [`Self::effect_upload`].
    pub fn effect_register(&mut self, spec: EffectSpec) -> EffectHandle {
        assert!(
            !self.effects_uploaded,
            "effects must register before their construction upload"
        );
        assert!(spec.emitters.len() <= MAX_EFFECT_EMITTERS);
        assert!(
            (self.material_count as usize + spec.emitters.len()) <= MAX_MATERIALS as usize,
            "particle material pool exhausted"
        );
        assert!(
            (self.shape_count as usize + spec.emitters.len()) <= MAX_SHAPES as usize,
            "particle shape pool exhausted"
        );
        let handle = EffectHandle(self.effects.len() as u32);
        let mut emitters = Vec::with_capacity(spec.emitters.len());
        for authored in spec.emitters {
            let material_index = self.material_count;
            let shape_index = self.shape_count;
            let shape = authored.shape.pack();
            let (resolved, mut material, luts) =
                resolve_emitter(authored, material_index, shape_index);
            material.curve_lut = self
                .curve_luts
                .gpu
                .offset((material_index * CURVE_ROWS * CURVE_SAMPLES) as i64);
            self.registered_materials.push(material);
            self.registered_shapes.push(shape);
            self.registered_luts.extend_from_slice(&luts);
            emitters.push(resolved);
            self.material_count += 1;
            self.shape_count += 1;
        }
        self.effects.push(RegisteredEffect {
            name: spec.name,
            emitters,
        });
        handle
    }

    /// Uploads all registered effect pools exactly once.
    /// Call after registration and before [`Self::record`].
    pub fn effect_upload(&mut self, gpu: &Gpu) {
        assert!(!self.effects_uploaded, "effect pools upload exactly once");
        assert_eq!(
            self.registered_materials.len(),
            self.material_count as usize
        );
        assert_eq!(self.registered_shapes.len(), self.shape_count as usize);
        assert_eq!(
            self.registered_luts.len(),
            self.material_count as usize * CURVE_ROWS as usize * CURVE_SAMPLES as usize
        );
        let cb = gpu.commands_begin(gpu::Queue::Main);
        let mut staging = Vec::with_capacity(3);
        upload_slice(
            gpu,
            cb,
            self.materials,
            &self.registered_materials,
            &mut staging,
        );
        upload_slice(gpu, cb, self.shapes, &self.registered_shapes, &mut staging);
        upload_slice(
            gpu,
            cb,
            self.curve_luts,
            &self.registered_luts,
            &mut staging,
        );
        gpu.cmd_barrier(cb, Stage::Transfer, Stage::All, HazardFlags::empty());
        gpu.queue_submit(gpu::Queue::Main, &[cb]);
        gpu.queue_wait_idle(gpu::Queue::Main);
        for src in staging {
            gpu.free(src);
        }
        self.effects_uploaded = true;
    }

    pub fn effect_name(&self, handle: EffectHandle) -> &'static str {
        self.effects
            .get(handle.0 as usize)
            .expect("invalid particle effect handle")
            .name
    }

    fn dt(&mut self, time: f32) -> f32 {
        assert!(time.is_finite() && time >= 0.0);
        let dt = self
            .last_time
            .map_or(1.0 / 60.0, |last| (time - last).max(0.0));
        self.last_time = Some(time);
        dt.min(0.1)
    }

    fn release_finished_slots(&mut self, time: f32) {
        release_finished_slots(&mut self.slots, &mut self.free_slots, time);
    }

    fn spawn(&mut self, spawns: &[ParticleSpawn], time: f32) {
        self.release_finished_slots(time);
        for spawn in spawns {
            let effect_index = spawn.effect_id as usize;
            let emitter_count = self
                .effects
                .get(effect_index)
                .expect("particle spawn named an unregistered effect")
                .emitters
                .len();
            assert!(
                self.free_slots.len() >= emitter_count,
                "particle emitter slot pool exhausted"
            );
            for emitter_index in 0..emitter_count {
                let resolved = self.effects[effect_index].emitters[emitter_index];
                let slot = self.free_slots.pop().expect("slot capacity checked");
                let transform = Mat4::from_cols_array_2d(&spawn.transform)
                    * Mat4::from_translation(Vec3::from_array(resolved.offset));
                self.slots[slot as usize] = Some(EmitterRuntime::new(
                    resolved,
                    transform,
                    spawn.seed ^ slot.wrapping_mul(0x9e37_79b9),
                    spawn.tint,
                    time,
                ));
                self.emitter_high_water = self.emitter_high_water.max(slot + 1);
            }
        }
    }

    fn upload_emitters(&mut self, dt: f32, time: f32) -> (u32, u32) {
        let mut requested = 0;
        for index in 0..self.emitter_high_water as usize {
            if let Some(runtime) = self.slots[index].as_mut() {
                let count = runtime.spawn_count(dt, time);
                self.emitters_cpu[index] = runtime.gpu();
                requested += count;
                // SAFETY: mapped high-water slots are exclusively owned this frame.
                unsafe { self.spawn_counts.cpu.add(index).write(count) };
            } else {
                unsafe { self.spawn_counts.cpu.add(index).write(0) };
            }
            unsafe { self.emitters.cpu.add(index).write(self.emitters_cpu[index]) };
        }
        (self.emitter_high_water, requested)
    }

    /// Records reset, emission, update, and binned-draw preparation.
    /// Returns the CPU-requested spawn count used by verification.
    pub fn record(
        &mut self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        spawns: &[ParticleSpawn],
        time: f32,
        view: ParticleView,
        verify_slot: Option<usize>,
    ) -> u32 {
        assert!(
            self.effects_uploaded,
            "effect pools must upload before particle recording"
        );
        assert!(view.screen_size.x > 0 && view.screen_size.y > 0);
        let dt = self.dt(time);
        self.spawn(spawns, time);
        let (emitter_count, requested_spawns) = self.upload_emitters(dt, time);
        gpu.cmd_barrier(cb, Stage::Transfer, Stage::Compute, HazardFlags::empty());
        let reset = fa.frame_alloc(ParticleResetData {
            alloc_counter: self.alloc_counter.gpu,
            update_args: self.update_args.gpu.cast(),
            draw_args: self.draw_args.gpu,
            max_particles: MAX_PARTICLES,
            primitive_count: PRIMITIVE_COUNT,
        });
        gpu.cmd_set_compute_shader(cb, self.reset_shader);
        gpu.cmd_dispatch(cb, reset, 1, 1, 1);
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::Compute,
            HazardFlags::SHADER_BUFFER,
        );
        let emit = fa.frame_alloc(ParticleEmitData {
            particles: self.particles.gpu,
            alloc_counter: self.alloc_counter.gpu,
            emitters: self.emitters.gpu,
            spawn_counts: self.spawn_counts.gpu,
            materials: self.materials.gpu,
            shapes: self.shapes.gpu,
            emitter_count,
            max_particles: MAX_PARTICLES,
            ..Default::default()
        });
        if emitter_count != 0 {
            gpu.cmd_set_compute_shader(cb, self.emit_shader);
            // Dispatch one workgroup per high-water emitter slot.
            gpu.cmd_dispatch(cb, emit, emitter_count, 1, 1);
        }
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::All,
            HazardFlags::DRAW_ARGUMENTS | HazardFlags::SHADER_BUFFER,
        );
        let update = fa.frame_alloc(ParticleUpdateData {
            particles: self.particles.gpu,
            emitters: self.emitters.gpu,
            materials: self.materials.gpu,
            dt,
            max_particles: MAX_PARTICLES,
            ..Default::default()
        });
        gpu.cmd_set_compute_shader(cb, self.update_shader);
        gpu.cmd_dispatch_indirect(cb, update, self.update_args);
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::Compute,
            HazardFlags::SHADER_BUFFER,
        );
        let draw_prepare = fa.frame_alloc(ParticleDrawPrepareData {
            particles: self.particles.gpu,
            alloc_counter: self.alloc_counter.gpu,
            emitters: self.emitters.gpu,
            materials: self.materials.gpu,
            visible: self.visible.gpu,
            draw_args: self.draw_args.gpu,
            view_proj: view.view_proj.to_cols_array_2d(),
            screen_size: view.screen_size.to_array(),
            max_particles: MAX_PARTICLES,
            primitive_count: PRIMITIVE_COUNT,
        });
        gpu.cmd_set_compute_shader(cb, self.draw_prepare_shader);
        gpu.cmd_dispatch(cb, draw_prepare, MAX_PARTICLES.div_ceil(256), 1, 1);
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::All,
            HazardFlags::DRAW_ARGUMENTS | HazardFlags::SHADER_BUFFER,
        );
        if let Some(slot) = verify_slot {
            assert!(slot < VERIFY_RING);
            self.record_verify_snapshot(gpu, cb, slot);
        }
        requested_spawns
    }

    pub(crate) fn visible_ptr(&self, primitive: u32) -> GpuPtr<u32> {
        assert!(primitive < PRIMITIVE_COUNT);
        self.visible.gpu.offset((primitive * MAX_PARTICLES) as i64)
    }

    pub(crate) fn draw_args_ptr(&self, gpu: &Gpu, primitive: u32) -> gpu::Ptr<u8> {
        assert!(primitive < PRIMITIVE_COUNT);
        gpu.mem_suballoc(
            self.draw_args.cast(),
            (primitive as usize * size_of::<DrawIndexedIndirectCommand>()) as i64,
            size_of::<DrawIndexedIndirectCommand>() as u64,
            1,
        )
    }
}

impl Pass for ParticleSimPass {
    const NAME: &'static str = "particles_sim";
    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.reset_shader);
        gpu.shader_destroy(self.emit_shader);
        gpu.shader_destroy(self.update_shader);
        gpu.shader_destroy(self.draw_prepare_shader);
        gpu.free(self.particles);
        gpu.free(self.alloc_counter);
        gpu.free(self.update_args);
        gpu.free(self.visible);
        gpu.free(self.draw_args);
        gpu.free(self.emitters);
        gpu.free(self.spawn_counts);
        gpu.free(self.materials);
        gpu.free(self.curve_luts);
        gpu.free(self.shapes);
        gpu.free(self.verify_ring);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zii_default_resolves_visible_particle() {
        let (resolved, material, _) = resolve_emitter(EmitterSpec::default(), 0, 0);
        assert_eq!(resolved.gpu.max_particles, 1);
        assert_eq!(material.direction, [0.0, 1.0, 0.0]);
        assert_eq!(material.scale_min, 0.5);
        assert_eq!(material.lifetime_max, 1.0);
    }

    #[test]
    fn lut_bakes_srgb_and_hdr_palette() {
        let lut = bake_lut(
            Curve::constant(1.0),
            Curve::constant(1.0),
            Curve::constant(1.0),
            Gradient::two_stop([0.5, 0.5, 0.5, 1.0], [0.5, 0.5, 0.5, 1.0]),
            Gradient::two_stop([3.0, 2.0, 1.0, 1.0], [3.0, 2.0, 1.0, 1.0]),
        );
        assert!((lut[CURVE_SAMPLES as usize][0] - srgb_to_linear(0.5)).abs() < 0.000_001);
        assert_eq!(lut[(2 * CURVE_SAMPLES) as usize], [3.0, 2.0, 1.0, 1.0]);
    }

    #[test]
    fn resolver_rejects_one_workgroup_overflow() {
        let result = std::panic::catch_unwind(|| {
            resolve_emitter(
                EmitterSpec {
                    max_particles: 257,
                    ..Default::default()
                },
                0,
                0,
            )
        });
        assert!(result.is_err());
    }

    #[test]
    fn spawn_accumulator_preserves_phase_fixed_fps_and_budget() {
        let (resolved, _, _) = resolve_emitter(
            EmitterSpec {
                one_shot: true,
                spawn_rate: 10.0,
                max_particles: 3,
                explosiveness: 0.0,
                fixed_fps: 10,
                ..Default::default()
            },
            0,
            0,
        );
        let mut emitter = EmitterRuntime::new(resolved, Mat4::IDENTITY, 1, [1.0; 4], 0.0);
        assert_eq!(emitter.spawn_count(0.05, 0.05), 1);
        assert_eq!(emitter.spawn_count(0.04, 0.09), 0);
        assert_eq!(emitter.spawn_count(0.01, 0.10), 1);
        assert_eq!(emitter.spawn_count(0.1, 0.20), 1);
        assert_eq!(emitter.spawn_count(0.1, 0.30), 0);
        assert_eq!(emitter.total_spawned, 3);
    }

    #[test]
    fn burst_delay_and_slot_reuse_are_stable() {
        let (resolved, _, _) = resolve_emitter(
            EmitterSpec {
                one_shot: true,
                explosiveness: 1.0,
                max_particles: 2,
                delay: 0.1,
                lifetime_range: [0.2, 0.2],
                ..Default::default()
            },
            0,
            0,
        );
        let mut emitter = EmitterRuntime::new(resolved, Mat4::IDENTITY, 1, [1.0; 4], 0.0);
        assert_eq!(emitter.spawn_count(0.05, 0.05), 0);
        assert_eq!(emitter.spawn_count(0.06, 0.11), 2);
        assert!(emitter.has_emitted && emitter.cleanup_at() == 0.3);
        let mut slots = vec![Some(emitter), None];
        let mut free = vec![1];
        release_finished_slots(&mut slots, &mut free, 0.3);
        assert!(slots[0].is_none());
        assert_eq!(free.pop(), Some(0));
    }
}
