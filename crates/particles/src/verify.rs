//! Verify ring readback and the CPU-side ledger assertions.

use core::mem::size_of;

use abi_core::DrawIndexedIndirectCommand;
use abi_particles::Particle;
use gpu::{CommandBuffer, Gpu, HazardFlags, Stage};

use crate::sim::{MAX_EMITTERS, MAX_MATERIALS, MAX_PARTICLES, ParticleSimPass};
use crate::spec::PRIMITIVE_COUNT;

pub const VERIFY_RING: usize = 3;

/// Public verification spawn counts used by hardware tests.
pub const QUAD_SPAWNS: u32 = 24;
pub const CUBE_SPAWNS: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ParticleVerifySnapshot {
    pub allocation: u32,
    pub draw_args: [DrawIndexedIndirectCommand; PRIMITIVE_COUNT as usize],
}

const _: () = assert!(size_of::<ParticleVerifySnapshot>() == 124);

pub fn assert_verify_step(
    before: ParticleVerifySnapshot,
    after: ParticleVerifySnapshot,
    requested_spawns: u32,
) {
    assert_eq!(
        after.allocation.wrapping_sub(before.allocation),
        requested_spawns
    );
    let alive = after.allocation.min(MAX_PARTICLES);
    let visible = after
        .draw_args
        .iter()
        .map(|arg| arg.instance_count)
        .sum::<u32>();
    assert!(
        visible <= alive,
        "visible particle instances exceed fixed pool"
    );
    assert!(
        after
            .draw_args
            .iter()
            .all(|arg| arg.instance_count <= MAX_PARTICLES)
    );
}

pub fn assert_visible_indices(indices: &[u32]) {
    assert!(indices.iter().all(|&index| index < MAX_PARTICLES));
}

pub fn assert_particle_states(particles: &[Particle]) {
    for (index, particle) in particles.iter().enumerate() {
        if particle.flags & 1 == 0 {
            continue;
        }
        assert!(
            particle.position.iter().all(|value| value.is_finite())
                && particle.velocity.iter().all(|value| value.is_finite())
                && particle.rotation.iter().all(|value| value.is_finite())
                && particle.color.iter().all(|value| value.is_finite())
                && particle.scale.is_finite()
                && particle.initial_scale.is_finite()
                && particle.lifetime.is_finite()
                && particle.max_lifetime.is_finite()
                && particle.angular_velocity.is_finite(),
            "particle {index} contains non-finite state"
        );
        assert!(particle.emitter_index < MAX_EMITTERS);
        assert!(particle.material_index < MAX_MATERIALS);
    }
}

impl ParticleSimPass {
    pub(crate) fn record_verify_snapshot(&self, gpu: &Gpu, cb: CommandBuffer, slot: usize) {
        let dst = gpu.mem_suballoc(
            self.verify_ring.cast(),
            (slot * size_of::<ParticleVerifySnapshot>()) as i64,
            size_of::<ParticleVerifySnapshot>() as u64,
            1,
        );
        // Readback copies are verification-only.
        gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
        gpu.cmd_mem_copy_raw(
            cb,
            gpu.mem_suballoc(dst, 0, size_of::<u32>() as u64, 1),
            self.alloc_counter.cast(),
            size_of::<u32>() as u64,
        );
        gpu.cmd_mem_copy_raw(
            cb,
            gpu.mem_suballoc(
                dst,
                size_of::<u32>() as i64,
                size_of::<DrawIndexedIndirectCommand>() as u64,
                PRIMITIVE_COUNT as u64,
            ),
            self.draw_args.cast(),
            size_of::<DrawIndexedIndirectCommand>() as u64 * PRIMITIVE_COUNT as u64,
        );
        gpu.cmd_barrier(cb, Stage::Transfer, Stage::All, HazardFlags::empty());
    }

    pub fn verify_snapshot(&self, slot: usize) -> ParticleVerifySnapshot {
        assert!(slot < VERIFY_RING);
        unsafe { self.verify_ring.cpu.add(slot).read() }
    }
}
