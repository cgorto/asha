//! Helpers shared across entry-point modules.

use abi_core::GpuPtr;
use spirv_std::memory::{Scope, Semantics};

/// Device-scope relaxed atomic add on a BDA counter.
pub(crate) fn atomic_add_device(counter: GpuPtr<u32>, value: u32) -> u32 {
    unsafe {
        spirv_std::arch::atomic_i_add::<u32, { Scope::Device as u32 }, { Semantics::NONE.bits() }>(
            &mut *counter.as_ptr(),
            value,
        )
    }
}

/// Device-scope relaxed atomic max on a BDA counter.
pub(crate) fn atomic_max_device(counter: GpuPtr<u32>, value: u32) -> u32 {
    unsafe {
        spirv_std::arch::atomic_u_max::<u32, { Scope::Device as u32 }, { Semantics::NONE.bits() }>(
            &mut *counter.as_ptr(),
            value,
        )
    }
}

/// Device-scope relaxed atomic or on a BDA bit set.
pub(crate) fn atomic_or_device(counter: GpuPtr<u32>, value: u32) -> u32 {
    unsafe {
        spirv_std::arch::atomic_or::<u32, { Scope::Device as u32 }, { Semantics::NONE.bits() }>(
            &mut *counter.as_ptr(),
            value,
        )
    }
}

/// Device-scope relaxed atomic min on a BDA counter.
pub(crate) fn atomic_min_device(counter: GpuPtr<u32>, value: u32) -> u32 {
    unsafe {
        spirv_std::arch::atomic_u_min::<u32, { Scope::Device as u32 }, { Semantics::NONE.bits() }>(
            &mut *counter.as_ptr(),
            value,
        )
    }
}

pub(crate) fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
