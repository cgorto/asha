//! Shader entry points that marshal ABI data and call domain math.

#![cfg_attr(target_arch = "spirv", no_std)]
#![deny(warnings)]

mod core;
mod debug;
mod lighting;
mod mesh;
mod particles;
mod post;
mod ui;

// `self::` disambiguates the local module from the extern-prelude crate.
pub use self::core::*;

pub use debug::*;
pub use lighting::*;
pub use mesh::*;
pub use particles::*;
pub use post::*;
pub use ui::*;
