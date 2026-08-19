//! Host-only Slug text preparation for the `abi_ui` GPU ABI.
//!
//! Parses supported TrueType outlines and builds quadratic curve bands.

#![cfg_attr(target_arch = "spirv", no_std)]
#![cfg_attr(target_arch = "spirv", allow(unused_imports))]

#[cfg(target_arch = "spirv")]
compile_error!("crates/text is a host-only CPU text preparation crate");

use std::mem::size_of;

mod encode;
mod font;
pub mod pass;

pub use abi_ui::{
    TextBandHeader, TextCamera, TextCurve, TextDraw, TextGlyphDescriptor, TextGlyphInstance,
};
pub use encode::{
    CacheError, CachedGlyph, CapacityKind, EncodedBand, EncodedGlyph, GlyphCurve, GlyphOutline,
    SLUG_BAND_EPSILON, SLUG_MAX_BANDS, TEXT_GLYPH_CACHE_INDEX_COUNT, TextGlyphBufferSlices,
    TextGlyphCache, TextGlyphCacheConfig, encode_glyph, pack_band_max, select_band,
};
pub use font::{Font, FontError};
pub use pass::{TextBatch, TextBatchDesc, TextPass, TextPassTarget, TextScissor};

const _: () = assert!(size_of::<TextGlyphDescriptor>() == 48);
const _: () = assert!(size_of::<TextGlyphInstance>() == 16);
const _: () = assert!(size_of::<TextBandHeader>() == 8);
