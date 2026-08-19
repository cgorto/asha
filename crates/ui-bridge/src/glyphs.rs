//! Resolves parley glyph runs into cached Slug outlines.
//!
//! Layouts expose font bytes and glyph IDs before atlas rasterization.
//! Cache identity includes font blob, face index, and glyph ID.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use swash::FontRef;
use swash::scale::ScaleContext;
use swash::scale::outline::Outline as SwashOutline;
use swash::zeno::{self, Verb};
use text::{
    CacheError, CachedGlyph, GlyphCurve, GlyphOutline, TEXT_GLYPH_CACHE_INDEX_COUNT,
    TextGlyphBufferSlices, TextGlyphCache, TextGlyphCacheConfig, TextGlyphDescriptor,
};

/// Cache key for one glyph in one font face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FontGlyphKey {
    blob_id: u64,
    font_index: u32,
    glyph_id: u32,
}

static CUBIC_WARNED: AtomicBool = AtomicBool::new(false);
static CAPACITY_WARNED: AtomicBool = AtomicBool::new(false);
static BAD_FONT_WARNED: AtomicBool = AtomicBool::new(false);

fn warn_once(flag: &AtomicBool, msg: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        bevy::log::warn!("{msg}");
    }
}

/// Default cache capacities for typical UI text.
fn default_cache_config() -> TextGlyphCacheConfig {
    TextGlyphCacheConfig {
        max_glyphs: 4096,
        curve_capacity: 256 * 1024,
        band_capacity: 64 * 1024,
        band_curve_index_capacity: 512 * 1024,
    }
}

/// Shared outline scaler and multi-font glyph cache.
///
/// Cache keys include font identity to prevent glyph-ID collisions.
#[derive(Resource)]
pub struct GlyphOutlineProvider {
    scale_ctx: ScaleContext,
    cache: TextGlyphCache,
    lookup: HashMap<FontGlyphKey, CachedGlyph>,
}

impl Default for GlyphOutlineProvider {
    fn default() -> Self {
        Self::new(default_cache_config())
    }
}

impl GlyphOutlineProvider {
    pub fn new(config: TextGlyphCacheConfig) -> Self {
        Self {
            scale_ctx: ScaleContext::new(),
            cache: TextGlyphCache::new(config),
            lookup: HashMap::new(),
        }
    }

    /// Resolves and caches one glyph outline.
    ///
    /// Returns `None` for invalid fonts, unsupported IDs, or full caches.
    /// A cubic outline is cached as an empty outline with its advance rather
    /// than subdivided, because Slug accepts quadratic curves only.
    pub fn resolve(
        &mut self,
        font_bytes: &[u8],
        font_index: u32,
        blob_id: u64,
        glyph_id: u32,
    ) -> Option<CachedGlyph> {
        let key = FontGlyphKey {
            blob_id,
            font_index,
            glyph_id,
        };
        if let Some(record) = self.lookup.get(&key) {
            return Some(*record);
        }

        let Ok(swash_glyph_id) = u16::try_from(glyph_id) else {
            return None;
        };
        let Some(font_ref) = FontRef::from_index(font_bytes, font_index as usize) else {
            warn_once(
                &BAD_FONT_WARNED,
                "ui-bridge: swash could not parse a resolved font blob",
            );
            return None;
        };
        let units_per_em = font_ref.metrics(&[]).units_per_em;
        if units_per_em == 0 {
            return None;
        }
        let divisor = units_per_em as f32;
        let advance_em = font_ref.glyph_metrics(&[]).advance_width(swash_glyph_id) / divisor;

        // Preserve unscaled y-up em-space coordinates.
        let mut scaler = self
            .scale_ctx
            .builder(font_ref)
            .size(0.0)
            .hint(false)
            .build();
        let Some(swash_outline) = scaler.scale_outline(swash_glyph_id) else {
            // Cache missing outlines to avoid repeated work.
            return self.insert(key, GlyphOutline::empty(advance_em));
        };

        let outline = match convert_outline(&swash_outline, advance_em, divisor) {
            Some(outline) => outline,
            None => {
                warn_once(
                    &CUBIC_WARNED,
                    "ui-bridge: cubic glyph outline skipped; quadratic curves are required",
                );
                // Preserve advance and cache the empty outline so this glyph
                // is not retried on every frame.
                GlyphOutline::empty(advance_em)
            }
        };
        self.insert(key, outline)
    }

    fn insert(&mut self, key: FontGlyphKey, outline: GlyphOutline) -> Option<CachedGlyph> {
        if self.cache.records().len() >= TEXT_GLYPH_CACHE_INDEX_COUNT {
            warn_once(
                &CAPACITY_WARNED,
                "ui-bridge: glyph outline cache exhausted its u16 synthetic-id space",
            );
            return None;
        }
        let synthetic_id = self.cache.records().len() as u16;
        match self.cache.cache_outline(synthetic_id, &outline) {
            Ok(record) => {
                let record = *record;
                self.lookup.insert(key, record);
                Some(record)
            }
            Err(CacheError::Capacity(_)) => {
                warn_once(
                    &CAPACITY_WARNED,
                    "ui-bridge: glyph outline cache is full (raise GlyphOutlineProvider's \
                     TextGlyphCacheConfig)",
                );
                None
            }
            Err(CacheError::Font(_)) => {
                unreachable!("cache_outline/cache_encoded never touch a Font")
            }
        }
    }

    /// Returns cache buffers for render extraction.
    pub fn buffer_slices(&self) -> TextGlyphBufferSlices<'_> {
        self.cache.buffer_slices()
    }

    pub fn descriptors(&self) -> &[TextGlyphDescriptor] {
        self.cache.descriptors()
    }

    pub fn curves(&self) -> &[text::TextCurve] {
        self.cache.curves()
    }

    pub fn bands(&self) -> &[text::TextBandHeader] {
        self.cache.bands()
    }

    pub fn band_curve_indices(&self) -> &[u32] {
        self.cache.band_curve_indices()
    }
}

/// Converts swash outlines into quadratic em-space curves.
///
/// Lines use `p2 == p3`; cubic segments return `None`.
fn convert_outline(outline: &SwashOutline, advance_em: f32, divisor: f32) -> Option<GlyphOutline> {
    let points = outline.points();
    let verbs = outline.verbs();
    let mut curves: Vec<GlyphCurve> = Vec::new();
    let mut contour_end: Vec<usize> = Vec::new();
    let mut idx = 0usize;
    let mut current = [0f32; 2];
    let mut start = [0f32; 2];

    let conv = |p: zeno::Point| [p.x / divisor, p.y / divisor];

    for verb in verbs {
        match verb {
            Verb::MoveTo => {
                let p = conv(points[idx]);
                idx += 1;
                current = p;
                start = p;
            }
            Verb::LineTo => {
                let p = conv(points[idx]);
                idx += 1;
                curves.push(GlyphCurve {
                    p1: current,
                    p2: p,
                    p3: p,
                });
                current = p;
            }
            Verb::QuadTo => {
                let c = conv(points[idx]);
                idx += 1;
                let p = conv(points[idx]);
                idx += 1;
                curves.push(GlyphCurve {
                    p1: current,
                    p2: c,
                    p3: p,
                });
                current = p;
            }
            Verb::CurveTo => return None,
            Verb::Close => {
                if current != start {
                    curves.push(GlyphCurve {
                        p1: current,
                        p2: start,
                        p3: start,
                    });
                }
                let recorded = contour_end.last().copied().unwrap_or(0);
                if curves.len() > recorded {
                    contour_end.push(curves.len());
                }
                current = start;
            }
        }
    }

    let bounds = outline.bounds();
    let bbox_em = if curves.is_empty() {
        [0.0, 0.0, 0.0, 0.0]
    } else {
        [
            bounds.min.x / divisor,
            bounds.min.y / divisor,
            bounds.max.x / divisor,
            bounds.max.y / divisor,
        ]
    };

    Some(GlyphOutline {
        curves,
        contour_end,
        advance_em,
        bbox_em,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRA_SANS_REGULAR: &[u8] =
        include_bytes!("../../widgets/src/assets/fonts/FiraSans-Regular.ttf");

    /// Encodes representative glyphs without capacity errors.
    #[test]
    fn outline_provider_encodes_a_and_e_acute() {
        let font_ref = FontRef::from_index(FIRA_SANS_REGULAR, 0).expect("parse FiraSans-Regular");
        let mut provider = GlyphOutlineProvider::default();
        const BLOB_ID: u64 = 1;

        let mut descriptor_indices = Vec::new();
        for ch in ['A', 'é'] {
            let glyph_id = font_ref.charmap().map(ch);
            assert_ne!(glyph_id, 0, "FiraSans-Regular must contain {ch:?}");

            let record = provider
                .resolve(FIRA_SANS_REGULAR, 0, BLOB_ID, glyph_id as u32)
                .unwrap_or_else(|| panic!("outline resolution failed for {ch:?}"));
            assert!(!record.empty, "{ch:?} should have a non-empty outline");

            let descriptor = provider.descriptors()[record.descriptor_index as usize];
            let vband_max = descriptor.band_max & 0xffff;
            let hband_max = descriptor.band_max >> 16;
            assert!(
                vband_max > 0,
                "{ch:?} should span more than one vertical band"
            );
            assert!(
                hband_max > 0,
                "{ch:?} should span more than one horizontal band"
            );

            descriptor_indices.push((ch, record.descriptor_index));
        }

        // Re-resolving must reuse the cached descriptor.
        let before = provider.descriptors().len();
        let a_id = font_ref.charmap().map('A');
        let again = provider
            .resolve(FIRA_SANS_REGULAR, 0, BLOB_ID, a_id as u32)
            .expect("re-resolve 'A'");
        assert_eq!(
            provider.descriptors().len(),
            before,
            "resolve must not re-encode a cached glyph"
        );
        assert_eq!(
            again.descriptor_index, descriptor_indices[0].1,
            "cache hit must return the original descriptor index"
        );
    }
}
