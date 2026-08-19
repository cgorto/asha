//! Slug encoding and CPU glyph caching.
//!
//! Converts outlines into ABI-compatible quadratic curve bands.

use crate::font::{Font, FontError};
use abi_ui::{TextBandHeader, TextCurve, TextGlyphDescriptor};

pub const SLUG_MAX_BANDS: u32 = 16;
pub const SLUG_BAND_EPSILON: f32 = 1.0 / 1024.0;
pub const TEXT_GLYPH_CACHE_INDEX_COUNT: usize = 1 << 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphCurve {
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlyphOutline {
    pub curves: Vec<GlyphCurve>,
    pub contour_end: Vec<usize>,
    pub advance_em: f32,
    pub bbox_em: [f32; 4],
}

impl GlyphOutline {
    pub fn empty(advance_em: f32) -> Self {
        Self {
            curves: Vec::new(),
            contour_end: Vec::new(),
            advance_em,
            bbox_em: [0.0; 4],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedBand {
    pub curve_indices: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct EncodedGlyph {
    pub curves: Vec<TextCurve>,
    pub hbands: Vec<EncodedBand>,
    pub vbands: Vec<EncodedBand>,
    pub band_scale: [f32; 2],
    pub band_offset: [f32; 2],
    pub band_max: [u16; 2],
    pub even_odd: bool,
    pub bbox_em: [f32; 4],
    pub advance_em: f32,
}

pub fn encode_glyph(outline: &GlyphOutline) -> EncodedGlyph {
    assert_outline(outline);

    let curve_count = outline.curves.len() as u32;
    let num_hbands = choose_band_count(curve_count);
    let num_vbands = choose_band_count(curve_count);

    let mut glyph = EncodedGlyph {
        curves: outline
            .curves
            .iter()
            .map(|curve| TextCurve {
                p1: curve.p1,
                p2: curve.p2,
                p3: curve.p3,
            })
            .collect(),
        hbands: vec![
            EncodedBand {
                curve_indices: Vec::new()
            };
            num_hbands as usize
        ],
        vbands: vec![
            EncodedBand {
                curve_indices: Vec::new()
            };
            num_vbands as usize
        ],
        band_scale: [
            axis_scale(num_vbands, outline.bbox_em[0], outline.bbox_em[2]),
            axis_scale(num_hbands, outline.bbox_em[1], outline.bbox_em[3]),
        ],
        band_offset: [0.0; 2],
        band_max: [(num_vbands - 1) as u16, (num_hbands - 1) as u16],
        even_odd: false,
        bbox_em: outline.bbox_em,
        advance_em: outline.advance_em,
    };
    glyph.band_offset = [
        -outline.bbox_em[0] * glyph.band_scale[0],
        -outline.bbox_em[1] * glyph.band_scale[1],
    ];

    assign_horizontal_bands(&mut glyph, outline);
    assign_vertical_bands(&mut glyph, outline);
    glyph
}

pub fn select_band(value_em: f32, band_scale: f32, band_offset: f32, max_band: u16) -> u16 {
    clamped_band(value_em * band_scale + band_offset, max_band)
}

pub fn pack_band_max(vband_max: u16, hband_max: u16) -> u32 {
    u32::from(vband_max) | (u32::from(hband_max) << 16)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityKind {
    Glyphs,
    Descriptors,
    Curves,
    Bands,
    BandCurveIndices,
    U32Index,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheError {
    Font(FontError),
    Capacity(CapacityKind),
}

impl From<FontError> for CacheError {
    fn from(value: FontError) -> Self {
        Self::Font(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextGlyphCacheConfig {
    pub max_glyphs: usize,
    pub curve_capacity: usize,
    pub band_capacity: usize,
    pub band_curve_index_capacity: usize,
}

impl Default for TextGlyphCacheConfig {
    fn default() -> Self {
        Self {
            max_glyphs: 128,
            curve_capacity: 64 * 1024,
            band_capacity: 4 * 1024,
            band_curve_index_capacity: 64 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CachedGlyph {
    pub glyph_id: u16,
    pub descriptor_index: u32,
    pub advance_em: f32,
    pub empty: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct TextGlyphBufferSlices<'a> {
    pub descriptors: &'a [TextGlyphDescriptor],
    pub curves: &'a [TextCurve],
    pub bands: &'a [TextBandHeader],
    pub band_curve_indices: &'a [u32],
}

#[derive(Debug, Clone)]
pub struct TextGlyphCache {
    lookup: Vec<u32>,
    records: Vec<CachedGlyph>,
    descriptors: Vec<TextGlyphDescriptor>,
    curves: Vec<TextCurve>,
    bands: Vec<TextBandHeader>,
    band_curve_indices: Vec<u32>,
    config: TextGlyphCacheConfig,
}

impl TextGlyphCache {
    pub fn new(config: TextGlyphCacheConfig) -> Self {
        assert!(config.max_glyphs > 0);
        assert!(config.max_glyphs <= TEXT_GLYPH_CACHE_INDEX_COUNT);
        Self {
            lookup: vec![0; TEXT_GLYPH_CACHE_INDEX_COUNT],
            records: Vec::with_capacity(config.max_glyphs),
            descriptors: Vec::with_capacity(config.max_glyphs),
            curves: Vec::with_capacity(config.curve_capacity),
            bands: Vec::with_capacity(config.band_capacity),
            band_curve_indices: Vec::with_capacity(config.band_curve_index_capacity),
            config,
        }
    }

    pub fn request_glyph(
        &mut self,
        font: &Font<'_>,
        glyph_id: u16,
    ) -> Result<&CachedGlyph, CacheError> {
        if let Some(index) = self.lookup_index(glyph_id) {
            return Ok(&self.records[index]);
        }
        let outline = font.glyph_outline(glyph_id)?;
        self.cache_outline(glyph_id, &outline)
    }

    pub fn request_codepoint(
        &mut self,
        font: &Font<'_>,
        codepoint: char,
    ) -> Result<&CachedGlyph, CacheError> {
        self.request_glyph(font, font.glyph_index(codepoint))
    }

    pub fn cache_outline(
        &mut self,
        glyph_id: u16,
        outline: &GlyphOutline,
    ) -> Result<&CachedGlyph, CacheError> {
        if let Some(index) = self.lookup_index(glyph_id) {
            return Ok(&self.records[index]);
        }
        let encoded = encode_glyph(outline);
        self.cache_encoded(glyph_id, &encoded)
    }

    pub fn cache_encoded(
        &mut self,
        glyph_id: u16,
        encoded: &EncodedGlyph,
    ) -> Result<&CachedGlyph, CacheError> {
        if let Some(index) = self.lookup_index(glyph_id) {
            return Ok(&self.records[index]);
        }
        self.preflight(encoded)?;

        let descriptor_index = checked_u32(self.descriptors.len())?;
        let hband_base = checked_u32(self.bands.len())?;
        let vband_base = checked_u32(self.bands.len() + encoded.hbands.len())?;
        let curve_base = self.curves.len();

        self.curves.extend_from_slice(&encoded.curves);
        append_bands(
            &mut self.bands,
            &mut self.band_curve_indices,
            &encoded.hbands,
            curve_base,
        )?;
        append_bands(
            &mut self.bands,
            &mut self.band_curve_indices,
            &encoded.vbands,
            curve_base,
        )?;

        self.descriptors.push(TextGlyphDescriptor {
            bbox_em: encoded.bbox_em,
            band_scale: encoded.band_scale,
            band_offset: encoded.band_offset,
            hband_base,
            vband_base,
            band_max: pack_band_max(encoded.band_max[0], encoded.band_max[1]),
            _pad0: 0,
        });

        let record = CachedGlyph {
            glyph_id,
            descriptor_index,
            advance_em: encoded.advance_em,
            empty: encoded.curves.is_empty(),
        };
        self.records.push(record);
        self.lookup[usize::from(glyph_id)] = descriptor_index + 1;
        Ok(self.records.last().expect("just pushed glyph record"))
    }

    pub fn records(&self) -> &[CachedGlyph] {
        &self.records
    }

    pub fn descriptors(&self) -> &[TextGlyphDescriptor] {
        &self.descriptors
    }

    pub fn curves(&self) -> &[TextCurve] {
        &self.curves
    }

    pub fn bands(&self) -> &[TextBandHeader] {
        &self.bands
    }

    pub fn band_curve_indices(&self) -> &[u32] {
        &self.band_curve_indices
    }

    pub fn buffer_slices(&self) -> TextGlyphBufferSlices<'_> {
        TextGlyphBufferSlices {
            descriptors: &self.descriptors,
            curves: &self.curves,
            bands: &self.bands,
            band_curve_indices: &self.band_curve_indices,
        }
    }

    fn lookup_index(&self, glyph_id: u16) -> Option<usize> {
        let value = self.lookup[usize::from(glyph_id)];
        if value == 0 {
            None
        } else {
            Some((value - 1) as usize)
        }
    }

    fn preflight(&self, encoded: &EncodedGlyph) -> Result<(), CacheError> {
        if self.records.len() >= self.config.max_glyphs {
            return Err(CacheError::Capacity(CapacityKind::Glyphs));
        }
        if self.descriptors.len() >= self.config.max_glyphs {
            return Err(CacheError::Capacity(CapacityKind::Descriptors));
        }
        if self.curves.len() + encoded.curves.len() > self.config.curve_capacity {
            return Err(CacheError::Capacity(CapacityKind::Curves));
        }
        let band_count = encoded.hbands.len() + encoded.vbands.len();
        if self.bands.len() + band_count > self.config.band_capacity {
            return Err(CacheError::Capacity(CapacityKind::Bands));
        }
        let band_curve_count = encoded
            .hbands
            .iter()
            .chain(encoded.vbands.iter())
            .map(|band| band.curve_indices.len())
            .sum::<usize>();
        if self.band_curve_indices.len() + band_curve_count > self.config.band_curve_index_capacity
        {
            return Err(CacheError::Capacity(CapacityKind::BandCurveIndices));
        }
        checked_u32(self.descriptors.len())?;
        checked_u32(self.bands.len() + band_count)?;
        checked_u32(self.curves.len() + encoded.curves.len())?;
        checked_u32(self.band_curve_indices.len() + band_curve_count)?;
        Ok(())
    }
}

fn append_bands(
    headers: &mut Vec<TextBandHeader>,
    indices: &mut Vec<u32>,
    bands: &[EncodedBand],
    curve_base: usize,
) -> Result<(), CacheError> {
    for band in bands {
        let first = checked_u32(indices.len())?;
        let count = checked_u32(band.curve_indices.len())?;
        headers.push(TextBandHeader { first, count });
        for curve_index in &band.curve_indices {
            let absolute = curve_base
                .checked_add(*curve_index as usize)
                .ok_or(CacheError::Capacity(CapacityKind::U32Index))?;
            indices.push(checked_u32(absolute)?);
        }
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, CacheError> {
    u32::try_from(value).map_err(|_| CacheError::Capacity(CapacityKind::U32Index))
}

fn assign_horizontal_bands(glyph: &mut EncodedGlyph, outline: &GlyphOutline) {
    for band_index in 0..glyph.hbands.len() {
        let (band_min, band_max) = band_bounds(outline.bbox_em, glyph.hbands.len(), band_index, 1);
        let mut curves = Vec::new();
        for (curve_index, curve) in outline.curves.iter().enumerate() {
            if curve_horizontal(*curve) {
                continue;
            }
            let (curve_min, curve_max) = curve_bounds(*curve, 1);
            if intervals_overlap(curve_min, curve_max, band_min, band_max) {
                curves.push(curve_index as u32);
            }
        }
        sort_curve_indices(&mut curves, outline, 0);
        glyph.hbands[band_index].curve_indices = curves;
    }
}

fn assign_vertical_bands(glyph: &mut EncodedGlyph, outline: &GlyphOutline) {
    for band_index in 0..glyph.vbands.len() {
        let (band_min, band_max) = band_bounds(outline.bbox_em, glyph.vbands.len(), band_index, 0);
        let mut curves = Vec::new();
        for (curve_index, curve) in outline.curves.iter().enumerate() {
            if curve_vertical(*curve) {
                continue;
            }
            let (curve_min, curve_max) = curve_bounds(*curve, 0);
            if intervals_overlap(curve_min, curve_max, band_min, band_max) {
                curves.push(curve_index as u32);
            }
        }
        sort_curve_indices(&mut curves, outline, 1);
        glyph.vbands[band_index].curve_indices = curves;
    }
}

fn choose_band_count(curve_count: u32) -> u32 {
    if curve_count <= 1 {
        return 1;
    }
    let mut best = 1;
    for candidate in 1..=SLUG_MAX_BANDS {
        best = candidate;
        if candidate * candidate >= curve_count {
            break;
        }
    }
    best
}

fn axis_scale(count: u32, min_value: f32, max_value: f32) -> f32 {
    debug_assert!(count > 0);
    debug_assert!(min_value <= max_value);
    let span = max_value - min_value;
    if span <= 0.0 {
        0.0
    } else {
        count as f32 / span
    }
}

fn band_bounds(bbox: [f32; 4], count: usize, index: usize, axis: usize) -> (f32, f32) {
    debug_assert!(count > 0);
    debug_assert!(index < count);
    debug_assert!(axis <= 1);
    let min_value = bbox[axis];
    let max_value = bbox[axis + 2];
    let span = max_value - min_value;
    if span <= 0.0 {
        return (min_value, max_value);
    }
    let step = span / count as f32;
    let band_min = min_value + step * index as f32;
    let mut band_max = band_min + step;
    if index + 1 == count {
        band_max = max_value;
    }
    (band_min, band_max)
}

fn curve_bounds(curve: GlyphCurve, axis: usize) -> (f32, f32) {
    let min_value = curve.p1[axis].min(curve.p2[axis]).min(curve.p3[axis]);
    let max_value = curve.p1[axis].max(curve.p2[axis]).max(curve.p3[axis]);
    (min_value, max_value)
}

fn curve_sort_key(curve: GlyphCurve, axis: usize) -> f32 {
    curve_bounds(curve, axis).1
}

fn sort_curve_indices(curves: &mut [u32], outline: &GlyphOutline, sort_axis: usize) {
    for index in 1..curves.len() {
        let current = curves[index];
        let current_key = curve_sort_key(outline.curves[current as usize], sort_axis);
        let mut write = index;
        while write > 0 {
            let prev = curves[write - 1];
            let prev_key = curve_sort_key(outline.curves[prev as usize], sort_axis);
            if prev_key >= current_key {
                break;
            }
            curves[write] = prev;
            write -= 1;
        }
        curves[write] = current;
    }
}

fn intervals_overlap(curve_min: f32, curve_max: f32, band_min: f32, band_max: f32) -> bool {
    curve_min - SLUG_BAND_EPSILON <= band_max && curve_max + SLUG_BAND_EPSILON >= band_min
}

fn curve_horizontal(curve: GlyphCurve) -> bool {
    curve.p1[1] == curve.p2[1] && curve.p2[1] == curve.p3[1]
}

fn curve_vertical(curve: GlyphCurve) -> bool {
    curve.p1[0] == curve.p2[0] && curve.p2[0] == curve.p3[0]
}

fn clamped_band(value: f32, max_band: u16) -> u16 {
    if value <= 0.0 {
        0
    } else if value >= f32::from(max_band) {
        max_band
    } else {
        value as u16
    }
}

fn assert_outline(outline: &GlyphOutline) {
    assert!(outline.bbox_em[0] <= outline.bbox_em[2]);
    assert!(outline.bbox_em[1] <= outline.bbox_em[3]);
    assert!(outline.advance_em >= 0.0);
    let mut last_end = 0;
    for contour_end in &outline.contour_end {
        assert!(*contour_end > last_end);
        assert!(*contour_end <= outline.curves.len());
        last_end = *contour_end;
    }
    assert_eq!(last_end, outline.curves.len());
    assert!(outline.curves.is_empty() || !outline.contour_end.is_empty());
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;
    use crate::font::tests::tiny_ttf;
    use abi_ui::TextGlyphInstance;

    #[test]
    fn empty_outline_encodes_degenerate_single_bands() {
        let outline = GlyphOutline::empty(0.6);
        let glyph = encode_glyph(&outline);

        assert!(glyph.curves.is_empty());
        assert_eq!(glyph.hbands.len(), 1);
        assert_eq!(glyph.vbands.len(), 1);
        assert!(glyph.hbands[0].curve_indices.is_empty());
        assert!(glyph.vbands[0].curve_indices.is_empty());
        assert_eq!(glyph.band_scale, [0.0, 0.0]);
        assert_eq!(glyph.band_offset, [0.0, 0.0]);
        assert_eq!(glyph.band_max, [0, 0]);
    }

    #[test]
    fn bands_sort_and_selected_bands_cover_crossing_curves() {
        let outline = property_outline();
        let glyph = encode_glyph(&outline);

        assert_sorted(&glyph.hbands, &outline, 0);
        assert_sorted(&glyph.vbands, &outline, 1);

        for curve_index in 0..outline.curves.len() as u32 {
            let curve = outline.curves[curve_index as usize];
            if !curve_horizontal(curve) {
                assert!(band_hits(&glyph.hbands, curve_index) > 0);
            }
            if !curve_vertical(curve) {
                assert!(band_hits(&glyph.vbands, curve_index) > 0);
            }
        }

        for [x, y] in [[0.15, 0.2], [0.45, 0.5], [0.75, 0.25], [0.9, 0.8]] {
            let hband = select_band(
                y,
                glyph.band_scale[1],
                glyph.band_offset[1],
                glyph.band_max[1],
            );
            let vband = select_band(
                x,
                glyph.band_scale[0],
                glyph.band_offset[0],
                glyph.band_max[0],
            );
            assert_selected_band_covers(&glyph, &outline, hband, vband, x, y);
        }
    }

    #[test]
    fn cache_overflow_leaves_buffers_untouched() {
        let outline = square_outline();
        let mut cache = TextGlyphCache::new(TextGlyphCacheConfig {
            max_glyphs: 4,
            curve_capacity: 0,
            band_capacity: 16,
            band_curve_index_capacity: 16,
        });

        let before = snapshot_lengths(&cache);
        let result = cache.cache_outline(1, &outline);

        assert_eq!(result, Err(CacheError::Capacity(CapacityKind::Curves)));
        assert_eq!(snapshot_lengths(&cache), before);
        assert_eq!(cache.lookup[1], 0);
    }

    #[test]
    fn descriptor_and_instance_match_abi_expectations() {
        let outline = square_outline();
        let mut cache = TextGlyphCache::new(TextGlyphCacheConfig {
            max_glyphs: 4,
            curve_capacity: 16,
            band_capacity: 16,
            band_curve_index_capacity: 32,
        });
        let record = *cache
            .cache_outline(7, &outline)
            .expect("cache square glyph");

        assert_eq!(size_of::<TextGlyphDescriptor>(), 48);
        assert_eq!(size_of::<TextGlyphInstance>(), 16);
        assert_eq!(record.descriptor_index, 0);
        assert!(!record.empty);
        assert_ne!(cache.band_curve_indices().len(), 0);

        let descriptor = cache.descriptors()[0];
        assert_eq!(descriptor.bbox_em, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(descriptor.hband_base, 0);
        assert_eq!(
            descriptor.vband_base,
            cache.buffer_slices().bands.len() as u32 / 2
        );
        assert_eq!(descriptor.band_max, pack_band_max(1, 1));

        let instance = TextGlyphInstance {
            pen_doc: [2.0, 3.0],
            glyph_id: record.descriptor_index,
            color: 0xff00_ff00,
        };
        assert_eq!(instance.pen_doc, [2.0, 3.0]);
        assert_eq!(instance.glyph_id, 0);
        assert_eq!(instance.color, 0xff00_ff00);
    }

    #[test]
    fn request_glyph_reuses_cache_record_and_buffers() {
        let bytes = tiny_ttf();
        let font = Font::from_bytes(&bytes).expect("load tiny ttf");
        let mut cache = TextGlyphCache::new(TextGlyphCacheConfig {
            max_glyphs: 4,
            curve_capacity: 16,
            band_capacity: 16,
            band_curve_index_capacity: 32,
        });

        let first = *cache.request_codepoint(&font, 'A').expect("cache A");
        let lengths = snapshot_lengths(&cache);
        let second = *cache.request_codepoint(&font, 'A').expect("reuse A");

        assert_eq!(first, second);
        assert_eq!(snapshot_lengths(&cache), lengths);
        assert_eq!(cache.records().len(), 1);
        assert_eq!(cache.curves().len(), 3);
    }

    #[test]
    fn cached_glyphs_pack_absolute_curve_indices_and_band_bases() {
        let first_outline = square_outline();
        let second_outline = property_outline();
        let encoded_second = encode_glyph(&second_outline);
        let mut cache = TextGlyphCache::new(TextGlyphCacheConfig {
            max_glyphs: 4,
            curve_capacity: 32,
            band_capacity: 32,
            band_curve_index_capacity: 128,
        });

        cache
            .cache_outline(1, &first_outline)
            .expect("cache first glyph");
        let second_curve_base = cache.curves().len();
        let second_band_base = cache.bands().len();
        let second_index_base = cache.band_curve_indices().len();
        let second = *cache
            .cache_outline(2, &second_outline)
            .expect("cache second glyph");

        assert_eq!(second.descriptor_index, 1);
        assert_curves_eq(
            &cache.curves()[second_curve_base..],
            encoded_second.curves.as_slice(),
        );

        let descriptor = cache.descriptors()[second.descriptor_index as usize];
        assert_eq!(descriptor.hband_base, second_band_base as u32);
        assert_eq!(
            descriptor.vband_base,
            (second_band_base + encoded_second.hbands.len()) as u32
        );
        assert_eq!(
            descriptor.band_max,
            pack_band_max(encoded_second.band_max[0], encoded_second.band_max[1])
        );

        let vband_index_base =
            second_index_base + encoded_band_curve_count(encoded_second.hbands.as_slice());
        assert_packed_bands(
            &cache,
            second_band_base,
            second_index_base,
            second_curve_base,
            encoded_second.hbands.as_slice(),
        );
        assert_packed_bands(
            &cache,
            second_band_base + encoded_second.hbands.len(),
            vband_index_base,
            second_curve_base,
            encoded_second.vbands.as_slice(),
        );
    }

    #[test]
    fn encoded_glyph_caps_bands_at_sixteen_per_axis_and_packs_maxima() {
        let outline = dense_diagonal_outline(257);
        let glyph = encode_glyph(&outline);

        assert_eq!(glyph.hbands.len(), SLUG_MAX_BANDS as usize);
        assert_eq!(glyph.vbands.len(), SLUG_MAX_BANDS as usize);
        assert_eq!(glyph.band_max, [15, 15]);

        let mut cache = TextGlyphCache::new(TextGlyphCacheConfig {
            max_glyphs: 1,
            curve_capacity: glyph.curves.len(),
            band_capacity: glyph.hbands.len() + glyph.vbands.len(),
            band_curve_index_capacity: encoded_band_curve_count(glyph.hbands.as_slice())
                + encoded_band_curve_count(glyph.vbands.as_slice()),
        });
        let record = *cache.cache_encoded(9, &glyph).expect("cache dense glyph");
        let descriptor = cache.descriptors()[record.descriptor_index as usize];

        assert_eq!(descriptor.hband_base, 0);
        assert_eq!(descriptor.vband_base, SLUG_MAX_BANDS);
        assert_eq!(descriptor.band_max, 0x000f_000f);
    }

    #[test]
    fn band_curve_index_overflow_preserves_existing_buffers_and_lookup() {
        let first_outline = square_outline();
        let first = encode_glyph(&first_outline);
        let first_index_capacity = encoded_band_curve_count(first.hbands.as_slice())
            + encoded_band_curve_count(first.vbands.as_slice());
        let mut cache = TextGlyphCache::new(TextGlyphCacheConfig {
            max_glyphs: 4,
            curve_capacity: 32,
            band_capacity: 32,
            band_curve_index_capacity: first_index_capacity,
        });
        cache.cache_encoded(1, &first).expect("cache first glyph");
        let before = snapshot_buffers(&cache);

        let result = cache.cache_outline(2, &property_outline());

        assert_eq!(
            result,
            Err(CacheError::Capacity(CapacityKind::BandCurveIndices))
        );
        assert_eq!(snapshot_buffers(&cache), before);
        assert_eq!(cache.lookup[2], 0);
    }

    fn snapshot_lengths(cache: &TextGlyphCache) -> (usize, usize, usize, usize, usize) {
        (
            cache.records().len(),
            cache.descriptors().len(),
            cache.curves().len(),
            cache.bands().len(),
            cache.band_curve_indices().len(),
        )
    }

    #[derive(Debug, PartialEq)]
    struct CacheBufferSnapshot {
        records: Vec<CachedGlyph>,
        descriptors: Vec<DescriptorSnapshot>,
        curves: Vec<CurveSnapshot>,
        bands: Vec<(u32, u32)>,
        band_curve_indices: Vec<u32>,
    }

    #[derive(Debug, PartialEq)]
    struct DescriptorSnapshot {
        bbox_em: [f32; 4],
        band_scale: [f32; 2],
        band_offset: [f32; 2],
        hband_base: u32,
        vband_base: u32,
        band_max: u32,
        pad0: u32,
    }

    #[derive(Debug, PartialEq)]
    struct CurveSnapshot {
        p1: [f32; 2],
        p2: [f32; 2],
        p3: [f32; 2],
    }

    fn snapshot_buffers(cache: &TextGlyphCache) -> CacheBufferSnapshot {
        CacheBufferSnapshot {
            records: cache.records().to_vec(),
            descriptors: cache
                .descriptors()
                .iter()
                .map(|descriptor| DescriptorSnapshot {
                    bbox_em: descriptor.bbox_em,
                    band_scale: descriptor.band_scale,
                    band_offset: descriptor.band_offset,
                    hband_base: descriptor.hband_base,
                    vband_base: descriptor.vband_base,
                    band_max: descriptor.band_max,
                    pad0: descriptor._pad0,
                })
                .collect(),
            curves: cache
                .curves()
                .iter()
                .map(|curve| CurveSnapshot {
                    p1: curve.p1,
                    p2: curve.p2,
                    p3: curve.p3,
                })
                .collect(),
            bands: cache
                .bands()
                .iter()
                .map(|band| (band.first, band.count))
                .collect(),
            band_curve_indices: cache.band_curve_indices().to_vec(),
        }
    }

    fn assert_curves_eq(actual: &[TextCurve], expected: &[TextCurve]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual.p1, expected.p1);
            assert_eq!(actual.p2, expected.p2);
            assert_eq!(actual.p3, expected.p3);
        }
    }

    fn encoded_band_curve_count(bands: &[EncodedBand]) -> usize {
        bands.iter().map(|band| band.curve_indices.len()).sum()
    }

    fn assert_packed_bands(
        cache: &TextGlyphCache,
        band_base: usize,
        index_base: usize,
        curve_base: usize,
        expected: &[EncodedBand],
    ) {
        let bands = &cache.bands()[band_base..band_base + expected.len()];
        let mut next_index = index_base;
        for (header, expected_band) in bands.iter().zip(expected.iter()) {
            assert_eq!(header.first as usize, next_index);
            assert_eq!(header.count as usize, expected_band.curve_indices.len());
            for (offset, expected_curve_index) in expected_band.curve_indices.iter().enumerate() {
                assert_eq!(
                    cache.band_curve_indices()[next_index + offset],
                    (curve_base + *expected_curve_index as usize) as u32
                );
            }
            next_index += expected_band.curve_indices.len();
        }
    }

    fn assert_sorted(bands: &[EncodedBand], outline: &GlyphOutline, sort_axis: usize) {
        for band in bands {
            for pair in band.curve_indices.windows(2) {
                let prev = curve_sort_key(outline.curves[pair[0] as usize], sort_axis);
                let current = curve_sort_key(outline.curves[pair[1] as usize], sort_axis);
                assert!(prev >= current);
            }
        }
    }

    fn assert_selected_band_covers(
        glyph: &EncodedGlyph,
        outline: &GlyphOutline,
        hband: u16,
        vband: u16,
        x: f32,
        y: f32,
    ) {
        for (curve_index, curve) in outline.curves.iter().copied().enumerate() {
            let curve_index = curve_index as u32;
            let (y_min, y_max) = curve_bounds(curve, 1);
            if !curve_horizontal(curve)
                && y + SLUG_BAND_EPSILON >= y_min
                && y - SLUG_BAND_EPSILON <= y_max
            {
                assert!(
                    glyph.hbands[hband as usize]
                        .curve_indices
                        .contains(&curve_index)
                );
            }
            let (x_min, x_max) = curve_bounds(curve, 0);
            if !curve_vertical(curve)
                && x + SLUG_BAND_EPSILON >= x_min
                && x - SLUG_BAND_EPSILON <= x_max
            {
                assert!(
                    glyph.vbands[vband as usize]
                        .curve_indices
                        .contains(&curve_index)
                );
            }
        }
    }

    fn band_hits(bands: &[EncodedBand], curve_index: u32) -> u32 {
        bands
            .iter()
            .flat_map(|band| band.curve_indices.iter())
            .filter(|index| **index == curve_index)
            .count() as u32
    }

    fn square_outline() -> GlyphOutline {
        make_outline(
            &[
                line(0.0, 0.0, 1.0, 0.0),
                line(1.0, 0.0, 1.0, 1.0),
                line(1.0, 1.0, 0.0, 1.0),
                line(0.0, 1.0, 0.0, 0.0),
            ],
            &[4],
            [0.0, 0.0, 1.0, 1.0],
        )
    }

    fn property_outline() -> GlyphOutline {
        make_outline(
            &[
                line(0.0, 0.0, 1.0, 0.0),
                line(1.0, 0.0, 0.2, 0.8),
                line(0.2, 0.8, 0.0, 0.0),
                GlyphCurve {
                    p1: [0.2, 0.2],
                    p2: [0.5, 0.9],
                    p3: [0.8, 0.2],
                },
                line(0.8, 0.2, 0.8, 0.7),
                line(0.8, 0.7, 0.2, 0.7),
            ],
            &[3, 6],
            [0.0, 0.0, 1.0, 1.0],
        )
    }

    fn dense_diagonal_outline(curve_count: usize) -> GlyphOutline {
        let mut curves = Vec::with_capacity(curve_count);
        for index in 0..curve_count {
            let column = (index % SLUG_MAX_BANDS as usize) as f32;
            let row = (index / SLUG_MAX_BANDS as usize) as f32;
            let x0 = (column + 0.25) / (SLUG_MAX_BANDS as f32 + 1.0);
            let y0 = (row + 0.25) / (SLUG_MAX_BANDS as f32 + 1.0);
            let x1 = (x0 + 0.03).min(0.99);
            let y1 = (y0 + 0.03).min(0.99);
            curves.push(line(x0, y0, x1, y1));
        }
        make_outline(&curves, &[curve_count], [0.0, 0.0, 1.0, 1.0])
    }

    fn make_outline(
        curves: &[GlyphCurve],
        contour_end: &[usize],
        bbox_em: [f32; 4],
    ) -> GlyphOutline {
        GlyphOutline {
            curves: curves.to_vec(),
            contour_end: contour_end.to_vec(),
            advance_em: 1.0,
            bbox_em,
        }
    }

    fn line(x0: f32, y0: f32, x1: f32, y1: f32) -> GlyphCurve {
        GlyphCurve {
            p1: [x0, y0],
            p2: [x1, y1],
            p3: [x1, y1],
        }
    }
}
