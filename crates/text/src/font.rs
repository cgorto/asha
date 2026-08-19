//! TrueType parsing for format-4 cmap and simple quadratic glyf outlines.
//!
//! Composite glyphs are unsupported; metrics scale coordinates into em units.

use crate::encode::{GlyphCurve, GlyphOutline};

const TRUE_TYPE_TAG: u32 = 0x0001_0000;
const TAG_CFF: u32 = 0x4346_4620;
const TAG_CMAP: u32 = 0x636d_6170;
const TAG_GLYF: u32 = 0x676c_7966;
const TAG_HEAD: u32 = 0x6865_6164;
const TAG_HHEA: u32 = 0x6868_6561;
const TAG_HMTX: u32 = 0x686d_7478;
const TAG_LOCA: u32 = 0x6c6f_6361;
const TAG_MAXP: u32 = 0x6d61_7870;
const TAG_OS2: u32 = 0x4f53_2f32;

const FLAG_ON_CURVE: u8 = 0x01;
const FLAG_X_SHORT: u8 = 0x02;
const FLAG_Y_SHORT: u8 = 0x04;
const FLAG_REPEAT: u8 = 0x08;
const FLAG_X_SAME: u8 = 0x10;
const FLAG_Y_SAME: u8 = 0x20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontError {
    Truncated,
    UnsupportedSfnt,
    UnsupportedCff,
    MissingTable,
    InvalidTable,
    UnsupportedComposite,
    GlyphOutOfRange,
}

#[derive(Debug, Clone, Copy, Default)]
struct FontTable {
    offset: u32,
    size: u32,
}

impl FontTable {
    fn present(self) -> bool {
        self.size != 0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Font<'a> {
    bytes: &'a [u8],
    os2: FontTable,
    cmap: FontTable,
    glyf: FontTable,
    head: FontTable,
    hhea: FontTable,
    hmtx: FontTable,
    loca: FontTable,
    maxp: FontTable,
    units_per_em: u16,
    index_to_loc_format: i16,
    num_glyphs: u16,
    num_h_metrics: u16,
    ascent: i16,
    descent: i16,
    line_gap: i16,
    cap_height: i16,
    x_height: i16,
}

impl<'a> Font<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, FontError> {
        let mut font = Self {
            bytes,
            os2: FontTable::default(),
            cmap: FontTable::default(),
            glyf: FontTable::default(),
            head: FontTable::default(),
            hhea: FontTable::default(),
            hmtx: FontTable::default(),
            loca: FontTable::default(),
            maxp: FontTable::default(),
            units_per_em: 0,
            index_to_loc_format: 0,
            num_glyphs: 0,
            num_h_metrics: 0,
            ascent: 0,
            descent: 0,
            line_gap: 0,
            cap_height: 0,
            x_height: 0,
        };
        font.load_tables()?;
        font.validate_tables()?;
        font.load_metrics()?;
        Ok(font)
    }

    pub fn units_per_em(&self) -> u16 {
        self.units_per_em
    }

    pub fn num_glyphs(&self) -> u16 {
        self.num_glyphs
    }

    pub fn num_h_metrics(&self) -> u16 {
        self.num_h_metrics
    }

    pub fn ascent(&self) -> i16 {
        self.ascent
    }

    pub fn descent(&self) -> i16 {
        self.descent
    }

    pub fn line_gap(&self) -> i16 {
        self.line_gap
    }

    pub fn cap_height(&self) -> i16 {
        self.cap_height
    }

    pub fn x_height(&self) -> i16 {
        self.x_height
    }

    pub fn glyph_index(&self, codepoint: char) -> u16 {
        let codepoint = codepoint as u32;
        if codepoint > u16::MAX as u32 {
            return 0;
        }
        let Ok((table_offset, table_size)) = self.cmap_format4() else {
            return 0;
        };
        self.cmap_format4_glyph(table_offset, table_size, codepoint as u16)
    }

    pub fn glyph_outline(&self, glyph_id: u16) -> Result<GlyphOutline, FontError> {
        if glyph_id >= self.num_glyphs {
            return Err(FontError::GlyphOutOfRange);
        }

        let mut outline = GlyphOutline {
            curves: Vec::new(),
            contour_end: Vec::new(),
            advance_em: self.glyph_advance_em(glyph_id)?,
            bbox_em: [0.0; 4],
        };
        let (glyph_start, glyph_end) = self.glyph_range(glyph_id)?;
        if glyph_start == glyph_end {
            return Ok(outline);
        }

        let number_of_contours = read_i16(self.bytes, glyph_start)?;
        if number_of_contours < 0 {
            return Err(FontError::UnsupportedComposite);
        }
        if number_of_contours == 0 {
            return Ok(outline);
        }

        outline.bbox_em = self.glyph_bbox(glyph_start)?;
        let (points, contour_ends) =
            self.glyph_outline_points(glyph_start, glyph_end, number_of_contours as u16)?;
        self.glyph_outline_curves(&mut outline, &points, &contour_ends)?;
        Ok(outline)
    }

    fn load_tables(&mut self) -> Result<(), FontError> {
        if self.bytes.len() < 12 {
            return Err(FontError::Truncated);
        }
        let sfnt_tag = read_u32(self.bytes, 0)?;
        if sfnt_tag != TRUE_TYPE_TAG {
            return Err(FontError::UnsupportedSfnt);
        }

        let num_tables = read_u16(self.bytes, 4)?;
        let directory_size = 12usize
            .checked_add(num_tables as usize * 16)
            .ok_or(FontError::InvalidTable)?;
        if directory_size > self.bytes.len() {
            return Err(FontError::Truncated);
        }

        for table_index in 0..num_tables {
            let record_offset = 12 + u32::from(table_index) * 16;
            self.load_table_record(record_offset)?;
        }
        Ok(())
    }

    fn load_table_record(&mut self, record_offset: u32) -> Result<(), FontError> {
        let tag = read_u32(self.bytes, record_offset)?;
        let offset = read_u32(self.bytes, record_offset + 8)?;
        let size = read_u32(self.bytes, record_offset + 12)?;
        let end = offset.checked_add(size).ok_or(FontError::InvalidTable)?;
        if end as usize > self.bytes.len() {
            return Err(FontError::Truncated);
        }

        let table = FontTable { offset, size };
        match tag {
            TAG_OS2 => self.os2 = table,
            TAG_CMAP => self.cmap = table,
            TAG_GLYF => self.glyf = table,
            TAG_HEAD => self.head = table,
            TAG_HHEA => self.hhea = table,
            TAG_HMTX => self.hmtx = table,
            TAG_LOCA => self.loca = table,
            TAG_MAXP => self.maxp = table,
            TAG_CFF => return Err(FontError::UnsupportedCff),
            _ => {}
        }
        Ok(())
    }

    fn validate_tables(&self) -> Result<(), FontError> {
        if !self.head.present()
            || !self.maxp.present()
            || !self.cmap.present()
            || !self.hhea.present()
            || !self.os2.present()
            || !self.glyf.present()
            || !self.hmtx.present()
            || !self.loca.present()
        {
            return Err(FontError::MissingTable);
        }
        if self.head.size < 54
            || self.maxp.size < 6
            || self.cmap.size < 4
            || self.hhea.size < 36
            || self.os2.size < 90
        {
            return Err(FontError::InvalidTable);
        }
        Ok(())
    }

    fn load_metrics(&mut self) -> Result<(), FontError> {
        self.units_per_em = self.table_u16(self.head, 18)?;
        self.index_to_loc_format = self.table_i16(self.head, 50)?;
        self.num_glyphs = self.table_u16(self.maxp, 4)?;
        self.num_h_metrics = self.table_u16(self.hhea, 34)?;
        self.ascent = self.table_i16(self.hhea, 4)?;
        self.descent = self.table_i16(self.hhea, 6)?;
        self.line_gap = self.table_i16(self.hhea, 8)?;
        self.x_height = self.table_i16(self.os2, 86)?;
        self.cap_height = self.table_i16(self.os2, 88)?;

        if self.units_per_em == 0 || self.num_glyphs == 0 || self.num_h_metrics == 0 {
            return Err(FontError::InvalidTable);
        }
        if self.num_h_metrics > self.num_glyphs {
            return Err(FontError::InvalidTable);
        }
        if self.index_to_loc_format != 0 && self.index_to_loc_format != 1 {
            return Err(FontError::InvalidTable);
        }
        let loca_entry_size = if self.index_to_loc_format == 0 { 2 } else { 4 };
        if (usize::from(self.num_glyphs) + 1) * loca_entry_size > self.loca.size as usize {
            return Err(FontError::InvalidTable);
        }
        if usize::from(self.num_h_metrics) * 4 > self.hmtx.size as usize {
            return Err(FontError::InvalidTable);
        }
        Ok(())
    }

    fn cmap_format4(&self) -> Result<(u32, u32), FontError> {
        let num_subtables = self.table_u16(self.cmap, 2)?;
        if 4usize + usize::from(num_subtables) * 8 > self.cmap.size as usize {
            return Err(FontError::InvalidTable);
        }

        for index in 0..num_subtables {
            let record_offset = 4 + u32::from(index) * 8;
            let platform_id = self.table_u16(self.cmap, record_offset)?;
            let encoding_id = self.table_u16(self.cmap, record_offset + 2)?;
            let subtable = self.table_u32(self.cmap, record_offset + 4)?;
            if cmap_is_unicode(platform_id, encoding_id) {
                if let Ok(found) = self.cmap_format4_at(subtable) {
                    return Ok(found);
                }
            }
        }
        Err(FontError::InvalidTable)
    }

    fn cmap_format4_at(&self, subtable: u32) -> Result<(u32, u32), FontError> {
        if subtable.checked_add(16).ok_or(FontError::InvalidTable)? > self.cmap.size {
            return Err(FontError::InvalidTable);
        }
        let format = self.table_u16(self.cmap, subtable)?;
        if format != 4 {
            return Err(FontError::InvalidTable);
        }
        let length = u32::from(self.table_u16(self.cmap, subtable + 2)?);
        if length < 16
            || subtable
                .checked_add(length)
                .ok_or(FontError::InvalidTable)?
                > self.cmap.size
        {
            return Err(FontError::InvalidTable);
        }
        Ok((self.cmap.offset + subtable, length))
    }

    fn cmap_format4_glyph(&self, table_offset: u32, table_size: u32, codepoint: u16) -> u16 {
        let Ok(seg_count_x2) = read_u16(self.bytes, table_offset + 6) else {
            return 0;
        };
        if seg_count_x2 % 2 != 0 {
            return 0;
        }
        let seg_count = seg_count_x2 / 2;
        let array_size = u32::from(seg_count) * 2;
        if 16 + array_size * 4 > table_size {
            return 0;
        }

        let end_codes = table_offset + 14;
        let start_codes = end_codes + array_size + 2;
        let id_deltas = start_codes + array_size;
        let id_range_offsets = id_deltas + array_size;
        for segment in 0..seg_count {
            if let Some(glyph_id) = self.cmap_format4_segment(
                table_offset,
                table_size,
                codepoint,
                Cmap4Arrays {
                    end_codes,
                    start_codes,
                    id_deltas,
                    id_range_offsets,
                },
                segment,
            ) {
                return glyph_id;
            }
        }
        0
    }

    fn cmap_format4_segment(
        &self,
        table_offset: u32,
        table_size: u32,
        codepoint: u16,
        arrays: Cmap4Arrays,
        segment: u16,
    ) -> Option<u16> {
        let entry_offset = u32::from(segment) * 2;
        let end_code = read_u16(self.bytes, arrays.end_codes + entry_offset).ok()?;
        if codepoint > end_code {
            return None;
        }
        let start_code = read_u16(self.bytes, arrays.start_codes + entry_offset).ok()?;
        if codepoint < start_code {
            return Some(0);
        }
        let id_delta = read_i16(self.bytes, arrays.id_deltas + entry_offset).ok()?;
        let id_range_offset = read_u16(self.bytes, arrays.id_range_offsets + entry_offset).ok()?;
        if id_range_offset == 0 {
            return Some(apply_id_delta(codepoint, id_delta));
        }

        let glyph_offset = arrays
            .id_range_offsets
            .checked_add(entry_offset)?
            .checked_add(u32::from(id_range_offset))?
            .checked_add(u32::from(codepoint - start_code) * 2)?;
        if glyph_offset + 2 > table_offset + table_size {
            return Some(0);
        }
        let glyph_id = read_u16(self.bytes, glyph_offset).ok()?;
        if glyph_id == 0 {
            Some(0)
        } else {
            Some(apply_id_delta(glyph_id, id_delta))
        }
    }

    fn glyph_outline_points(
        &self,
        glyph_start: u32,
        glyph_end: u32,
        contour_count: u16,
    ) -> Result<(Vec<FontPoint>, Vec<u16>), FontError> {
        let ends_offset = glyph_start + 10;
        if ends_offset + u32::from(contour_count) * 2 + 2 > glyph_end {
            return Err(FontError::Truncated);
        }

        let mut contour_ends = Vec::with_capacity(contour_count as usize);
        for contour in 0..contour_count {
            let end_point = read_u16(self.bytes, ends_offset + u32::from(contour) * 2)?;
            if let Some(previous) = contour_ends.last() {
                if end_point <= *previous {
                    return Err(FontError::InvalidTable);
                }
            }
            contour_ends.push(end_point);
        }

        let point_count = usize::from(*contour_ends.last().ok_or(FontError::InvalidTable)?) + 1;
        let instruction_offset = ends_offset + u32::from(contour_count) * 2;
        let instruction_length = u32::from(read_u16(self.bytes, instruction_offset)?);
        let cursor = instruction_offset + 2 + instruction_length;
        if cursor > glyph_end {
            return Err(FontError::Truncated);
        }

        let mut points = vec![FontPoint::default(); point_count];
        let mut flags = vec![0; point_count];
        let flags_cursor = decode_flags(self.bytes, cursor, glyph_end, &mut flags)?;
        let x_cursor = self.decode_x(flags_cursor, glyph_end, &flags, &mut points)?;
        self.decode_y(x_cursor, glyph_end, &flags, &mut points)?;
        Ok((points, contour_ends))
    }

    fn decode_x(
        &self,
        cursor: u32,
        limit: u32,
        flags: &[u8],
        points: &mut [FontPoint],
    ) -> Result<u32, FontError> {
        let mut next_cursor = cursor;
        let mut x = 0i32;
        let scale = 1.0 / f32::from(self.units_per_em);
        for (flag, point) in flags.iter().zip(points.iter_mut()) {
            if flag & FLAG_X_SHORT != 0 {
                if next_cursor + 1 > limit {
                    return Err(FontError::Truncated);
                }
                let delta = i32::from(self.bytes[next_cursor as usize]);
                next_cursor += 1;
                if flag & FLAG_X_SAME != 0 {
                    x += delta;
                } else {
                    x -= delta;
                }
            } else if flag & FLAG_X_SAME == 0 {
                let delta = read_i16(self.bytes, next_cursor)?;
                if next_cursor + 2 > limit {
                    return Err(FontError::Truncated);
                }
                next_cursor += 2;
                x += i32::from(delta);
            }
            point.position[0] = x as f32 * scale;
            point.on_curve = flag & FLAG_ON_CURVE != 0;
        }
        Ok(next_cursor)
    }

    fn decode_y(
        &self,
        cursor: u32,
        limit: u32,
        flags: &[u8],
        points: &mut [FontPoint],
    ) -> Result<u32, FontError> {
        let mut next_cursor = cursor;
        let mut y = 0i32;
        let scale = 1.0 / f32::from(self.units_per_em);
        for (flag, point) in flags.iter().zip(points.iter_mut()) {
            if flag & FLAG_Y_SHORT != 0 {
                if next_cursor + 1 > limit {
                    return Err(FontError::Truncated);
                }
                let delta = i32::from(self.bytes[next_cursor as usize]);
                next_cursor += 1;
                if flag & FLAG_Y_SAME != 0 {
                    y += delta;
                } else {
                    y -= delta;
                }
            } else if flag & FLAG_Y_SAME == 0 {
                let delta = read_i16(self.bytes, next_cursor)?;
                if next_cursor + 2 > limit {
                    return Err(FontError::Truncated);
                }
                next_cursor += 2;
                y += i32::from(delta);
            }
            point.position[1] = y as f32 * scale;
        }
        Ok(next_cursor)
    }

    fn glyph_outline_curves(
        &self,
        outline: &mut GlyphOutline,
        points: &[FontPoint],
        contour_ends: &[u16],
    ) -> Result<(), FontError> {
        let mut contour_start = 0usize;
        for contour_end in contour_ends {
            let end = usize::from(*contour_end) + 1;
            if end > points.len() || contour_start >= end {
                return Err(FontError::InvalidTable);
            }
            glyph_outline_contour(outline, &points[contour_start..end]);
            // Degenerate contours have no area; omit them.
            let recorded = outline.contour_end.last().copied().unwrap_or(0);
            if outline.curves.len() > recorded {
                outline.contour_end.push(outline.curves.len());
            }
            contour_start = end;
        }
        if contour_start != points.len() {
            return Err(FontError::InvalidTable);
        }
        Ok(())
    }

    fn glyph_range(&self, glyph_id: u16) -> Result<(u32, u32), FontError> {
        let start = self.loca_offset(glyph_id)?;
        let end = self.loca_offset(glyph_id + 1)?;
        if end < start || end > self.glyf.size {
            return Err(FontError::InvalidTable);
        }
        Ok((self.glyf.offset + start, self.glyf.offset + end))
    }

    fn loca_offset(&self, glyph_id: u16) -> Result<u32, FontError> {
        if self.index_to_loc_format == 0 {
            Ok(u32::from(self.table_u16(self.loca, u32::from(glyph_id) * 2)?) * 2)
        } else {
            self.table_u32(self.loca, u32::from(glyph_id) * 4)
        }
    }

    fn glyph_bbox(&self, glyph_start: u32) -> Result<[f32; 4], FontError> {
        let scale = 1.0 / f32::from(self.units_per_em);
        Ok([
            f32::from(read_i16(self.bytes, glyph_start + 2)?) * scale,
            f32::from(read_i16(self.bytes, glyph_start + 4)?) * scale,
            f32::from(read_i16(self.bytes, glyph_start + 6)?) * scale,
            f32::from(read_i16(self.bytes, glyph_start + 8)?) * scale,
        ])
    }

    fn glyph_advance_em(&self, glyph_id: u16) -> Result<f32, FontError> {
        let metric_index = glyph_id.min(self.num_h_metrics - 1);
        let advance = self.table_u16(self.hmtx, u32::from(metric_index) * 4)?;
        Ok(f32::from(advance) / f32::from(self.units_per_em))
    }

    fn table_u16(self, table: FontTable, relative_offset: u32) -> Result<u16, FontError> {
        if relative_offset + 2 > table.size {
            return Err(FontError::Truncated);
        }
        read_u16(self.bytes, table.offset + relative_offset)
    }

    fn table_i16(self, table: FontTable, relative_offset: u32) -> Result<i16, FontError> {
        if relative_offset + 2 > table.size {
            return Err(FontError::Truncated);
        }
        read_i16(self.bytes, table.offset + relative_offset)
    }

    fn table_u32(self, table: FontTable, relative_offset: u32) -> Result<u32, FontError> {
        if relative_offset + 4 > table.size {
            return Err(FontError::Truncated);
        }
        read_u32(self.bytes, table.offset + relative_offset)
    }
}

#[derive(Debug, Clone, Copy)]
struct Cmap4Arrays {
    end_codes: u32,
    start_codes: u32,
    id_deltas: u32,
    id_range_offsets: u32,
}

#[derive(Debug, Clone, Copy, Default)]
struct FontPoint {
    position: [f32; 2],
    on_curve: bool,
}

fn glyph_outline_contour(outline: &mut GlyphOutline, points: &[FontPoint]) {
    let (start, mut cursor) = contour_start(points);
    let mut current = start;
    while cursor < points.len() {
        let point = points[cursor];
        if point.on_curve {
            outline.curves.push(GlyphCurve {
                p1: current,
                p2: point.position,
                p3: point.position,
            });
            current = point.position;
            cursor += 1;
        } else {
            let next = if cursor + 1 < points.len() {
                points[cursor + 1]
            } else {
                points[0]
            };
            if next.on_curve {
                outline.curves.push(GlyphCurve {
                    p1: current,
                    p2: point.position,
                    p3: next.position,
                });
                current = next.position;
                cursor += 2;
            } else {
                let midpoint = point_midpoint(point.position, next.position);
                outline.curves.push(GlyphCurve {
                    p1: current,
                    p2: point.position,
                    p3: midpoint,
                });
                current = midpoint;
                cursor += 1;
            }
        }
    }
    if current != start {
        outline.curves.push(GlyphCurve {
            p1: current,
            p2: start,
            p3: start,
        });
    }
}

fn contour_start(points: &[FontPoint]) -> ([f32; 2], usize) {
    let first = points[0];
    let last = points[points.len() - 1];
    if first.on_curve {
        (first.position, 1)
    } else if last.on_curve {
        (last.position, 0)
    } else {
        (point_midpoint(last.position, first.position), 0)
    }
}

fn point_midpoint(left: [f32; 2], right: [f32; 2]) -> [f32; 2] {
    [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5]
}

fn decode_flags(bytes: &[u8], cursor: u32, limit: u32, flags: &mut [u8]) -> Result<u32, FontError> {
    let mut next_cursor = cursor;
    let mut flag_count = 0;
    while flag_count < flags.len() {
        if next_cursor + 1 > limit || next_cursor as usize >= bytes.len() {
            return Err(FontError::Truncated);
        }
        let flag = bytes[next_cursor as usize];
        next_cursor += 1;
        flags[flag_count] = flag;
        flag_count += 1;
        if flag & FLAG_REPEAT != 0 {
            if next_cursor + 1 > limit || next_cursor as usize >= bytes.len() {
                return Err(FontError::Truncated);
            }
            let repeat_count = bytes[next_cursor as usize] as usize;
            next_cursor += 1;
            if flag_count + repeat_count > flags.len() {
                return Err(FontError::InvalidTable);
            }
            for repeat_index in 0..repeat_count {
                flags[flag_count + repeat_index] = flag;
            }
            flag_count += repeat_count;
        }
    }
    Ok(next_cursor)
}

fn cmap_is_unicode(platform_id: u16, encoding_id: u16) -> bool {
    platform_id == 0 || (platform_id == 3 && (encoding_id == 1 || encoding_id == 10))
}

fn apply_id_delta(value: u16, delta: i16) -> u16 {
    value.wrapping_add(delta as u16)
}

fn read_u16(bytes: &[u8], offset: u32) -> Result<u16, FontError> {
    let offset = offset as usize;
    let end = offset.checked_add(2).ok_or(FontError::InvalidTable)?;
    if end > bytes.len() {
        return Err(FontError::Truncated);
    }
    Ok(u16::from(bytes[offset]) << 8 | u16::from(bytes[offset + 1]))
}

fn read_i16(bytes: &[u8], offset: u32) -> Result<i16, FontError> {
    Ok(read_u16(bytes, offset)? as i16)
}

fn read_u32(bytes: &[u8], offset: u32) -> Result<u32, FontError> {
    let offset = offset as usize;
    let end = offset.checked_add(4).ok_or(FontError::InvalidTable)?;
    if end > bytes.len() {
        return Err(FontError::Truncated);
    }
    Ok(u32::from(bytes[offset]) << 24
        | u32::from(bytes[offset + 1]) << 16
        | u32::from(bytes[offset + 2]) << 8
        | u32::from(bytes[offset + 3]))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const EPSILON: f32 = 0.000_01;

    #[test]
    fn tiny_truetype_fixture_parses_cmap_empty_and_simple_glyphs() {
        let bytes = tiny_ttf();
        let font = Font::from_bytes(&bytes).expect("load tiny ttf");

        assert_eq!(font.units_per_em(), 1000);
        assert_eq!(font.num_glyphs(), 3);
        assert_eq!(font.num_h_metrics(), 2);
        assert_eq!(font.ascent(), 800);
        assert_eq!(font.descent(), -200);
        assert_eq!(font.cap_height(), 700);
        assert_eq!(font.x_height(), 500);
        assert_eq!(font.glyph_index('A'), 1);
        assert_eq!(font.glyph_index(' '), 2);
        assert_eq!(font.glyph_index('B'), 0);

        let outline = font.glyph_outline(1).expect("outline A");
        assert_eq!(outline.contour_end, [3]);
        assert_eq!(outline.curves.len(), 3);
        assert_near(outline.advance_em, 0.6);
        assert_eq!(outline.bbox_em, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(outline.curves[0].p1, [0.0, 0.0]);
        assert_eq!(outline.curves[0].p3, [1.0, 0.0]);

        let space = font.glyph_outline(2).expect("outline space");
        assert!(space.curves.is_empty());
        assert!(space.contour_end.is_empty());
        assert_near(space.advance_em, 0.6);
    }

    fn assert_near(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "{actual} != {expected}"
        );
    }

    pub(crate) fn tiny_ttf() -> Vec<u8> {
        let tables = [
            (*b"OS/2", os2_table()),
            (*b"cmap", cmap_table()),
            (*b"glyf", glyf_table()),
            (*b"head", head_table()),
            (*b"hhea", hhea_table()),
            (*b"hmtx", hmtx_table()),
            (*b"loca", loca_table()),
            (*b"maxp", maxp_table()),
        ];
        let mut bytes = Vec::new();
        push_u32(&mut bytes, TRUE_TYPE_TAG);
        push_u16(&mut bytes, tables.len() as u16);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);

        let directory_start = bytes.len();
        bytes.resize(directory_start + tables.len() * 16, 0);
        let mut records = Vec::new();
        for (tag, data) in tables {
            while bytes.len() % 4 != 0 {
                bytes.push(0);
            }
            let offset = bytes.len() as u32;
            let size = data.len() as u32;
            bytes.extend_from_slice(&data);
            records.push((tag, offset, size));
        }

        for (index, (tag, offset, size)) in records.iter().enumerate() {
            let record = directory_start + index * 16;
            bytes[record..record + 4].copy_from_slice(tag);
            patch_u32(&mut bytes, record + 8, *offset);
            patch_u32(&mut bytes, record + 12, *size);
        }
        bytes
    }

    fn head_table() -> Vec<u8> {
        let mut bytes = vec![0; 54];
        patch_u16(&mut bytes, 18, 1000);
        patch_i16(&mut bytes, 50, 0);
        bytes
    }

    fn maxp_table() -> Vec<u8> {
        let mut bytes = vec![0; 6];
        patch_u16(&mut bytes, 4, 3);
        bytes
    }

    fn hhea_table() -> Vec<u8> {
        let mut bytes = vec![0; 36];
        patch_i16(&mut bytes, 4, 800);
        patch_i16(&mut bytes, 6, -200);
        patch_i16(&mut bytes, 8, 0);
        patch_u16(&mut bytes, 34, 2);
        bytes
    }

    fn os2_table() -> Vec<u8> {
        let mut bytes = vec![0; 90];
        patch_i16(&mut bytes, 86, 500);
        patch_i16(&mut bytes, 88, 700);
        bytes
    }

    fn hmtx_table() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u16(&mut bytes, 500);
        push_i16(&mut bytes, 0);
        push_u16(&mut bytes, 600);
        push_i16(&mut bytes, 0);
        bytes
    }

    fn loca_table() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 15);
        push_u16(&mut bytes, 15);
        bytes
    }

    fn glyf_table() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_i16(&mut bytes, 1);
        push_i16(&mut bytes, 0);
        push_i16(&mut bytes, 0);
        push_i16(&mut bytes, 1000);
        push_i16(&mut bytes, 1000);
        push_u16(&mut bytes, 2);
        push_u16(&mut bytes, 0);
        bytes.extend_from_slice(&[FLAG_ON_CURVE, FLAG_ON_CURVE, FLAG_ON_CURVE]);
        push_i16(&mut bytes, 0);
        push_i16(&mut bytes, 1000);
        push_i16(&mut bytes, -1000);
        push_i16(&mut bytes, 0);
        push_i16(&mut bytes, 0);
        push_i16(&mut bytes, 1000);
        while bytes.len() % 2 != 0 {
            bytes.push(0);
        }
        assert_eq!(bytes.len(), 30);
        bytes
    }

    fn cmap_table() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 1);
        push_u16(&mut bytes, 3);
        push_u16(&mut bytes, 1);
        push_u32(&mut bytes, 12);

        push_u16(&mut bytes, 4);
        push_u16(&mut bytes, 40);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 6);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 32);
        push_u16(&mut bytes, 65);
        push_u16(&mut bytes, 0xffff);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 32);
        push_u16(&mut bytes, 65);
        push_u16(&mut bytes, 0xffff);
        push_i16(&mut bytes, -30);
        push_i16(&mut bytes, -64);
        push_i16(&mut bytes, 1);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        assert_eq!(bytes.len(), 52);
        bytes
    }

    fn push_u16(bytes: &mut Vec<u8>, value: u16) {
        bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn push_i16(bytes: &mut Vec<u8>, value: i16) {
        bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn patch_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
    }

    fn patch_i16(bytes: &mut [u8], offset: usize, value: i16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
    }

    fn patch_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
    }
}
