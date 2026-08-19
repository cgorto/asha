//! Offline analysis for deterministic render-capture sequences.
//! Metrics cover flicker, stability, oracle error, edges, and temporal lag.

use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("flicker") => flicker(Path::new(args.get(2).expect("flicker <dir>"))),
        Some("stability") => stability(Path::new(args.get(2).expect("stability <dir>"))),
        Some("ghost") => ghost(
            Path::new(args.get(2).expect("ghost <v2> <exact>")),
            Path::new(args.get(3).expect("ghost <v2> <exact>")),
        ),
        Some("diff") => diff(
            Path::new(args.get(2).expect("diff <a> <b>")),
            Path::new(args.get(3).expect("diff <a> <b>")),
        ),
        Some("error") => error(
            Path::new(args.get(2).expect("error <a> <b>")),
            Path::new(args.get(3).expect("error <a> <b>")),
        ),
        Some("edges") => edges(
            Path::new(args.get(2).expect("edges <a> <b>")),
            Path::new(args.get(3).expect("edges <a> <b>")),
        ),
        Some("lag") => lag(
            Path::new(args.get(2).expect("lag <a> <b>")),
            Path::new(args.get(3).expect("lag <a> <b>")),
        ),
        _ => panic!(
            "usage: light_lab_analyze flicker|stability <dir> | ghost|diff|error|edges|lag <a> <b>"
        ),
    }
}

fn frames(dir: &Path) -> Vec<PathBuf> {
    let mut frames: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|entry| {
            let path = entry.expect("dir entry").path();
            (path.extension().is_some_and(|ext| ext == "png")
                && path.file_stem().is_some_and(|stem| {
                    let stem = stem.to_string_lossy();
                    stem.len() == 6
                        && stem.starts_with('f')
                        && stem[1..].bytes().all(|b| b.is_ascii_digit())
                }))
            .then_some(path)
        })
        .collect();
    frames.sort();
    assert!(
        frames.len() >= 2,
        "need at least two frames in {}",
        dir.display()
    );
    frames
}

struct Image {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn load(path: &Path) -> Image {
    let decoder = png::Decoder::new(std::fs::File::open(path).expect("open png"));
    let mut reader = decoder.read_info().expect("read png info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode png");
    buf.truncate(info.buffer_size());
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => {
            let mut rgba = Vec::with_capacity(buf.len() / 3 * 4);
            for px in buf.chunks_exact(3) {
                rgba.extend_from_slice(px);
                rgba.push(255);
            }
            rgba
        }
        other => panic!("expect RGB(A)8 dumps, got {other:?}"),
    };
    Image {
        width: info.width,
        height: info.height,
        rgba,
    }
}

fn save_heatmap(path: &Path, width: u32, height: u32, heat: &[u32], scale: f32) {
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for (i, &h) in heat.iter().enumerate() {
        let v = ((h as f32 * scale).min(1.0) * 255.0) as u8;
        rgba[i * 4] = v;
        rgba[i * 4 + 1] = if h > 0 { 24 } else { 0 };
        rgba[i * 4 + 2] = if h > 0 { 24 } else { 0 };
        rgba[i * 4 + 3] = 255;
    }
    let file = std::fs::File::create(path).expect("create heatmap");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("png header")
        .write_image_data(&rgba)
        .expect("png data");
}

const THRESHOLD: i32 = 6;

fn luma(rgba: &[u8], i: usize) -> i32 {
    (rgba[i * 4] as i32 * 54 + rgba[i * 4 + 1] as i32 * 183 + rgba[i * 4 + 2] as i32 * 19) >> 8
}

fn flicker(dir: &Path) {
    let paths = frames(dir);
    let first = load(&paths[0]);
    let pixels = (first.width * first.height) as usize;
    let mut heat = vec![0u32; pixels];
    let mut prev = first;
    for (index, path) in paths.iter().enumerate().skip(1) {
        let current = load(path);
        assert_eq!((current.width, current.height), (prev.width, prev.height));
        let mut changed = 0u64;
        for i in 0..pixels {
            if (luma(&current.rgba, i) - luma(&prev.rgba, i)).abs() > THRESHOLD {
                heat[i] += 1;
                changed += 1;
            }
        }
        println!(
            "flicker,{index},{changed},{:.4}%",
            changed as f64 / pixels as f64 * 100.0
        );
        prev = current;
    }
    let hot = heat.iter().filter(|&&h| h > 0).count();
    let chronic = heat
        .iter()
        .filter(|&&h| h as usize > paths.len() / 4)
        .count();
    println!(
        "flicker_total,pixels={pixels},hot={hot} ({:.3}%),chronic={chronic} ({:.3}%)",
        hot as f64 / pixels as f64 * 100.0,
        chronic as f64 / pixels as f64 * 100.0,
    );
    let scale = 1.0 / (paths.len() - 1) as f32;
    save_heatmap(
        &dir.join("flicker_heatmap.png"),
        prev.width,
        prev.height,
        &heat,
        scale,
    );
}

fn stability(dir: &Path) {
    let paths = frames(dir);
    let first = load(&paths[0]);
    let pixels = (first.width * first.height) as usize;
    let mut mean = vec![0.0f64; pixels];
    let mut m2 = vec![0.0f64; pixels];
    let mut delta_sum = vec![0.0f64; pixels];
    let mut prev: Vec<i32> = (0..pixels).map(|i| luma(&first.rgba, i)).collect();
    for i in 0..pixels {
        mean[i] = prev[i] as f64;
    }
    let mut count = 1.0f64;
    for path in paths.iter().skip(1) {
        let current = load(path);
        assert_eq!((current.width, current.height), (first.width, first.height));
        count += 1.0;
        for i in 0..pixels {
            let l = luma(&current.rgba, i);
            let d = (l as f64) - mean[i];
            mean[i] += d / count;
            m2[i] += d * ((l as f64) - mean[i]);
            delta_sum[i] += (l - prev[i]).abs() as f64;
            prev[i] = l;
        }
    }
    let frames_n = count;
    let mut stddev: Vec<f32> = m2
        .iter()
        .map(|&m| ((m / (frames_n - 1.0)).max(0.0)).sqrt() as f32)
        .collect();

    let bw = first.width.div_ceil(8);
    let bh = first.height.div_ceil(8);
    let mut shimmer_blocks = 0u32;
    for by in 0..bh {
        for bx in 0..bw {
            let mut acc = 0.0f64;
            let mut n = 0.0f64;
            for y in by * 8..((by + 1) * 8).min(first.height) {
                for x in bx * 8..((bx + 1) * 8).min(first.width) {
                    acc += delta_sum[(y * first.width + x) as usize];
                    n += 1.0;
                }
            }
            if acc / (n * (frames_n - 1.0)) > 0.5 {
                shimmer_blocks += 1;
            }
        }
    }
    let mut sorted = stddev.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
    let mean_std = stddev.iter().map(|&s| s as f64).sum::<f64>() / pixels as f64;
    println!(
        "stability,frames={:.0},mean_stddev={mean_std:.3},p95_stddev={p95:.3},shimmer_blocks={shimmer_blocks} ({:.2}%)",
        frames_n,
        shimmer_blocks as f64 / (bw * bh) as f64 * 100.0
    );
    for s in stddev.iter_mut() {
        *s *= 32.0;
    }
    let heat: Vec<u32> = stddev.iter().map(|&s| s.min(255.0) as u32).collect();
    save_heatmap(
        &dir.join("stability_heatmap.png"),
        first.width,
        first.height,
        &heat,
        1.0 / 255.0,
    );
}

fn ghost(v2: &Path, exact: &Path) {
    let pa = frames(v2);
    let pb = frames(exact);
    let count = pa.len().min(pb.len());
    let probe = load(&pa[0]);
    let pixels = (probe.width * probe.height) as usize;
    let mut dark_heat = vec![0u32; pixels];
    let mut light_heat = vec![0u32; pixels];
    let mut worst = (0usize, 0u64);
    for index in 0..count {
        let ia = load(&pa[index]);
        let ib = load(&pb[index]);
        assert_eq!((ia.width, ia.height), (ib.width, ib.height));
        let mut dark = 0u64;
        let mut light = 0u64;
        for i in 0..pixels {
            let d = luma(&ia.rgba, i) - luma(&ib.rgba, i);
            if d < -THRESHOLD {
                dark += 1;
                dark_heat[i] = dark_heat[i].max((-d) as u32);
            } else if d > THRESHOLD {
                light += 1;
                light_heat[i] = light_heat[i].max(d as u32);
            }
        }
        if dark + light > worst.1 {
            worst = (index, dark + light);
        }
        println!(
            "ghost,{index},dark={dark},light={light},{:.4}%",
            (dark + light) as f64 / pixels as f64 * 100.0
        );
    }
    println!(
        "ghost_worst,frame={},pixels={},{:.4}%",
        worst.0,
        worst.1,
        worst.1 as f64 / pixels as f64 * 100.0
    );
    let mut rgba = vec![0u8; pixels * 4];
    for i in 0..pixels {
        rgba[i * 4] = (dark_heat[i] * 3).min(255) as u8;
        rgba[i * 4 + 1] = (light_heat[i] * 3).min(255) as u8;
        rgba[i * 4 + 3] = 255;
    }
    let file = std::fs::File::create(v2.join("ghost_heatmap.png")).expect("create heatmap");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), probe.width, probe.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("png header")
        .write_image_data(&rgba)
        .expect("png data");
}

fn diff(a: &Path, b: &Path) {
    let pa = frames(a);
    let pb = frames(b);
    let count = pa.len().min(pb.len());
    let mut worst = (0usize, 0u64);
    let probe = load(&pa[0]);
    let pixels = (probe.width * probe.height) as usize;
    let mut heat = vec![0u32; pixels];
    for index in 0..count {
        let ia = load(&pa[index]);
        let ib = load(&pb[index]);
        assert_eq!((ia.width, ia.height), (ib.width, ib.height));
        let mut differing = 0u64;
        for i in 0..pixels {
            let d = (0..3)
                .map(|c| (ia.rgba[i * 4 + c] as i32 - ib.rgba[i * 4 + c] as i32).abs())
                .max()
                .unwrap_or(0);
            if d > THRESHOLD {
                differing += 1;
                heat[i] += 1;
            }
        }
        if differing > worst.1 {
            worst = (index, differing);
        }
        println!(
            "diff,{index},{differing},{:.4}%",
            differing as f64 / pixels as f64 * 100.0
        );
    }
    println!(
        "diff_worst,frame={},pixels={},{:.4}%",
        worst.0,
        worst.1,
        worst.1 as f64 / pixels as f64 * 100.0
    );
    let scale = 1.0 / count as f32;
    save_heatmap(
        &a.join("diff_heatmap.png"),
        probe.width,
        probe.height,
        &heat,
        scale,
    );
}

/// Luma plane of one frame.
fn lumas(img: &Image) -> Vec<i32> {
    (0..(img.width * img.height) as usize)
        .map(|i| luma(&img.rgba, i))
        .collect()
}

fn percentile(hist: &[u64; 256], total: u64, q: f64) -> u32 {
    let target = (total as f64 * q) as u64;
    let mut cumulative = 0u64;
    for (level, &n) in hist.iter().enumerate() {
        cumulative += n;
        if cumulative >= target {
            return level as u32;
        }
    }
    255
}

fn error(a: &Path, b: &Path) {
    let pa = frames(a);
    let pb = frames(b);
    let count = pa.len().min(pb.len());
    let probe = load(&pa[0]);
    let pixels = (probe.width * probe.height) as usize;
    let mut mean = vec![0.0f64; pixels];
    let mut m2 = vec![0.0f64; pixels];
    for index in 0..count {
        let ia = load(&pa[index]);
        let ib = load(&pb[index]);
        assert_eq!((ia.width, ia.height), (ib.width, ib.height));
        let mut hist = [0u64; 256];
        let mut abs_sum = 0.0f64;
        let mut sq_sum = 0.0f64;
        let mut signed_sum = 0.0f64;
        let n = (index + 1) as f64;
        for i in 0..pixels {
            let e = (luma(&ia.rgba, i) - luma(&ib.rgba, i)) as f64;
            abs_sum += e.abs();
            sq_sum += e * e;
            signed_sum += e;
            hist[e.abs() as usize] += 1;
            let d = e - mean[i];
            mean[i] += d / n;
            m2[i] += d * (e - mean[i]);
        }
        println!(
            "error,{index},mean_abs={:.4},rms={:.4},p95={},p99={},bias={:+.4}",
            abs_sum / pixels as f64,
            (sq_sum / pixels as f64).sqrt(),
            percentile(&hist, pixels as u64, 0.95),
            percentile(&hist, pixels as u64, 0.99),
            signed_sum / pixels as f64,
        );
    }
    let frames_n = count as f64;
    let fluct: Vec<f64> = m2
        .iter()
        .map(|&m| ((m / (frames_n - 1.0)).max(0.0)).sqrt())
        .collect();
    let bias_energy: f64 = mean.iter().map(|&b| b * b).sum();
    let fluct_energy: f64 = m2.iter().map(|&m| m / (frames_n - 1.0)).sum();
    let mut sorted_bias: Vec<f64> = mean.iter().map(|&b| b.abs()).collect();
    sorted_bias.sort_by(|x, y| x.total_cmp(y));
    let mut sorted_fluct = fluct.clone();
    sorted_fluct.sort_by(|x, y| x.total_cmp(y));
    let p95 = |sorted: &[f64]| sorted[(sorted.len() as f64 * 0.95) as usize];
    println!(
        "error_total,frames={count},mean_abs_bias={:.4},p95_bias={:.4},mean_fluct={:.4},p95_fluct={:.4},fluct_energy_share={:.2}%",
        mean.iter().map(|&b| b.abs()).sum::<f64>() / pixels as f64,
        p95(&sorted_bias),
        fluct.iter().sum::<f64>() / pixels as f64,
        p95(&sorted_fluct),
        fluct_energy / (bias_energy + fluct_energy).max(1e-9) * 100.0,
    );
    let mut rgba = vec![0u8; pixels * 4];
    for i in 0..pixels {
        let v = (mean[i].abs() * 4.0).min(255.0) as u8;
        if mean[i] < 0.0 {
            rgba[i * 4] = v;
        } else {
            rgba[i * 4 + 1] = v;
        }
        rgba[i * 4 + 3] = 255;
    }
    save_rgba(
        &a.join("bias_heatmap.png"),
        probe.width,
        probe.height,
        &rgba,
    );
    let heat: Vec<u32> = fluct.iter().map(|&f| (f * 8.0).min(255.0) as u32).collect();
    save_heatmap(
        &a.join("fluct_heatmap.png"),
        probe.width,
        probe.height,
        &heat,
        1.0 / 255.0,
    );
}

fn save_rgba(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    let file = std::fs::File::create(path).expect("create heatmap");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("png header")
        .write_image_data(rgba)
        .expect("png data");
}

/// Oracle-edge band radius in pixels: covers the half-res silhouette band
const EDGE_BAND_RADIUS: i32 = 2;
/// A 3x3 oracle luma range above this is an edge (shadow or geometry).
const EDGE_RANGE: i32 = 24;

fn edges(a: &Path, b: &Path) {
    let pa = frames(a);
    let pb = frames(b);
    let count = pa.len().min(pb.len());
    let probe = load(&pa[0]);
    let (w, h) = (probe.width as i32, probe.height as i32);
    let pixels = (w * h) as usize;
    let mut band_err_total = 0.0f64;
    let mut interior_err_total = 0.0f64;
    let mut band_px_total = 0u64;
    let mut interior_heat = vec![0u32; pixels];
    for index in 0..count {
        let ia = load(&pa[index]);
        let ib = load(&pb[index]);
        assert_eq!((ia.width, ia.height), (ib.width, ib.height));
        let lb = lumas(&ib);
        let mut edge = vec![false; pixels];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let mut lo = i32::MAX;
                let mut hi = i32::MIN;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let l = lb[((y + dy) * w + (x + dx)) as usize];
                        lo = lo.min(l);
                        hi = hi.max(l);
                    }
                }
                edge[(y * w + x) as usize] = hi - lo > EDGE_RANGE;
            }
        }
        let mut dilated_x = vec![false; pixels];
        for y in 0..h {
            for x in 0..w {
                let from = (x - EDGE_BAND_RADIUS).max(0);
                let to = (x + EDGE_BAND_RADIUS).min(w - 1);
                dilated_x[(y * w + x) as usize] = (from..=to).any(|sx| edge[(y * w + sx) as usize]);
            }
        }
        let mut band = vec![false; pixels];
        for y in 0..h {
            for x in 0..w {
                let from = (y - EDGE_BAND_RADIUS).max(0);
                let to = (y + EDGE_BAND_RADIUS).min(h - 1);
                band[(y * w + x) as usize] = (from..=to).any(|sy| dilated_x[(sy * w + x) as usize]);
            }
        }
        let mut band_err = 0.0f64;
        let mut interior_err = 0.0f64;
        let mut band_px = 0u64;
        for i in 0..pixels {
            let e = (luma(&ia.rgba, i) - lb[i]).abs();
            if band[i] {
                band_err += e as f64;
                band_px += 1;
            } else {
                interior_err += e as f64;
                interior_heat[i] = interior_heat[i].max(e as u32);
            }
        }
        let total = (band_err + interior_err).max(1e-9);
        println!(
            "edges,{index},band_area={:.2}%,band_err_share={:.2}%,band_err={:.1},interior_err={:.1}",
            band_px as f64 / pixels as f64 * 100.0,
            band_err / total * 100.0,
            band_err,
            interior_err,
        );
        band_err_total += band_err;
        interior_err_total += interior_err;
        band_px_total += band_px;
    }
    let total = (band_err_total + interior_err_total).max(1e-9);
    let band_share = band_err_total / total;
    let area_share = band_px_total as f64 / (pixels as u64 * count as u64) as f64;
    println!(
        "edges_total,band_area={:.2}%,band_err_share={:.2}%,concentration={:.1}x",
        area_share * 100.0,
        band_share * 100.0,
        band_share / area_share.max(1e-9),
    );
    save_heatmap(
        &a.join("interior_heatmap.png"),
        probe.width,
        probe.height,
        &interior_heat,
        3.0 / 255.0,
    );
}

/// Deepest temporal shift tested by `lag`.
const MAX_LAG: usize = 6;

fn lag(a: &Path, b: &Path) {
    let pa = frames(a);
    let pb = frames(b);
    let count = pa.len().min(pb.len());
    assert!(count > MAX_LAG, "need more than {MAX_LAG} frames");
    let probe = load(&pa[0]);
    let pixels = (probe.width * probe.height) as usize;
    let mut window: std::collections::VecDeque<Vec<i32>> = (0..=MAX_LAG)
        .map(|offset| lumas(&load(&pb[MAX_LAG - offset])))
        .collect();
    let mut totals = [0.0f64; MAX_LAG + 1];
    let mut best_hist = [0u64; MAX_LAG + 1];
    for index in MAX_LAG..count {
        if index > MAX_LAG {
            window.pop_back();
            window.push_front(lumas(&load(&pb[index])));
        }
        let la = lumas(&load(&pa[index]));
        let mut per_k = [0.0f64; MAX_LAG + 1];
        for (k, oracle) in window.iter().enumerate() {
            let mut abs_sum = 0.0f64;
            for i in 0..pixels {
                abs_sum += (la[i] - oracle[i]).abs() as f64;
            }
            per_k[k] = abs_sum / pixels as f64;
        }
        let best = (0..=MAX_LAG)
            .min_by(|&x, &y| per_k[x].total_cmp(&per_k[y]))
            .expect("non-empty");
        best_hist[best] += 1;
        totals.iter_mut().zip(per_k).for_each(|(t, e)| *t += e);
        println!(
            "lag,{index},best={best},{}",
            per_k
                .iter()
                .map(|e| format!("{e:.4}"))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    let n = (count - MAX_LAG) as f64;
    println!(
        "lag_total,mean_by_k={},best_hist={}",
        totals
            .iter()
            .map(|t| format!("{:.4}", t / n))
            .collect::<Vec<_>>()
            .join(","),
        best_hist
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
}
