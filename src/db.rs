use imageproc::contours::find_contours;
use imageproc::image::{GrayImage, Luma};
use imageproc::point::Point;

use super::geom::mini_boxes;
use super::preprocess::py_round;

const THRESH: f64 = 0.3;
const BOX_THRESH: f64 = 0.6;
const UNCLIP_RATIO: f64 = 1.5;
const MAX_CANDIDATES: usize = 3000;
const MIN_SIZE: f64 = 3.0;

fn contour_area(pts: &[[f32; 2]]) -> f64 {
    let n = pts.len();
    let mut s = 0.0f64;
    for i in 0..n {
        let (x0, y0) = (pts[i][0] as f64, pts[i][1] as f64);
        let (x1, y1) = (
            pts[(i + 1) % n][0] as f64,
            pts[(i + 1) % n][1] as f64,
        );
        s += x0 * y1 - x1 * y0;
    }
    (s / 2.0).abs()
}

fn arc_length(pts: &[[f32; 2]]) -> f64 {
    let n = pts.len();
    if n < 2 {
        return 0.0;
    }
    (0..n)
        .map(|i| {
            let p1 = pts[i];
            let p2 = pts[(i + 1) % n];
            let dx = (p2[0] - p1[0]) as f64;
            let dy = (p2[1] - p1[1]) as f64;
            (dx * dx + dy * dy).sqrt()
        })
        .sum()
}

fn offset_round(pts: &[[f32; 2]], d: f64) -> Vec<[f64; 2]> {
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;

    fn r64(v: f64) -> i64 {
        if v < 0.0 {
            (v - 0.5) as i64
        } else {
            (v + 0.5) as i64
        }
    }

    let mut src: Vec<[i64; 2]> = pts
        .iter()
        .map(|p| [p[0] as i64, p[1] as i64])
        .collect();
    while src.len() > 1 && src[0] == src[src.len() - 1] {
        src.pop();
    }
    let mut dedup: Vec<[i64; 2]> = Vec::with_capacity(src.len());
    for p in src {
        if dedup.last() != Some(&p) {
            dedup.push(p);
        }
    }
    src = dedup;
    if src.len() < 3 {
        return Vec::new();
    }

    let n = src.len();
    let mut a_sum = 0.0f64;
    {
        let mut j = n - 1;
        for i in 0..n {
            a_sum += (src[j][0] as f64 + src[i][0] as f64) * (src[j][1] as f64 - src[i][1] as f64);
            j = i;
        }
    }
    if -a_sum * 0.5 < 0.0 {
        src.reverse();
    }

    if d.abs() < 1e-20 {
        return src.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    }

    let ad = d.abs();
    let y = if 0.25 > ad * 0.25 { ad * 0.25 } else { 0.25 };
    let mut steps_total = PI / (1.0 - y / ad).acos();
    if steps_total > ad * PI {
        steps_total = ad * PI;
    }
    let mut m_sin = (TAU / steps_total).sin();
    let m_cos = (TAU / steps_total).cos();
    if d < 0.0 {
        m_sin = -m_sin;
    }
    let steps_per_rad = steps_total / TAU;

    fn unit_normal(p1: [i64; 2], p2: [i64; 2]) -> [f64; 2] {
        if p1[0] == p2[0] && p1[1] == p2[1] {
            return [0.0, 0.0];
        }
        let dx = (p2[0] - p1[0]) as f64;
        let dy = (p2[1] - p1[1]) as f64;
        let f = 1.0 / (dx * dx + dy * dy).sqrt();
        [dy * f, -dx * f]
    }
    let mut normals = Vec::with_capacity(n);
    for i in 0..n - 1 {
        normals.push(unit_normal(src[i], src[i + 1]));
    }
    normals.push(unit_normal(src[n - 1], src[0]));

    let mut out: Vec<[i64; 2]> = Vec::with_capacity(n * 4);
    let mut k = n - 1;
    for j in 0..n {
        // OffsetPoint(j, k, jtRound)
        let mut sin_a = normals[k][0] * normals[j][1] - normals[j][0] * normals[k][1];
        if (sin_a * d).abs() < 1.0 {
            let cos_a = normals[k][0] * normals[j][0] + normals[j][1] * normals[k][1];
            if cos_a > 0.0 {
                out.push([
                    r64(src[j][0] as f64 + normals[k][0] * d),
                    r64(src[j][1] as f64 + normals[k][1] * d),
                ]);
                continue;
            }
        } else {
            sin_a = sin_a.clamp(-1.0, 1.0);
        }

        if sin_a * d < 0.0 {
            out.push([
                r64(src[j][0] as f64 + normals[k][0] * d),
                r64(src[j][1] as f64 + normals[k][1] * d),
            ]);
            out.push(src[j]);
            out.push([
                r64(src[j][0] as f64 + normals[j][0] * d),
                r64(src[j][1] as f64 + normals[j][1] * d),
            ]);
        } else {
            // DoRound
            let a = sin_a.atan2(
                normals[k][0] * normals[j][0] + normals[k][1] * normals[j][1],
            );
            let steps = r64(steps_per_rad * a.abs()).max(1) as usize;
            let mut x = normals[k][0];
            let mut yy = normals[k][1];
            for _ in 0..steps {
                out.push([
                    r64(src[j][0] as f64 + x * d),
                    r64(src[j][1] as f64 + yy * d),
                ]);
                let x2 = x;
                x = x * m_cos - m_sin * yy;
                yy = x2 * m_sin + yy * m_cos;
            }
            out.push([
                r64(src[j][0] as f64 + normals[j][0] * d),
                r64(src[j][1] as f64 + normals[j][1] * d),
            ]);
        }
        k = j;
    }
    out.iter().map(|p| [p[0] as f64, p[1] as f64]).collect()
}

fn unclip(pts: &[[f32; 2]], ratio: f64) -> Vec<[f64; 2]> {
    let area = contour_area(pts);
    let length = arc_length(pts);
    if length <= 0.0 {
        return pts.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    }
    let d = area * ratio / length;
    offset_round(pts, d)
}

fn box_score_fast(pred: &[f32], bw: usize, bh: usize, pts: &[[f32; 2]]) -> f64 {
    let xmin_f = pts.iter().map(|p| p[0] as f64).fold(f64::INFINITY, f64::min).floor();
    let xmax_f = pts.iter().map(|p| p[0] as f64).fold(f64::NEG_INFINITY, f64::max).ceil();
    let ymin_f = pts.iter().map(|p| p[1] as f64).fold(f64::INFINITY, f64::min).floor();
    let ymax_f = pts.iter().map(|p| p[1] as f64).fold(f64::NEG_INFINITY, f64::max).ceil();
    let xmin = (xmin_f.max(0.0) as usize).min(bw - 1);
    let xmax = (xmax_f.max(0.0) as usize).min(bw - 1);
    let ymin = (ymin_f.max(0.0) as usize).min(bh - 1);
    let ymax = (ymax_f.max(0.0) as usize).min(bh - 1);
    if xmax < xmin || ymax < ymin {
        return 0.0;
    }
    let poly: Vec<[i32; 2]> = pts
        .iter()
        .map(|p| {
            [
                (p[0] - xmin as f32) as i32,
                (p[1] - ymin as f32) as i32,
            ]
        })
        .collect();
    let (w, h) = (xmax - xmin + 1, ymax - ymin + 1);
    let mask = fill_poly_exact(w, h, &poly);
    let (mut sum, mut cnt) = (0f64, 0u64);
    for y in 0..h {
        let row = ymin + y;
        for x in 0..w {
            if mask[y * w + x] != 0 {
                sum += pred[row * bw + (xmin + x)] as f64;
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        0.0
    } else {
        sum / cnt as f64
    }
}

fn fill_poly_exact(w: usize, h: usize, poly: &[[i32; 2]]) -> Vec<u8> {
    let mut mask = vec![0u8; w * h];
    let n = poly.len();
    if n < 3 {
        return mask;
    }
    for i in 0..n {
        let a = poly[(i + n - 1) % n];
        line8(a[0], a[1], poly[i][0], poly[i][1], w, h, &mut mask);
    }
    struct Edge {
        y0: i32,
        y1: i32,
        xs: i64,
        dx: i64,
    }
    let mut edges: Vec<Edge> = Vec::with_capacity(n);
    let mut prev = poly[n - 1];
    for &cur in poly {
        if prev[1] != cur[1] {
            let x0 = ((prev[0] as i64) << 16) + 32768;
            let x1 = ((cur[0] as i64) << 16) + 32768;
            let dx = (x1 - x0) / (cur[1] as i64 - prev[1] as i64);
            edges.push(if prev[1] < cur[1] {
                Edge { y0: prev[1], y1: cur[1], xs: x0, dx }
            } else {
                Edge { y0: cur[1], y1: prev[1], xs: x1, dx }
            });
        }
        prev = cur;
    }
    if edges.is_empty() {
        return mask;
    }
    let y_min = edges.iter().map(|e| e.y0).min().unwrap().max(0);
    let y_max = edges.iter().map(|e| e.y1).max().unwrap().min(h as i32);
    for y in y_min..y_max {
        let mut active: Vec<i64> = edges
            .iter()
            .filter(|e| e.y0 <= y && y < e.y1)
            .map(|e| e.xs + (y - e.y0) as i64 * e.dx)
            .collect();
        active.sort_unstable_by_key(|x| *x);
        let mut k = 0;
        while k + 1 < active.len() {
            let mut x1 = (active[k] >> 16) as i32;
            let mut x2 = (active[k + 1] >> 16) as i32;
            if x1 < 0 {
                x1 = 0;
            }
            if x2 >= w as i32 {
                x2 = w as i32 - 1;
            }
            if x1 <= x2 {
                let row = y as usize * w;
                for x in x1..=x2 {
                    mask[row + x as usize] = 1;
                }
            }
            k += 2;
        }
    }
    mask
}

fn line8(x1_: i32, y1_: i32, x2_: i32, y2_: i32, w: usize, h: usize, mask: &mut [u8]) {
    let (mut delta_x, mut delta_y) = (1i32, 1i32);
    let (mut dx, mut dy) = (x2_ - x1_, y2_ - y1_);
    let (mut x1, mut y1) = (x1_, y1_);
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        x1 = x2_;
        y1 = y2_;
    }
    if dy < 0 {
        dy = -dy;
        delta_y = -1;
    }
    let vert = dy > dx;
    if vert {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut delta_x, &mut delta_y);
    }
    let mut err = dx - 2 * dy;
    let plus_delta = 2 * dx;
    let minus_delta = -2 * dy;
    let (mut minus_shift, mut plus_shift) = (delta_x, 0);
    let (mut minus_step, mut plus_step) = (0, delta_y);
    let count = dx + 1;
    if vert {
        std::mem::swap(&mut plus_step, &mut plus_shift);
        std::mem::swap(&mut minus_step, &mut minus_shift);
    }
    let (mut x, mut y) = (x1, y1);
    for _ in 0..count {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            mask[y as usize * w + x as usize] = 1;
        }
        let m = if err < 0 { -1 } else { 0 };
        err += minus_delta + (plus_delta & m);
        x += minus_shift + (plus_shift & m);
        y += minus_step + (plus_step & m);
    }
}

pub(crate) fn detect(pred: &[f32], bw: usize, bh: usize, src_w: usize, src_h: usize) -> Vec<[[i32; 2]; 4]> {
    debug_assert_eq!(pred.len(), bw * bh);
    let mut gray = GrayImage::new(bw as u32, bh as u32);
    for y in 0..bh {
        for x in 0..bw {
            let v = if pred[y * bw + x] as f64 > THRESH { 255 } else { 0 };
            gray.put_pixel(x as u32, y as u32, Luma([v]));
        }
    }
    let contours: Vec<imageproc::contours::Contour<i32>> = find_contours(&gray);
    let width_scale = src_w as f64 / bw as f64;
    let height_scale = src_h as f64 / bh as f64;

    let mut boxes = Vec::new();
    for contour in contours.iter().take(MAX_CANDIDATES) {
        let pts: Vec<[f64; 2]> = contour.points.iter().map(|p: &Point<i32>| [p.x as f64, p.y as f64]).collect();
        if pts.len() < 3 {
            continue;
        }
        let (mini, sside) = mini_boxes(&pts);
        if (sside as f64) < MIN_SIZE {
            continue;
        }
        let score = box_score_fast(pred, bw, bh, &mini);
        if BOX_THRESH > score {
            continue;
        }
        let expanded = unclip(&mini, UNCLIP_RATIO);
        if expanded.len() < 3 {
            continue;
        }
        let (mut box4, sside2) = mini_boxes(&expanded);
        if (sside2 as f64) < MIN_SIZE + 2.0 {
            continue;
        }
        for p in box4.iter_mut() {
            let x = py_round(p[0] as f64 * width_scale).clamp(0, src_w as i64);
            let y = py_round(p[1] as f64 * height_scale).clamp(0, src_h as i64);
            p[0] = x as f32;
            p[1] = y as f32;
        }
        boxes.push(box4.map(|p| [p[0] as i32, p[1] as i32]));
    }
    boxes
}

pub(crate) fn sort_quad_boxes(boxes: &mut [[[i32; 2]; 4]]) {
    boxes.sort_by_key(|b| (b[0][1], b[0][0]));
    let n = boxes.len();
    for i in 0..n.saturating_sub(1) {
        for j in (0..=i).rev() {
            let near = (boxes[j + 1][0][1] - boxes[j][0][1]).abs() < 10;
            if near && boxes[j + 1][0][0] < boxes[j][0][0] {
                boxes.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_poly_matches_cv2_reference() {
        let poly: Vec<[i32; 2]> = [(0i32, 1), (4, 0), (6, 16), (1, 17)]
            .iter()
            .map(|&(x, y)| [x, y])
            .collect();
        let (w, h) = (8, 19);
        let mask = fill_poly_exact(w, h, &poly);
        let want = [
            "...##...",
            "#####...",
            "#####...",
            "#####...",
            "######..",
            "######..",
            "######..",
            "######..",
            "######..",
            "######..",
            ".#####..",
            ".#####..",
            ".######.",
            ".######.",
            ".######.",
            ".######.",
            ".######.",
            ".###....",
            "........",
        ];
        for (y, row) in want.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                let m = mask[y * w + x];
                assert_eq!(
                    m,
                    if c == '#' { 1 } else { 0 },
                    "mask({},{}) want {} got {}",
                    x,
                    y,
                    c,
                    m
                );
            }
        }
    }
}
