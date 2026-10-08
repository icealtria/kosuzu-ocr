use super::Img;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MinRect {
    pub center: [f64; 2],
    pub w: f64,
    pub h: f64,
    pub u: [f64; 2],
}

fn convex_hull(pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut p: Vec<[f64; 2]> = pts.to_vec();
    p.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });
    p.dedup_by(|a, b| a[0] == b[0] && a[1] == b[1]);
    if p.len() < 3 {
        return p;
    }
    let cross = |o: &[f64; 2], a: &[f64; 2], b: &[f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut lower = Vec::new();
    for q in &p {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], q) <= 0.0 {
            lower.pop();
        }
        lower.push(*q);
    }
    let mut upper = Vec::new();
    for q in p.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], q) <= 0.0 {
            upper.pop();
        }
        upper.push(*q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

pub(crate) fn min_area_rect(pts: &[[f64; 2]]) -> MinRect {
    let hull = convex_hull(pts);
    debug_assert!(hull.len() >= 3, "min_area_rect: degenerate point set");
    let n = hull.len();
    let mut best: Option<(f64, [f64; 2], f64, f64, f64, f64)> = None;
    for i in 0..n {
        let (a, b) = (hull[i], hull[(i + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            continue;
        }
        let u = [dx / len, dy / len];
        let v = [-u[1], u[0]];
        let mut u_min = f64::INFINITY;
        let mut u_max = f64::NEG_INFINITY;
        let mut v_min = f64::INFINITY;
        let mut v_max = f64::NEG_INFINITY;
        for p in &hull {
            let pu = p[0] * u[0] + p[1] * u[1];
            let pv = p[0] * v[0] + p[1] * v[1];
            u_min = u_min.min(pu);
            u_max = u_max.max(pu);
            v_min = v_min.min(pv);
            v_max = v_max.max(pv);
        }
        let area = (u_max - u_min) * (v_max - v_min);
        if best.map(|(a, ..)| area < a - 1e-9).unwrap_or(true) {
            best = Some((area, u, u_min, u_max, v_min, v_max));
        }
    }
    let (area, u, u_min, u_max, v_min, v_max) = best.expect("min_area_rect: no valid edge");
    debug_assert!(area.is_finite());
    let v = [-u[1], u[0]];
    let cu = (u_min + u_max) / 2.0;
    let cv = (v_min + v_max) / 2.0;
    MinRect {
        center: [cu * u[0] + cv * v[0], cu * u[1] + cv * v[1]],
        w: u_max - u_min,
        h: v_max - v_min,
        u,
    }
}

pub(crate) fn box_points(r: &MinRect) -> [[f32; 2]; 4] {
    let v = [-r.u[1], r.u[0]];
    let (hw, hh) = (r.w / 2.0, r.h / 2.0);
    let corners: [[f64; 2]; 4] = [
        [r.center[0] - hw * r.u[0] - hh * v[0], r.center[1] - hw * r.u[1] - hh * v[1]],
        [r.center[0] + hw * r.u[0] - hh * v[0], r.center[1] + hw * r.u[1] - hh * v[1]],
        [r.center[0] + hw * r.u[0] + hh * v[0], r.center[1] + hw * r.u[1] + hh * v[1]],
        [r.center[0] - hw * r.u[0] + hh * v[0], r.center[1] - hw * r.u[1] + hh * v[1]],
    ];
    corners.map(|c| [c[0] as f32, c[1] as f32])
}

pub(crate) fn mini_boxes(pts: &[[f64; 2]]) -> ([[f32; 2]; 4], f32) {
    let r = min_area_rect(pts);
    let sside = r.w.min(r.h) as f32;
    let mut p = box_points(&r);
    p.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    let (i1, i4) = if p[1][1] > p[0][1] { (0usize, 1usize) } else { (1, 0) };
    let (i2, i3) = if p[3][1] > p[2][1] { (2usize, 3usize) } else { (3, 2) };
    ([p[i1], p[i2], p[i3], p[i4]], sside)
}

fn perspective_from(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> [[f64; 3]; 3] {
    let mut a = [[0f64; 8]; 8];
    let mut b = [0f64; 8];
    for i in 0..4 {
        let (x, y) = (src[i][0], src[i][1]);
        let (u, v) = (dst[i][0], dst[i][1]);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u];
        b[2 * i] = u;
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v];
        b[2 * i + 1] = v;
    }
    for col in 0..8 {
        let mut piv = col;
        for r in col + 1..8 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        debug_assert!(d.abs() > 1e-12, "perspective solve singular");
        for r in col + 1..8 {
            let f = a[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..8 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0f64; 8];
    for r in (0..8).rev() {
        let mut s = b[r];
        for c in r + 1..8 {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }
    [[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], 1.0]]
}

fn apply3(m: &[[f64; 3]; 3], x: f64, y: f64) -> (f64, f64) {
    let w = m[2][0] * x + m[2][1] * y + m[2][2];
    (
        (m[0][0] * x + m[0][1] * y + m[0][2]) / w,
        (m[1][0] * x + m[1][1] * y + m[1][2]) / w,
    )
}

fn invert3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    debug_assert!(det.abs() > 1e-12);
    let inv = |a: f64| a / det;
    [
        [
            inv(m[1][1] * m[2][2] - m[1][2] * m[2][1]),
            inv(m[0][2] * m[2][1] - m[0][1] * m[2][2]),
            inv(m[0][1] * m[1][2] - m[0][2] * m[1][1]),
        ],
        [
            inv(m[1][2] * m[2][0] - m[1][0] * m[2][2]),
            inv(m[0][0] * m[2][2] - m[0][2] * m[2][0]),
            inv(m[0][2] * m[1][0] - m[0][0] * m[1][2]),
        ],
        [
            inv(m[1][0] * m[2][1] - m[1][1] * m[2][0]),
            inv(m[0][1] * m[2][0] - m[0][0] * m[2][1]),
            inv(m[0][0] * m[1][1] - m[0][1] * m[1][0]),
        ],
    ]
}

fn cubic(t: f64) -> f64 {
    const A: f64 = -0.75;
    let t = t.abs();
    if t <= 1.0 {
        ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0
    } else if t < 2.0 {
        ((A * t - 5.0 * A) * t + 8.0 * A) * t - 4.0 * A
    } else {
        0.0
    }
}

fn cubic_sample(img: &Img, x: f64, y: f64, c: usize) -> u8 {
    let (w, h) = (img.w as i64, img.h as i64);
    let fx = x.floor();
    let fy = y.floor();
    let mut acc = 0.0;
    let mut wsum = 0.0;
    for dy in -1i64..=2 {
        let cy = fy + dy as f64;
        let sy = (cy as i64).clamp(0, h - 1) as u32;
        let wy = cubic(y - cy);
        if wy == 0.0 {
            continue;
        }
        for dx in -1i64..=2 {
            let cx = fx + dx as f64;
            let sx = (cx as i64).clamp(0, w - 1) as u32;
            let wx = cubic(x - cx);
            if wx == 0.0 {
                continue;
            }
            let wgt = wx * wy;
            acc += wgt * img.data[((sy * img.w + sx) * 3) as usize + c] as f64;
            wsum += wgt;
        }
    }
    let v = if wsum.abs() > 1e-12 { acc / wsum } else { 0.0 };
    (v + 0.5).floor().clamp(0.0, 255.0) as u8
}

pub(crate) fn crop_rotated(bgr: &Img, quad: &[[i32; 2]; 4]) -> Option<Img> {
    let pts_f: [[f64; 2]; 4] = quad.map(|p| [p[0] as f64, p[1] as f64]);
    let (ordered, _) = mini_boxes(&pts_f);

    let norm = |a: [f32; 2], b: [f32; 2]| {
        let (dx, dy) = ((a[0] - b[0]) as f64, (a[1] - b[1]) as f64);
        (dx * dx + dy * dy).sqrt()
    };
    let cw = norm(ordered[0], ordered[1])
        .max(norm(ordered[2], ordered[3]))
        .floor() as i64;
    let ch = norm(ordered[0], ordered[3])
        .max(norm(ordered[1], ordered[2]))
        .floor() as i64;
    if cw <= 0 || ch <= 0 {
        return None;
    }
    let (cw, ch) = (cw as u32, ch as u32);

    let src: [[f64; 2]; 4] = ordered.map(|p| [p[0] as f64, p[1] as f64]);
    let dst = [
        [0.0, 0.0],
        [cw as f64, 0.0],
        [cw as f64, ch as f64],
        [0.0, ch as f64],
    ];
    let m = perspective_from(&src, &dst);
    let minv = invert3(&m);

    let mut out = vec![0u8; (cw * ch * 3) as usize];
    for y in 0..ch {
        for x in 0..cw {
            let (sx, sy) = apply3(&minv, x as f64, y as f64);
            let o = ((y * cw + x) * 3) as usize;
            for c in 0..3 {
                out[o + c] = cubic_sample(bgr, sx, sy, c);
            }
        }
    }
    let mut img = Img { data: out, w: cw, h: ch };

    if img.h as f64 / img.w as f64 >= 1.5 {
        let (iw, ih) = (img.w, img.h);
        let mut t = vec![0u8; img.data.len()];
        for y in 0..iw {
            for x in 0..ih {
                let src_idx = (x as usize * iw as usize + (iw as usize - 1 - y as usize)) * 3;
                let dst_idx = (y as usize * ih as usize + x as usize) * 3;
                t[dst_idx..dst_idx + 3].copy_from_slice(&img.data[src_idx..src_idx + 3]);
            }
        }
        img = Img { data: t, w: ih, h: iw };
    }
    Some(img)
}
