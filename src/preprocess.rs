use super::Img;

pub(crate) fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

pub(crate) fn resize_bilinear(src: &Img, nw: u32, nh: u32) -> Img {
    if nw == src.w && nh == src.h {
        return Img { data: src.data.clone(), w: src.w, h: src.h };
    }
    let (sw, sh) = (src.w as i64, src.h as i64);

    let tables = |dst_len: u32, src_len: i64| -> (Vec<i32>, Vec<i32>, Vec<i32>, usize) {
        let scale = 1.0f64 / (dst_len as f64 / src_len as f64);
        let mut sx = vec![0i32; dst_len as usize];
        let mut a0 = vec![0i32; dst_len as usize];
        let mut a1 = vec![0i32; dst_len as usize];
        let mut xmax = dst_len as i32;
        for d in 0..dst_len as i32 {
            let mut fx = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let mut s = fx.floor() as i32;
            fx -= s as f32;
            if s < 0 {
                fx = 0.0;
                s = 0;
            }
            if s + 1 >= src_len as i32 {
                xmax = xmax.min(d);
                fx = 0.0;
                s = src_len as i32 - 1;
            }
            sx[d as usize] = s;
            a0[d as usize] = round_cv((1.0f32 - fx) * 2048.0);
            a1[d as usize] = round_cv(fx * 2048.0);
        }
        (sx, a0, a1, xmax as usize)
    };
    let (xsx, xa0, xa1, xmax) = tables(nw, sw);
    let (ysy, ya0, ya1, _ymax) = tables(nh, sh);

    let row_len = (nw * 3) as usize;
    let mut hbuf = vec![0i32; sh as usize * row_len];
    for y in 0..sh as usize {
        let src_row = &src.data[y * src.w as usize * 3..][..src.w as usize * 3];
        let out = &mut hbuf[y * row_len..(y + 1) * row_len];
        for d in 0..nw as usize {
            let (s, w0, w1) = (xsx[d], xa0[d], xa1[d]);
            let base = d * 3;
            if d < xmax {
                let s = (s as usize) * 3;
                for c in 0..3 {
                    out[base + c] =
                        src_row[s + c] as i32 * w0 + src_row[s + c + 3] as i32 * w1;
                }
            } else {
                let s = (s as usize) * 3;
                for c in 0..3 {
                    out[base + c] = src_row[s + c] as i32 * 2048;
                }
            }
        }
    }

    let mut out = vec![0u8; (nw * nh * 3) as usize];
    for d in 0..nh as usize {
        let (sy, b0, b1) = (ysy[d] as usize, ya0[d], ya1[d]);
        let sy1 = (sy + 1).min(sh as usize - 1);
        let r0 = &hbuf[sy * row_len..(sy + 1) * row_len];
        let r1 = &hbuf[sy1 * row_len..(sy1 + 1) * row_len];
        let o = &mut out[d * row_len..(d + 1) * row_len];
        for i in 0..row_len {
            let v = ((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16);
            o[i] = ((v + 2) >> 2) as u8;
        }
    }
    Img { data: out, w: nw, h: nh }
}

fn round_cv(v: f32) -> i32 {
    v.round_ties_even() as i32
}

pub(crate) struct DetInput {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
    pub src_w: usize,
    pub src_h: usize,
}

const LIMIT_SIDE: f64 = 64.0;
const MAX_SIDE_LIMIT: usize = 4000;

fn det_norm(bgr: &Img) -> Vec<f32> {
    const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
    const STD: [f64; 3] = [0.229, 0.224, 0.225];
    let alpha: [f32; 3] = std::array::from_fn(|i| ((1.0f64 / 255.0) / STD[i]) as f32);
    let beta: [f32; 3] = std::array::from_fn(|i| (-MEAN[i] / STD[i]) as f32);
    let (w, h) = (bgr.w as usize, bgr.h as usize);
    let mut out = vec![0f32; 3 * w * h];
    for c in 0..3 {
        let base = c * w * h;
        for i in 0..w * h {
            out[base + i] = (bgr.data[i * 3 + c] as f32) * alpha[c] + beta[c];
        }
    }
    out
}

pub(crate) fn det_preprocess(bgr: &Img) -> DetInput {
    let (src_w, src_h) = (bgr.w as usize, bgr.h as usize);
    debug_assert!(src_w + src_h >= 64, "tiny-image pad path not implemented");

    let ratio = {
        let min_side = src_h.min(src_w) as f64;
        if min_side < LIMIT_SIDE {
            LIMIT_SIDE / min_side
        } else {
            1.0
        }
    };
    let mut rh = (src_h as f64 * ratio) as i64;
    let mut rw = (src_w as f64 * ratio) as i64;
    if rh.max(rw) > MAX_SIDE_LIMIT as i64 {
        let r = MAX_SIDE_LIMIT as f64 / rh.max(rw) as f64;
        rh = (rh as f64 * r) as i64;
        rw = (rw as f64 * r) as i64;
    }
    let rh = (py_round(rh as f64 / 32.0) * 32).max(32) as usize;
    let rw = (py_round(rw as f64 / 32.0) * 32).max(32) as usize;

    let resized = if rh == src_h && rw == src_w {
        bgr.clone()
    } else {
        resize_bilinear(bgr, rw as u32, rh as u32)
    };
    DetInput {
        data: det_norm(&resized),
        w: rw,
        h: rh,
        src_w,
        src_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_f32(p: &Path) -> Vec<f32> {
        let bytes = std::fs::read(p).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn det_preprocess_matches_fixture() {
        let Some(fx) = crate::fixture_dir() else {
            eprintln!("SKIPPED: private OCR fixture bundle absent (fixtures/shot.png)");
            return;
        };
        let img = image::open(fx.join("shot.png")).unwrap().to_rgb8();
        let bgr = crate::to_bgr(&img);
        let det_in = det_preprocess(&bgr);
        assert_eq!((det_in.w, det_in.h), (1920, 1088), "size must be the 32-aligned resize");
        assert_eq!((det_in.src_w, det_in.src_h), (1920, 1080));

        let want = load_f32(&fx.join("det_x.bin"));
        assert_eq!(want.len(), det_in.data.len(), "det_x element count");

        const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
        const STD: [f64; 3] = [0.229, 0.224, 0.225];
        let alpha: [f32; 3] = std::array::from_fn(|i| ((1.0f64 / 255.0) / STD[i]) as f32);
        let beta: [f32; 3] = std::array::from_fn(|i| (-MEAN[i] / STD[i]) as f32);

        let (w, h) = (det_in.w, det_in.h);
        let mut max_d = 0i32;
        let mut nz = 0usize;
        for c in 0..3 {
            let base = c * w * h;
            for i in 0..w * h {
                let a = ((want[base + i] - beta[c]) / alpha[c]).round() as i32;
                let b = ((det_in.data[base + i] - beta[c]) / alpha[c]).round() as i32;
                let d = (a - b).abs();
                max_d = max_d.max(d);
                if d > 0 {
                    nz += 1;
                }
            }
        }
        println!(
            "det preprocess vs det_x: gray max_diff={}, nonzero {} / {} ({:.2}%)",
            max_d,
            nz,
            want.len(),
            nz as f64 * 100.0 / want.len() as f64
        );
        assert_eq!(nz, 0, "must match cv2.INTER_LINEAR bit-exact, {} nonzero diffs", nz);
    }

    #[test]
    fn py_round_half_even() {
        assert_eq!(py_round(33.75), 34);
        assert_eq!(py_round(32.5), 32);
        assert_eq!(py_round(33.5), 34);
        assert_eq!(py_round(60.0), 60);
        assert_eq!(py_round(0.5), 0);
        assert_eq!(py_round(1.5), 2);
        assert_eq!(py_round(-2.5), -2);
        assert_eq!(py_round(-3.5), -4);
        assert_eq!(py_round(-0.5), 0);
        assert_eq!(py_round(-1.5), -2);
        assert_eq!(py_round(-2.4), -2);
        assert_eq!(py_round(-2.6), -3);
    }
}
