use super::preprocess::resize_bilinear;
use super::Img;

const REC_H: usize = 48;
const REC_BASE_W: usize = 320;
const MAX_IMGW: usize = 3200;

pub(crate) fn rec_input(crop: &Img) -> (Vec<f32>, usize) {
    let (w, h) = (crop.w as usize, crop.h as usize);
    debug_assert!(w > 0 && h > 0, "empty crop");
    let ratio = w as f64 / h as f64;
    let max_wh_ratio = (REC_BASE_W as f64 / REC_H as f64).max(ratio);
    let mut imgw = (REC_H as f64 * max_wh_ratio) as usize;
    let resized_w = if imgw > MAX_IMGW {
        imgw = MAX_IMGW;
        MAX_IMGW
    } else {
        let ceil_w = (REC_H as f64 * ratio).ceil() as usize;
        if ceil_w > imgw {
            imgw
        } else {
            ceil_w
        }
    };

    let resized = resize_bilinear(crop, resized_w as u32, REC_H as u32);
    let mut out = vec![0f32; 3 * REC_H * imgw];
    let plane = REC_H * imgw;
    for c in 0..3 {
        for y in 0..REC_H {
            for x in 0..resized_w {
                let v = resized.data[(y * resized_w + x) * 3 + c] as f32 / 255.0;
                out[c * plane + y * imgw + x] = (v - 0.5) / 0.5;
            }
        }
    }
    (out, imgw)
}

pub(crate) fn ctc_decode(logits: &[f32], time_steps: usize, classes: usize, dict: &[&str]) -> (String, f32) {
    debug_assert!(classes >= 1);
    let mut text = String::new();
    let mut score_sum = 0.0;
    let mut count = 0;
    let mut prev: Option<usize> = None;

    for t in 0..time_steps {
        let base = t * classes;
        let mut best = 0usize;
        let mut best_v = logits[base];
        for k in 1..classes {
            let v = logits[base + k];
            if v > best_v {
                best_v = v;
                best = k;
            }
        }

        let dup = prev == Some(best);
        prev = Some(best);
        if dup || best == 0 {
            continue;
        }
        if best <= dict.len() {
            text.push_str(dict[best - 1]);
        } else if best == dict.len() + 1 {
            text.push(' ');
        } else {
            continue;
        }
        score_sum += best_v;
        count += 1;
    }
    (text, if count > 0 { score_sum / count as f32 } else { 0.0 })
}
