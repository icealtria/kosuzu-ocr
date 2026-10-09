mod db;
mod geom;
mod models;
mod preprocess;
mod rec;

pub use models::{ensure_preset, is_downloaded, normalize_preset};

use std::path::Path;
use std::sync::Once;

use anyhow::{Context, Result};
use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

const CHARSET_TINY: &str = include_str!("../fixtures/charset.txt");
const CHARSET_EXT: &str = include_str!("../fixtures/charset18k.txt");

#[derive(Debug, Clone, serde::Serialize)]
pub struct OcrLine {
    pub poly: [[i32; 2]; 4],
    pub text: String,
    pub score: f32,
}

#[derive(Clone)]
pub(crate) struct Img {
    pub data: Vec<u8>,
    pub w: u32,
    pub h: u32,
}

pub struct OcrEngine {
    det: Session,
    rec: Session,
    dict: Vec<&'static str>,
}

static ORT_ONCE: Once = Once::new();

/// The OCR parity fixtures are private: everything in `fixtures/` except the two
/// charset files is derived from a confidential screenshot. Tests that need the
/// bundle skip loudly when it is absent (fresh clone / published package).
#[cfg(test)]
pub(crate) fn fixture_dir() -> Option<std::path::PathBuf> {
    let fx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    fx.join("shot.png").is_file().then_some(fx)
}

fn ensure_ort() {
    ORT_ONCE.call_once(|| {
        ort::init().commit();
    });
}

pub(crate) fn to_bgr(image: &RgbImage) -> Img {
    let mut data = Vec::with_capacity(image.pixels().count() * 3);
    for p in image.pixels() {
        let [r, g, b] = p.0;
        data.extend_from_slice(&[b, g, r]);
    }
    let (w, h) = image.dimensions();
    Img { data, w, h }
}

pub fn models_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("KOSUZU_MODELS") {
        return dir.into();
    }
    let data = std::env::var_os("XDG_DATA_HOME")
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join(".local/share")
        });
    data.join("kosuzu").join("models")
}

impl OcrEngine {
    pub fn from_dir(dir: &Path, preset: &str) -> Result<Self> {
        ensure_ort();
        let det_path = models::det_path(dir, preset);
        let rec_path = models::rec_path(dir, preset);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8);
        let det = Session::builder()
            .context("ort session builder")?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(&det_path)
            .with_context(|| format!("load {}", det_path.display()))?;
        let rec = Session::builder()
            .context("ort session builder")?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(&rec_path)
            .with_context(|| format!("load {}", rec_path.display()))?;
        let raw = match preset {
            "tiny" => CHARSET_TINY,
            "small" | "medium" => CHARSET_EXT,
            other => anyhow::bail!("unknown OCR preset {other}"),
        };
        let dict: Vec<&'static str> = raw.split('\n').collect();
        Ok(Self { det, rec, dict })
    }

    pub fn recognize(&mut self, image: &RgbImage) -> Result<Vec<OcrLine>> {
        let bgr = to_bgr(image);
        let det_in = preprocess::det_preprocess(&bgr);

        let input = Tensor::from_array((vec![1usize, 3, det_in.h, det_in.w], det_in.data))
            .context("det input tensor")?;
        let outs = self.det.run(ort::inputs!["x" => input]).context("det inference")?;
        let (shape, pred) = outs[0].try_extract_tensor::<f32>().context("det output")?;
        let (ph, pw) = (shape[2] as usize, shape[3] as usize);
        debug_assert_eq!(pw, det_in.w);
        debug_assert_eq!(ph, det_in.h);

        let mut boxes = db::detect(pred, pw, ph, det_in.src_w, det_in.src_h);
        db::sort_quad_boxes(&mut boxes);

        let mut lines = Vec::with_capacity(boxes.len());
        for quad in &boxes {
            let Some(crop) = geom::crop_rotated(&bgr, quad) else {
                continue;
            };
            let (data, imgw) = rec::rec_input(&crop);
            let input = Tensor::from_array((vec![1usize, 3, 48, imgw], data))
                .context("rec input tensor")?;
            let outs = self.rec.run(ort::inputs!["x" => input]).context("rec inference")?;
            let (rshape, logits) = outs[0].try_extract_tensor::<f32>().context("rec output")?;
            let (t, classes) = (rshape[1] as usize, rshape[2] as usize);
            let (text, score) = rec::ctc_decode(logits, t, classes, &self.dict);
            lines.push(OcrLine { poly: *quad, text, score });
        }
        Ok(lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip() {
        eprintln!("SKIPPED: private OCR fixture bundle absent (fixtures/shot.png)");
    }

    fn load_f32(p: &Path) -> Vec<f32> {
        let bytes = std::fs::read(p).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", what);
        let mut max_d = 0f32;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            if d > max_d {
                max_d = d;
            }
            assert!(
                d <= tol,
                "{}: idx={} got={} want={} (tol={})",
                what, i, x, y, tol
            );
        }
        println!("{}: n={} max_abs={:.3e}", what, a.len(), max_d);
    }

    #[test]
    fn det_inference_parity() {
        let Some(fx) = fixture_dir() else {
            return skip();
        };
        let mut eng = OcrEngine::from_dir(&fx, "tiny").unwrap();
        let x = load_f32(&fx.join("det_x.bin"));
        let expect = load_f32(&fx.join("det_y_ortref.bin"));
        let input = Tensor::from_array((vec![1usize, 3, 1088, 1920], x)).unwrap();
        let outs = eng.det.run(ort::inputs!["x" => input]).unwrap();
        let (_, y) = outs[0].try_extract_tensor::<f32>().unwrap();
        assert_close(y, &expect, 1e-6, "det output");
    }

    #[test]
    fn rec_inference_parity() {
        let Some(fx) = fixture_dir() else {
            return skip();
        };
        let mut eng = OcrEngine::from_dir(&fx, "tiny").unwrap();
        let x = load_f32(&fx.join("rec_x.bin"));
        let expect = load_f32(&fx.join("rec_y_ortref.bin"));
        let input = Tensor::from_array((vec![1usize, 3, 48, 320], x)).unwrap();
        let outs = eng.rec.run(ort::inputs!["x" => input]).unwrap();
        let (_, y) = outs[0].try_extract_tensor::<f32>().unwrap();
        assert_close(y, &expect, 1e-6, "rec output");
    }

    #[test]
    fn charset_sizes() {
        assert_eq!(CHARSET_TINY.split('\n').count(), 6904);
        assert_eq!(CHARSET_EXT.split('\n').count(), 18708);
    }

    #[test]
    fn ctc_mapping() {
        let dict: Vec<&str> = CHARSET_TINY.split('\n').collect();
        assert_eq!(dict.len(), 6904);
        let classes = 6906;
        let mut logits = vec![0f32; 2 * classes];
        logits[68] = 5.0;
        logits[classes + 68] = 4.0;
        logits[classes] = 4.5;
        logits[classes + 68] = 5.0;
        let (text, score) = rec::ctc_decode(&logits, 2, classes, &dict);
        assert_eq!(text, "Z");
        assert!(score > 0.9, "score={}", score);
        let mut logits = vec![0f32; classes];
        logits[6905] = 3.0;
        let (text, _) = rec::ctc_decode(&logits, 1, classes, &dict);
        assert_eq!(text, " ");
        let logits = vec![0f32; 4 * classes];
        let (text, score) = rec::ctc_decode(&logits, 4, classes, &dict);
        assert_eq!(text, "");
        assert_eq!(score, 0.0);
    }
}
