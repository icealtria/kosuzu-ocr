use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const PRESETS: [&str; 3] = ["medium", "small", "tiny"];
const DEFAULT_PRESET: &str = "small";

pub fn normalize_preset(s: &str) -> &'static str {
    PRESETS
        .iter()
        .copied()
        .find(|p| *p == s)
        .unwrap_or(DEFAULT_PRESET)
}

pub fn det_path(dir: &Path, preset: &str) -> PathBuf {
    dir.join(format!("{preset}_det.onnx"))
}

pub fn rec_path(dir: &Path, preset: &str) -> PathBuf {
    dir.join(format!("{preset}_rec.onnx"))
}

pub fn is_downloaded(dir: &Path, preset: &str) -> bool {
    det_path(dir, preset).is_file() && rec_path(dir, preset).is_file()
}

pub fn ensure_preset(dir: &Path, preset: &str) -> Result<()> {
    if is_downloaded(dir, preset) {
        return Ok(());
    }
    std::fs::create_dir_all(dir).with_context(|| format!("create model dir {}", dir.display()))?;
    for kind in ["det", "rec"] {
        download_one(dir, preset, kind)?;
    }
    Ok(())
}

fn download_one(dir: &Path, preset: &str, kind: &str) -> Result<()> {
    let url =
        format!("https://huggingface.co/PaddlePaddle/PP-OCRv6_{preset}_{kind}_onnx/resolve/main/inference.onnx");
    let dst = dir.join(format!("{preset}_{kind}.onnx"));
    let part = dir.join(format!("{preset}_{kind}.onnx.part"));
    if dst.is_file() {
        return Ok(());
    }
    eprintln!("[kosuzu] downloading {url}");
    let mut resp = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("build HTTP client")?
        .get(&url)
        .send()
        .with_context(|| format!("request {url}"))?
        .error_for_status()
        .with_context(|| format!("download {url}"))?;
    {
        let mut file =
            std::fs::File::create(&part).with_context(|| format!("create {}", part.display()))?;
        std::io::copy(&mut resp, &mut file).with_context(|| format!("write {}", part.display()))?;
    }
    std::fs::rename(&part, &dst).with_context(|| format!("install {}", dst.display()))?;
    eprintln!("[kosuzu] ready {}", dst.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_follows_python_semantics() {
        assert_eq!(normalize_preset("tiny"), "tiny");
        assert_eq!(normalize_preset("small"), "small");
        assert_eq!(normalize_preset("medium"), "medium");
        assert_eq!(normalize_preset(""), "small");
        assert_eq!(normalize_preset("large"), "small");
    }

    #[test]
    fn fixture_layout_matches_preset_naming() {
        let Some(fx) = crate::fixture_dir() else {
            eprintln!("SKIPPED: private OCR fixture bundle absent (fixtures/shot.png)");
            return;
        };
        assert!(is_downloaded(&fx, "tiny"));
        assert!(!is_downloaded(&fx, "small"));
    }
}
