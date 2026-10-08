//! Standalone binary: OCR an image, print [{poly, text, score}] as JSON to stdout.
//!
//! Presets: tiny/small/medium (default small). Models dir shared with the lib.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Image to OCR
    image: PathBuf,

    /// Model preset
    #[arg(default_value = "small", value_parser = ["tiny", "small", "medium"])]
    preset: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let dir = kosuzu_ocr::models_dir();
    kosuzu_ocr::ensure_preset(&dir, &args.preset)?;
    let mut engine = kosuzu_ocr::OcrEngine::from_dir(&dir, &args.preset)?;
    let img = image::open(&args.image)?.to_rgb8();
    let lines = engine.recognize(&img)?;
    println!("{}", serde_json::to_string_pretty(&lines)?);
    Ok(())
}
