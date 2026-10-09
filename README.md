# kosuzu-ocr

PP-OCRv6 recognition engine

## Usage

```rust
let mut engine = kosuzu_ocr::OcrEngine::from_dir(&kosuzu_ocr::models_dir(), "small")?;
let lines = engine.recognize(&image::open("shot.png")?.to_rgb8())?;
for l in &lines {
    println!("{:?} {:.2} {}", l.poly, l.score, l.text);
}
```

Presets: `tiny` (6904-char dict) / `small` / `medium` (shared 18708-char dict).
A missing preset is downloaded on demand from HuggingFace by `ensure_preset()`.
`models_dir()` honors `KOSUZU_MODELS` if set, otherwise the per-platform
data dir (`dirs::data_local_dir`): `%LOCALAPPDATA%\kosuzu\models` on
Windows, `~/Library/Application Support/kosuzu/models` on macOS,
`$XDG_DATA_HOME/kosuzu/models` (i.e. `~/.local/share/kosuzu/models`) on
Linux. On Windows/macOS the engine additionally needs the ONNX Runtime
dylib: `onnxruntime.dll` / `libonnxruntime.dylib` next to the executable
or in the library path, or `ORT_DYLIB_PATH` pointing at it.

## Parity gate

`cargo test` runs a three-preset end-to-end parity check: the full Rust
pipeline vs official PaddleOCR output — per-box centroid distance ≤3px plus
exact text match. Any differences are confined to a pinned whitelist in
`tests/parity.rs` (each entry is an evidence-backed tolerance).

### Private fixtures

The parity references are derived from a **confidential screenshot** and are
deliberately **not** tracked in git. Only `fixtures/charset*.txt` (public model
vocabularies, required by `include_str!` in `src/lib.rs`) is committed.

Consequently, on a fresh clone the six fixture-bound tests print
`SKIPPED: private OCR fixture bundle absent` and pass — **the suite goes green
without the reference data**, so a green build does not mean those six ran.
Run `cargo test -- --nocapture` to see which ones skipped. Everything that
needs no reference data (`ctc_decode`, `fill_poly_exact` vs cv2, `py_round`,
preset normalization) always runs.

To restore the bundle on a new machine, it is kept out-of-band in a private
repository; copy `shot.png` plus whichever files you have back into `fixtures/`
and re-enable the tests:

```sh
cp <private-bundle>/shot.png <private-bundle>/*.json fixtures/
cargo test            # the full gate; *.bin-backed tests skip if absent
```

The `.bin` probe tensors additionally back `det_inference_parity`,
`rec_inference_parity` and `det_preprocess_matches_fixture`; without them those
three print `SKIPPED` and pass. Regenerate them with
`.venv/bin/python fixtures/probe_onnx.py` once `shot.png` is in place.

The reference generators live in `scripts/` (dev tools, Python + PaddleOCR).
Their input is the private screenshot and is not part of this repository:

- `scripts/dump_e2e_ref.py <preset>` → `fixtures/e2e_ref[_preset].json` (full Paddle pipeline)
- `scripts/dump_rec_ref.py <preset>` → `fixtures/e2e_ort_ref[_preset].json` (ORT backend,
  the source of the whitelist evidence)

`.gitignore` lists the excluded fixture patterns by name, and `Cargo.toml`
uses an `include` allowlist so `cargo publish` can never pick them up.

## Layout

```
src/lib.rs          engine entry: recognize() full flow + OcrLine
src/preprocess.rs  det preprocessing (DetResizeForTest + NormalizeImage)
src/db.rs          DB postprocessing (threshold/contours/score/unclip), incl. exact clipper offset
src/geom.rs        minAreaRect / perspective crop / rot90
src/rec.rs         rec input construction + CTC decode
src/models.rs      preset normalization + on-demand download
fixtures/          public charsets (embedded) + private parity references (untracked)
scripts/           reference generators (Python, dev tools only)
```
