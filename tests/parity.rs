use serde::Deserialize;
use std::path::{Path, PathBuf};

const EDIT1_MAX: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Tol {
    OrtExact,
    OrtEdit1,
    RefEdit1,
}

struct Gate {
    preset: &'static str,
    ref_file: &'static str,
    ort_ref_file: &'static str,
    whitelist: &'static [(usize, Tol)],
}

const GATES: &[Gate] = &[
    Gate {
        preset: "tiny",
        ref_file: "e2e_ref.json",
        ort_ref_file: "e2e_ort_ref.json",
        whitelist: &[
            (3, Tol::RefEdit1),
            (17, Tol::OrtExact),
            (19, Tol::OrtExact),
            (24, Tol::OrtExact),
            (37, Tol::RefEdit1),
            (48, Tol::OrtExact),
            (65, Tol::OrtExact),
        ],
    },
    Gate {
        preset: "small",
        ref_file: "e2e_ref_small.json",
        ort_ref_file: "e2e_ort_ref_small.json",
        whitelist: &[
            (0, Tol::RefEdit1),
            (5, Tol::OrtExact),
            (39, Tol::OrtExact),
            (41, Tol::OrtExact),
            (46, Tol::OrtExact),
            (48, Tol::OrtExact),
            (53, Tol::RefEdit1),
        ],
    },
    Gate {
        preset: "medium",
        ref_file: "e2e_ref_medium.json",
        ort_ref_file: "e2e_ort_ref_medium.json",
        whitelist: &[
            (3, Tol::OrtEdit1),
            (9, Tol::OrtExact),
            (12, Tol::OrtExact),
            (17, Tol::RefEdit1),
            (40, Tol::OrtExact),
            (48, Tol::OrtExact),
        ],
    },
];

#[derive(Deserialize)]
struct RefItem {
    dt_polys: Vec<Vec<Vec<f64>>>,
    rec_texts: Vec<String>,
}

#[derive(Deserialize)]
struct RefTexts {
    rec_texts: Vec<String>,
}

fn golden_dir() -> Option<PathBuf> {
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    fx.join("shot.png").is_file().then_some(fx)
}

fn model_dir(gate: &Gate, fx: &Path) -> PathBuf {
    if gate.preset == "tiny" {
        fx.to_path_buf()
    } else {
        kosuzu_ocr::models_dir()
    }
}

fn centroid_f64(poly: &[[f64; 2]]) -> (f64, f64) {
    let n = poly.len() as f64;
    (
        poly.iter().map(|p| p[0]).sum::<f64>() / n,
        poly.iter().map(|p| p[1]).sum::<f64>() / n,
    )
}

#[test]
fn e2e_tiny_matches_paddleocr_reference() {
    let Some(fx) = golden_dir() else {
        return skip();
    };
    run_gate(&GATES[0], &fx);
}

#[test]
fn e2e_small_matches_paddleocr_reference() {
    let Some(fx) = golden_dir() else {
        return skip();
    };
    run_gate(&GATES[1], &fx);
}

#[test]
fn e2e_medium_matches_paddleocr_reference() {
    let Some(fx) = golden_dir() else {
        return skip();
    };
    run_gate(&GATES[2], &fx);
}

fn skip() {
    eprintln!("SKIPPED: private OCR fixture bundle absent (fixtures/shot.png)");
}

fn run_gate(g: &Gate, fx: &Path) {
    let img = image::open(fx.join("shot.png")).unwrap().to_rgb8();
    let dir = model_dir(g, &fx);
    let mut engine = kosuzu_ocr::OcrEngine::from_dir(&dir, g.preset).unwrap_or_else(|e| {
        panic!(
            "{preset} model load failed: {e:#}\ndir {dir} (run the app once to download small/medium)",
            preset = g.preset,
            dir = dir.display()
        )
    });
    let lines = engine.recognize(&img).expect("recognize");

    let raw = std::fs::read_to_string(fx.join(g.ref_file)).unwrap();
    let refs: Vec<RefItem> = serde_json::from_str(&raw).unwrap();
    assert_eq!(refs.len(), 1);
    let r = &refs[0];

    let raw_ort = std::fs::read_to_string(fx.join(g.ort_ref_file)).unwrap();
    let ort_refs: RefTexts = serde_json::from_str(&raw_ort).unwrap();
    assert_eq!(ort_refs.rec_texts.len(), r.rec_texts.len());
    let ort_diff_idx: Vec<usize> = (0..r.rec_texts.len())
        .filter(|&i| ort_refs.rec_texts[i].trim() != r.rec_texts[i].trim())
        .collect();
    let ort_star: Vec<usize> = g
        .whitelist
        .iter()
        .filter(|(_, t)| matches!(t, Tol::OrtExact | Tol::OrtEdit1))
        .map(|(i, _)| *i)
        .collect();
    assert_eq!(
        ort_diff_idx, ort_star,
        "inference boundary set drifted for {}, recheck dump_rec_ref.py",
        g.preset
    );

    let diag = diagnose_missing(&lines, r);
    assert_eq!(
        lines.len(),
        r.dt_polys.len(),
        "box count mismatch (Rust {} vs ref {})\n{}",
        lines.len(),
        r.dt_polys.len(),
        diag
    );

    let mut n_exact = 0usize;
    let mut n_tolerated = 0usize;
    let mut text_failures: Vec<String> = Vec::new();
    let mut max_dist = 0f64;
    let mut n_pt_diff = 0usize;
    let mut max_pt_diff = 0f64;
    for (i, (l, poly)) in lines.iter().zip(r.dt_polys.iter()).enumerate() {
        let mine: Vec<[f64; 2]> = l.poly.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
        let refq: Vec<[f64; 2]> = poly.iter().map(|p| [p[0], p[1]]).collect();
        let (ax, ay) = centroid_f64(&mine);
        let (bx, by) = centroid_f64(&refq);
        let d = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
        max_dist = max_dist.max(d);
        let pt_diff = mine
            .iter()
            .zip(refq.iter())
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
            .fold(0f64, f64::max);
        if pt_diff > 0.0 {
            n_pt_diff += 1;
        }
        if pt_diff > max_pt_diff {
            max_pt_diff = pt_diff;
        }
        if pt_diff > 0.0 && n_pt_diff <= 6 {
            println!(
                "box {} vertex diff {:.1}px: mine={:?}",
                i,
                pt_diff,
                l.poly
                    .iter()
                    .map(|p| [p[0] as i64, p[1] as i64])
                    .collect::<Vec<_>>()
            );
            println!(
                "           ref ={:?}",
                poly.iter()
                    .map(|p| [p[0] as i64, p[1] as i64])
                    .collect::<Vec<_>>()
            );
        }
        assert!(d <= 3.0, "box {} centroid distance {:.2}px > 3px", i, d);
        if l.text.trim() == r.rec_texts[i].trim() {
            n_exact += 1;
        } else if let Some((_, tol)) = g.whitelist.iter().find(|(j, _)| *j == i) {
            let mine = l.text.trim();
            let (ok, expect) = match tol {
                Tol::OrtExact => (mine == ort_refs.rec_texts[i].trim(), "Rust==ORT exact"),
                Tol::OrtEdit1 => (
                    levenshtein(mine, ort_refs.rec_texts[i].trim()) <= EDIT1_MAX,
                    "edit distance to ORT <= 1",
                ),
                Tol::RefEdit1 => (
                    levenshtein(mine, r.rec_texts[i].trim()) <= EDIT1_MAX,
                    "edit distance to ref <= 1",
                ),
            };
            if ok {
                n_tolerated += 1;
                println!(
                    "text {} whitelisted ({:?}, {}): mine={:?} ref={:?}",
                    i, tol, expect, l.text, r.rec_texts[i]
                );
            } else {
                text_failures.push(format!(
                    "text {} whitelist {expect} not met: mine={:?} ort={:?} ref={:?}",
                    i, l.text, ort_refs.rec_texts[i], r.rec_texts[i]
                ));
            }
        } else {
            text_failures.push(format!(
                "text {} mismatch (vertex max diff={:.1}px): mine={:?} ref={:?}",
                i, pt_diff, l.text, r.rec_texts[i]
            ));
        }
    }
    println!(
        "[{}] e2e: {} boxes, centroid max={:.2}px, {} boxes with vertex diffs (max={:.2}px); text: exact {}, whitelisted {}, failed {}",
        g.preset,
        lines.len(),
        max_dist,
        n_pt_diff,
        max_pt_diff,
        n_exact,
        n_tolerated,
        text_failures.len()
    );
    assert!(
        text_failures.is_empty(),
        "text gate failed at {} spots:\n{}",
        text_failures.len(),
        text_failures.join("\n")
    );
}

fn diagnose_missing(lines: &[kosuzu_ocr::OcrLine], r: &RefItem) -> String {
    let mut report = String::new();
    for (j, poly) in r.dt_polys.iter().enumerate() {
        let refq: Vec<[f64; 2]> = poly.iter().map(|p| [p[0], p[1]]).collect();
        let (bx, by) = centroid_f64(&refq);
        let best = lines
            .iter()
            .map(|l| {
                let mine: Vec<[f64; 2]> =
                    l.poly.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
                let (ax, ay) = centroid_f64(&mine);
                ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        if best > 3.0 {
            report.push_str(&format!(
                "ref[{}] unmatched: centroid=({:.1},{:.1}) best_dist={:.2} text={:?} poly={:?}\n",
                j, bx, by, best, r.rec_texts[j], poly
            ));
        }
    }
    for (i, l) in lines.iter().enumerate() {
        let mine: Vec<[f64; 2]> = l.poly.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
        let (ax, ay) = centroid_f64(&mine);
        let best = r
            .dt_polys
            .iter()
            .map(|poly| {
                let refq: Vec<[f64; 2]> = poly.iter().map(|p| [p[0], p[1]]).collect();
                let (bx, by) = centroid_f64(&refq);
                ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        if best > 3.0 {
            report.push_str(&format!(
                "mine[{}] unmatched: centroid=({:.1},{:.1}) best_dist={:.2} text={:?}\n",
                i, ax, ay, best, l.text
            ));
        }
    }
    report
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur.push((prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}
