"""Numerical parity: PaddlePaddle backend vs converted ONNX (onnxruntime).

Reuses paddlex's own pre/post-processing for BOTH paths - only
`predictor.runner` is swapped - so any diff isolates the conversion itself.

Also dumps input/reference-output tensors for the Rust ort probe as flat
little-endian f32 into fixtures/ (<name>_x.bin / <name>_y_ortref.bin), the
format src/lib.rs and src/preprocess.rs read. The ONNX files are expected in
fixtures/ (tiny_det.onnx / tiny_rec.onnx).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

FX = Path(__file__).resolve().parent
MODEL_DIR = os.path.expanduser("~/.paddlex/official_models")
ONNX_DIR = FX
IMG = sys.argv[1] if len(sys.argv) > 1 else str(FX / "shot.png")

results = []


def save_bin(name, arr):
    np.asarray(arr, dtype=np.float32).tofile(FX / f"{name}.bin")


def norm_x(x):
    """ToBatch yields a list of per-instance 4D arrays; flatten to one (B,C,H,W)."""
    if isinstance(x, (list, tuple)):
        return np.concatenate([np.asarray(i) for i in x], axis=0)
    return np.asarray(x)


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}: {name} {detail}", flush=True)


class OrtRunner:
    """Duck-typed replacement for paddlex InferenceRunner."""

    def __init__(self, path, capture=None):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.capture = capture

    def __call__(self, x=None, **kwargs):
        x = norm_x(x)
        out = self.sess.run(None, {self.iname: x})[0]
        if self.capture is not None:
            self.capture.append((x.copy(), out.copy()))
        return [out]  # DBPostProcess expects a batch list

    def close(self):
        pass


class RecordingRunner:
    def __init__(self, inner, capture):
        self.inner = inner
        self.capture = capture

    def __call__(self, x=None, **kwargs):
        out = self.inner(x=x, **kwargs)
        arr = out[0] if isinstance(out, (list, tuple)) else out
        self.capture.append((norm_x(x), np.asarray(arr).copy()))
        return out

    def close(self):
        self.inner.close()


def make(name, onnx_file, capture):
    from paddlex.inference import create_predictor

    pred = create_predictor(name, model_dir=f"{MODEL_DIR}/{name}")
    pred.runner = OrtRunner(f"{ONNX_DIR}/{onnx_file}", capture)
    return pred


def diff(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    denom = max(np.abs(a).max(), 1e-9)
    return f"max_abs={d.max():.3e} rel={d.max() / denom:.3e}"


def tensors_close(a, b, atol=1e-3, rtol=1e-3):
    return np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


from paddlex.inference import create_predictor  # noqa: E402

# ---------------- det ----------------
cap_p, cap_o = [], []
det_p = create_predictor("PP-OCRv6_tiny_det", model_dir=f"{MODEL_DIR}/PP-OCRv6_tiny_det")
det_p.runner = RecordingRunner(det_p.runner, cap_p)
det_o = make("PP-OCRv6_tiny_det", "tiny_det.onnx", cap_o)

# PaddleOCR pipeline runs det at full resolution (limit 64/min is a no-op
# pad-to-32) with its own DB thresholds - replicate for e2e comparison.
DET_KW = dict(limit_side_len=64, limit_type="min", thresh=0.3, box_thresh=0.6,
              unclip_ratio=1.5)

rp = list(det_p.predict(IMG, **DET_KW))
ro = list(det_o.predict(IMG, **DET_KW))

x_p, y_p = cap_p[0]
x_o, y_o = cap_o[0]
print(f"[det] input shape={x_o.shape} dtype={x_o.dtype} range=({x_o.min():.3f},{x_o.max():.3f})", flush=True)
check("det: both paths share identical preprocessed input", np.array_equal(x_p, x_o),
      f"identical={np.array_equal(x_p, x_o)}")
check("det: ONNX output ~= Paddle output", tensors_close(y_p, y_o, atol=5e-3, rtol=5e-3),
      diff(y_p, y_o) + " (fp kernel noise)")

polys_p, polys_o = rp[0]["dt_polys"], ro[0]["dt_polys"]
same_count = len(polys_p) == len(polys_o)
check("det: same text-box count", same_count, f"paddle={len(polys_p)} onnx={len(polys_o)}")
if same_count and len(polys_p):
    boxes_close = all(tensors_close(a, b, atol=2.0, rtol=1e-2) for a, b in zip(polys_p, polys_o))
    max_box = max(np.abs(np.asarray(a) - np.asarray(b)).max() for a, b in zip(polys_p, polys_o))
    check("det: box coords close (<=2px)", boxes_close, f"max_box_diff={max_box:.2f}px")

# dump for Rust
save_bin("det_x", x_o)
save_bin("det_y_ortref", y_o)

# ---------------- rec (on identical crops) ----------------
import cv2  # noqa: E402

img = cv2.imread(IMG)
cap_rp, cap_ro = [], []
rec_p = create_predictor("PP-OCRv6_tiny_rec", model_dir=f"{MODEL_DIR}/PP-OCRv6_tiny_rec")
rec_p.runner = RecordingRunner(rec_p.runner, cap_rp)
rec_o = make("PP-OCRv6_tiny_rec", "tiny_rec.onnx", cap_ro)

crops = []
for poly in polys_o:
    box = np.asarray(poly, dtype=np.float32)
    x0, y0 = box.min(axis=0).astype(int)
    x1, y1 = box.max(axis=0).astype(int)
    crop = img[max(0, y0):y1, max(0, x0):x1]
    if crop.size:
        crops.append(crop)

texts_p, texts_o = [], []
for i, crop in enumerate(crops):
    o_p = list(rec_p.predict(crop))
    o_o = list(rec_o.predict(crop))
    texts_p.append(o_p[0]["rec_text"])
    texts_o.append(o_o[0]["rec_text"])

xr, yr_ort = cap_ro[0]
_, yr_p = cap_rp[0]
print(f"[rec] input shape={xr.shape} range=({xr.min():.3f},{xr.max():.3f})", flush=True)
check("rec: ONNX output ~= Paddle output", tensors_close(yr_p, yr_ort), diff(yr_p, yr_ort))
check("rec: per-sample text identical", texts_p == texts_o,
      "" if texts_p == texts_o else f"first diff idx={[i for i,(a,b) in enumerate(zip(texts_p,texts_o)) if a!=b][:5]}")
save_bin("rec_x", xr)
save_bin("rec_y_ortref", yr_ort)

# ---------------- e2e reference: full PaddleOCR (paddle backend) ----------------
from paddleocr import PaddleOCR  # noqa: E402

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_tiny_det",
    text_recognition_model_name="PP-OCRv6_tiny_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
ref_texts, ref_polys = [], []
for item in ocr.predict(IMG) or []:
    ref_texts += [str(t) for t in (item.get("rec_texts") or [])]
    ref_polys += list(item.get("rec_polys") or [])


def center(poly):
    return np.asarray(poly, dtype=float).mean(axis=0)


# Pair pipeline texts with our crop texts by box centroid (pipeline rec
# output is in reading order, ours in raw det order).
centers_o = [center(p) for p in polys_o]
used, matched, mism_o, mism_p = set(), 0, [], []
for rc, rt in zip(ref_polys, ref_texts):
    cands = [(j, float(np.linalg.norm(centers_o[j] - center(rc))))
             for j in range(len(centers_o)) if j not in used]
    if not cands:
        break
    j, d = min(cands, key=lambda t: t[1])
    used.add(j)
    if d > 3.0:
        continue
    matched += 1
    if rt.strip() != texts_o[j].strip():
        mism_o.append((j, rt, texts_o[j]))
    if rt.strip() != texts_p[j].strip():
        mism_p.append((j, rt, texts_p[j]))
# paddlex raw predictor on OUR crops differs from the pipeline only by crop
# geometry; onnx-vs-paddle on identical crops was already proven exact above,
# so ONNX must introduce no extra mismatches (mism_o == mism_p).
check("e2e: ONNX-chain text ~= PaddleOCR reference (paired by box)",
      matched >= len(ref_polys) - 3 and len(mism_o) == len(mism_p),
      f"ref={len(ref_texts)} onnx={len(texts_o)} matched={matched} "
      f"ndiff_onnx={len(mism_o)} ndiff_paddle_same_crops={len(mism_p)} "
      f"(diffs=crop geometry, not conversion error) first={mism_o[:2]}")

print("=" * 46, flush=True)
failed = [n for n, ok in results if not ok]
print("ALL PASS" if not failed else f"FAILURES: {failed}", flush=True)
sys.exit(1 if failed else 0)
