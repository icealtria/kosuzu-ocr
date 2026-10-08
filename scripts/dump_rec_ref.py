"""Dump raw paddlex rec reference per preset: crop-normalized tensor + ORT-computed text.

Isolation purpose:
- T_ref    = e2e_ref text (Paddle pipeline output)
- T_ortref = text decoded from ORT rec on the paddlex crop tensor
  T_ortref == T_ref -> inference has no diff, all residual mismatch is crop/resize (Rust parity)
  T_ortref != T_ref -> ORT vs Paddle numerical boundary diff (backend issue)

Run (venv with paddlex + onnxruntime installed):
    .venv/bin/python scripts/dump_rec_ref.py <preset>

Writes fixtures/e2e_ort_ref[_preset].json (inference-boundary whitelist evidence
for tests/parity.rs) and prints the mismatch set vs e2e_ref, which must match the
pinned per-preset ort_diff; drift requires manual review. Intermediate crops and
tensors go to $DUMP_REC_OUT or a temp dir.
"""
import json, os, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

FX = Path(__file__).resolve().parents[1] / "fixtures"
PRESET = sys.argv[1]
assert PRESET in ("tiny", "small", "medium"), PRESET

suffix = "" if PRESET == "tiny" else f"_{PRESET}"
OUT = Path(os.environ.get("DUMP_REC_OUT") or tempfile.mkdtemp(prefix="rec_ref"))
OUT.mkdir(parents=True, exist_ok=True)

from paddlex.inference.pipelines.components.common.crop_image_regions import CropByPolys
from paddlex.inference.models.text_recognition.processors import OCRReisizeNormImg

img = cv2.imread(str(FX / "shot.png"))  # BGR
refs = json.load(open(FX / f"e2e_ref{suffix}.json"))[0]
polys, ref_texts = refs["dt_polys"], refs["rec_texts"]
charset_file = "charset.txt" if PRESET == "tiny" else "charset18k.txt"
d = open(FX / charset_file, encoding="utf-8").read().split("\n")
assert len(d) == {"tiny": 6904, "small": 18708, "medium": 18708}[PRESET], len(d)

if PRESET == "tiny":
    rec_onnx = FX / "tiny_rec.onnx"
else:
    models = os.environ.get("KOSUZU_MODELS") or os.path.expanduser(
        os.path.join(os.environ.get("XDG_DATA_HOME", "~/.local/share"), "kosuzu/models")
    )
    rec_onnx = Path(models) / f"{PRESET}_rec.onnx"
assert rec_onnx.is_file(), rec_onnx

crop_cls = CropByPolys(det_box_type="quad")
resize = OCRReisizeNormImg(rec_image_shape=[3, 48, 320])

sess = ort.InferenceSession(str(rec_onnx), providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
print("rec onnx input:", iname, sess.get_inputs()[0].shape)


def ctc_decode(y):
    # y: (1, T, C)
    idx = y[0].argmax(-1)
    out, prev = [], None
    for k in idx:
        k = int(k)
        if k == prev:
            prev = k
            continue
        prev = k
        if k == 0:
            continue
        out.append(" " if k == len(d) + 1 else d[k - 1])
    return "".join(out)


meta = []
t_ortref = []
for i, poly in enumerate(polys):
    crop = crop_cls.get_minarea_rect_crop(img, np.array(poly))
    h, w = crop.shape[:2]
    ratio = w / float(h)
    max_wh_ratio = max(320.0 / 48.0, ratio)
    tensor = resize.resize_norm_img(crop, max_wh_ratio)  # (3,48,W) f32
    cv2.imwrite(str(OUT / f"crop_{i:02d}.png"), crop)
    np.save(OUT / f"tensor_{i:02d}.npy", tensor)
    y = sess.run(None, {iname: tensor[None]})[0]
    txt = ctc_decode(y)
    t_ortref.append(txt)
    meta.append({"i": i, "w": w, "h": h, "imgW": int(tensor.shape[2]), "text": txt})

json.dump(meta, open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1)

# Pin evidence text into fixtures for the whitelist in tests/parity.rs
ort_ref = [None] * len(polys)
for m in meta:
    ort_ref[m["i"]] = m["text"]
assert all(t is not None for t in ort_ref)
json.dump(
    {"rec_texts": ort_ref},
    open(FX / f"e2e_ort_ref{suffix}.json", "w"),
    ensure_ascii=False,
    indent=1,
)

diff_idx = [i for i, (a, b) in enumerate(zip(t_ortref, ref_texts)) if a.strip() != b.strip()]
for i in diff_idx:
    print(f"#{i}: ort(refcrop)={t_ortref[i]!r} ref={ref_texts[i]!r}")
print(f"ORT-vs-Paddle inference boundary set: {diff_idx} ({len(diff_idx)}/{len(ref_texts)})")
print(f"dumped -> {OUT}")
