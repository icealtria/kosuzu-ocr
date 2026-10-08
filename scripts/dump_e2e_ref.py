"""Dump raw PaddleOCR end-to-end reference: det+rec output to fixtures/e2e_ref_<preset>.json.

Same shape as fixtures/e2e_ref.json (tiny), used by tests/parity.rs.
Generating tiny to a temp file can cross-check the pinned reference:

    .venv/bin/python scripts/dump_e2e_ref.py tiny /tmp/e2e_ref_tiny.json
    .venv/bin/python scripts/dump_e2e_ref.py small fixtures/e2e_ref_small.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

from paddleocr import PaddleOCR  # noqa: E402


def ndarray_to_list(o):
    return o.tolist()


FX = Path(__file__).resolve().parents[1] / "fixtures"
KEYS = ("dt_polys", "rec_polys", "rec_texts", "rec_scores")


def main() -> None:
    preset, out = sys.argv[1], sys.argv[2]
    engine = PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{preset}_det",
        text_recognition_model_name=f"PP-OCRv6_{preset}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    img = FX / "shot.png"
    items = []
    for r in engine.predict(str(img)):
        items.append({"image": str(img), **{k: r[k] for k in KEYS}})
    assert len(items) == 1, f"expected single-image single-result, got {len(items)}"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1, default=ndarray_to_list)
    print(f"{preset}: {len(items[0]['rec_texts'])} boxes -> {out}")


if __name__ == "__main__":
    main()
