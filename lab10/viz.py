"""Vẽ box đúng (xanh lá) và box dự đoán (đỏ) cho ảnh minh hoạ; chọn ảnh tốt/xấu theo số TP/FP/FN."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw  # noqa: E402

from scripts.labcommon import read_jsonl  # noqa: E402
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output  # noqa: E402

GT_COLOR, PRED_COLOR = (0, 220, 0), (255, 40, 40)


def to_px(b, w, h):
    return [b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h]


def draw(entry, pred, data_dir, out):
    im = Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    for g in _gt_boxes(entry):
        d.rectangle(to_px(g["bbox"], w, h), outline=GT_COLOR, width=2)
        d.text((to_px(g["bbox"], w, h)[0] + 2, to_px(g["bbox"], w, h)[3] + 1), g["label"], fill=GT_COLOR)
    for p in pred or []:
        b = to_px(p["bbox"], w, h)
        d.rectangle(b, outline=PRED_COLOR, width=2)
        d.text((b[0] + 2, max(0, b[1] - 11)), p["label"], fill=PRED_COLOR)
    im.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="report/figures")
    ap.add_argument("--prefix", default="e")
    ap.add_argument("--n", type=int, default=4, help="số ảnh tốt và số ảnh xấu")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    entries = {e["image_id"]: e for e in read_jsonl(Path(a.data_dir) / "splits" / f"{a.split}.jsonl")}
    scored = []
    for row in read_jsonl(a.pred):
        e = entries[row["image_id"]]
        pred = parse_output(row["raw"])
        pred = _to_norm1000(pred, "norm1000", e["width"], e["height"]) if pred else pred
        per, _ = _match(_gt_boxes(e), pred or [])
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        scored.append((tp, fp, fn, e, pred))
    good = sorted([s for s in scored if s[0] >= 2], key=lambda s: (s[1] + s[2]) - s[0])[:a.n]
    bad = sorted(scored, key=lambda s: -(s[1] + s[2]))[:a.n]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    captions = []
    for kind, group in (("good", good), ("bad", bad)):
        for i, (tp, fp, fn, e, pred) in enumerate(group):
            name = f"{a.prefix}_{kind}_{i}.jpg"
            draw(e, pred, a.data_dir, out / name)
            captions.append({"file": name, "image_id": e["image_id"], "tp": tp, "fp": fp, "fn": fn,
                             "gt": [o["label"] for o in e["objects"]],
                             "pred": [p["label"] for p in pred or []]})
    (out / f"{a.prefix}_captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")
    for c in captions:
        print(c)


if __name__ == "__main__":
    main()
