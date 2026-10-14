"""Chạy một thí nghiệm từ đầu đến cuối: (train) → suy luận → chấm bằng scripts/score.py.

    python run_exp.py e0                       # zero-shot
    python run_exp.py e1 | e2 | e3             # theo configs/eN.yaml
    python run_exp.py e4 --base e2 --data-size 0.1 --tag data_size_0.1
    python run_exp.py e1 --split val --limit 40 --max-steps 20   # gỡ lỗi nhanh
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lab10 import vlm  # noqa: E402


def score(pred, name, out, split, coord, meta=None):
    cmd = [sys.executable, "scripts/score.py", "--pred", str(pred), "--name", name,
           "--out", str(out), "--split", split, "--coord", coord]
    for k, v in (meta or {}).items():
        cmd += ["--meta", f"{k}={v}"]
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e0", "e1", "e2", "e3", "e4"])
    ap.add_argument("--base", default="e2", help="config gốc cho e4")
    ap.add_argument("--tag", default=None, help="hậu tố tên file e4 (vd data_size_0.1)")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--data-size", type=float, default=None)
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-infer", type=int, default=16)
    a = ap.parse_args()

    name = a.exp if a.exp != "e4" else f"e4_{a.tag}"
    debug = a.split != "test" or a.limit
    out_name = f"{name}_{a.split}{a.limit or ''}" if debug else name
    pred = ROOT / "work" / f"{out_name}.jsonl"
    evid = (ROOT / "work" / "debug" if debug else ROOT / "report" / "evidence") / f"{out_name}.json"
    test_rows = vlm.load_split(a.split, limit=a.limit)

    if a.exp == "e0":
        model, proc = vlm.load_model()
        vlm.predict(model, proc, test_rows, pred, coord=a.coord, batch_size=a.batch_infer)
        score(pred, "E0", evid, a.split, a.coord)
        return

    cfg = yaml.safe_load((ROOT / "configs" / f"{a.exp if a.exp != 'e4' else a.base}.yaml").read_text(encoding="utf-8"))
    lora = dict(cfg["lora"])
    if a.rank:
        lora["r"], lora["alpha"] = a.rank, 2 * a.rank
    steps = a.max_steps or cfg["max_steps"]
    train_rows = vlm.load_split("train", fraction=a.data_size, seed=cfg["seed"])
    print(f"[{name}] base={cfg['name']} lora={lora} projector={cfg['train_projector']} "
          f"coord={a.coord} train_images={len(train_rows)} steps={steps}", flush=True)

    model, proc = vlm.load_model()
    model, matched, extra = vlm.apply_setup(
        model, targets=lora["targets"], train_projector=cfg["train_projector"],
        r=lora["r"], alpha=lora["alpha"], dropout=lora["dropout"])
    adapter = ROOT / "work" / "adapters" / out_name
    run = vlm.train(model, proc, train_rows, adapter, coord=a.coord, lr=cfg["lr"],
                    batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
                    max_steps=steps, seed=cfg["seed"])
    run.update({"lora_modules": len(matched), "projector_tensors": len(extra), "config": cfg,
                "lora": lora, "coord": a.coord, "data_size": a.data_size or 1.0})
    (adapter / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    vlm.predict(model, proc, test_rows, pred, coord=a.coord, batch_size=a.batch_infer)
    meta = {k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}
    score(pred, name.upper() if a.exp != "e4" else f"E4 {a.tag}", evid, a.split, a.coord, meta)


if __name__ == "__main__":
    main()
