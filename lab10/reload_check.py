"""Kiểm tra GUIDE bước 7: nạp model gốc → adapter → projector.pt rồi suy luận lại 40 ảnh test; so với dự đoán lúc train."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab10 import vlm  # noqa: E402
from scripts.labcommon import read_jsonl  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "e3"
rows = vlm.load_split("test", limit=40)
model, proc = vlm.load_trained(f"work/adapters/{name}")
out = vlm.predict(model, proc, rows, f"work/{name}_reload40.jsonl")
a = {r["image_id"]: r["raw"] for r in read_jsonl(f"work/{name}.jsonl")}
b = {r["image_id"]: r["raw"] for r in read_jsonl(out)}
same = sum(a[k] == b[k] for k in b)
print(json.dumps({"exp": name, "identical_outputs": same, "n": len(b)}))
