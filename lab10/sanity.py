"""Kiểm tra GUIDE bước 2, 4, 5: nhãn đi-về, 220 token ảnh, mask loss, số module LoRA, số tham số."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab10 import vlm  # noqa: E402
from lab10.data import objects_to_answer, prompt_for  # noqa: E402

rows = vlm.load_split("train", limit=50)
r = next(x for x in rows if len(x["objects"]) >= 2)
ans = objects_to_answer(r["objects"])
print("answer:", ans)
for o, b in zip(sorted(r["objects"], key=lambda o: (o["yolo"][1] - o["yolo"][3] / 2)), json.loads(ans)):
    px = [b["bbox_2d"][i] / 1000 * (640 if i % 2 == 0 else 360) for i in range(4)]
    xc, yc, w, h = o["yolo"]
    gt = [(xc - w / 2) * 640, (yc - h / 2) * 360, (xc + w / 2) * 640, (yc + h / 2) * 360]
    print("  pixel từ JSON", [round(v, 1) for v in px], " gốc", [round(v, 1) for v in gt])

model, proc = vlm.load_model()
n_vit = sum(p.numel() for n, p in model.named_parameters() if ".visual." in n)
n_merger = sum(p.numel() for n, p in model.named_parameters() if "visual.merger" in n)
n_deep = sum(p.numel() for n, p in model.named_parameters() if "visual.deepstack_merger_list" in n)
n_all = sum(p.numel() for p in model.parameters())
print(f"params (4-bit đếm theo phần tử lưu): all={n_all:,} visual={n_vit:,} merger={n_merger:,} deepstack={n_deep:,}")
print("dtype:", {n: str(p.dtype) for n, p in model.named_parameters() if n.endswith(("visual.merger.linear_fc1.weight", "visual.blocks.0.attn.qkv.weight", "language_model.layers.0.mlp.up_proj.weight"))})

item = vlm.encode_example(proc, vlm.open_image(r), prompt_for(), ans)
img_tok = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
print("image tokens:", int((item["input_ids"] == img_tok).sum()), "keys:", list(item.keys()))
lab = item["labels"][0]
print("supervised text:", repr(proc.tokenizer.decode(lab[lab != -100])))

for targets, proj in (("llm", False), ("llm", True), ("llm+vit", True)):
    m, matched, extra = vlm.apply_setup(model, targets=targets, train_projector=proj)
    tr, _ = vlm.count_params(m)
    print(f"setup targets={targets} projector={proj}: modules={len(matched)} trainable={tr:,}")
    model = m.unload()  # gỡ LoRA để thử setup tiếp theo
    for n, p in model.named_parameters():
        if any(k in n for k in vlm.PROJECTOR_KEYS):
            p.data = p.data.half()
