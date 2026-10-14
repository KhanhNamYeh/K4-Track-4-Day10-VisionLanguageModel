"""Nạp Qwen3-VL-4B 4-bit, mã hoá ví dụ + mask loss, setup LoRA, vòng train, suy luận (GUIDE bước 3–7)."""
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image

from lab10.data import objects_to_answer, prompt_for
from scripts.labcommon import read_jsonl

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DTYPE = torch.float16
# transformers 5.x so khớp bằng re.match từ đầu tên module: "visual" KHÔNG khớp "model.visual.*"
SKIP_MODULES = ["model.visual", "lm_head"]
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
LLM_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_TARGETS = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"


# ---------- Bước 3: nạp model ----------
def load_model(model_id=MODEL_ID):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP_MODULES)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=DTYPE, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def load_split(split, data_dir="data", limit=None, fraction=None, seed=42):
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    if fraction is not None and fraction < 1.0:
        rows = random.Random(seed).sample(rows, max(1, int(len(rows) * fraction)))
    return rows[:limit] if limit else rows


def open_image(row, data_dir="data"):
    return Image.open(Path(data_dir) / row["image_path"]).convert("RGB")


def user_msg(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


# ---------- Bước 3/7: suy luận ----------
@torch.no_grad()
def predict(model, processor, rows, out_path, coord="norm1000", batch_size=16,
            max_new_tokens=256, data_dir="data", log_every=5):
    model.eval()
    processor.tokenizer.padding_side = "left"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for bi in range(0, len(rows), batch_size):
            batch = rows[bi:bi + batch_size]
            msgs = [[user_msg(open_image(r, data_dir), prompt_for(coord, r["width"], r["height"]))]
                    for r in batch]
            inputs = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True).to(model.device)
            with torch.autocast("cuda", dtype=DTYPE):
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for r, raw in zip(batch, raws):
                f.write(json.dumps({"image_id": r["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            if (bi // batch_size) % log_every == 0:
                print(f"  predict {bi + len(batch)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    return out_path


# ---------- Bước 4: ví dụ huấn luyện + mask loss ----------
def encode_example(processor, image, prompt, answer):
    user = user_msg(image, prompt)
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    f_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **kw)
    n = p_enc["input_ids"].shape[1]
    assert torch.equal(f_enc["input_ids"][0, :n], p_enc["input_ids"][0]), "prompt không phải tiền tố"
    labels = f_enc["input_ids"].clone()
    labels[:, :n] = -100
    # chỉ học JSON + <|im_end|>; bỏ "\n" mà template thêm sau <|im_end|>
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = (f_enc["input_ids"][0] == im_end).nonzero().flatten()
    labels[:, int(ends[-1]) + 1:] = -100
    item = {k: v for k, v in f_enc.items()}
    item["labels"] = labels
    return item


def collate(batch, pad_id):
    L = max(b["input_ids"].shape[1] for b in batch)

    def pad(t, value):
        return torch.cat([t, t.new_full((1, L - t.shape[1]), value)], dim=1)

    out = {
        "input_ids": torch.cat([pad(b["input_ids"], pad_id) for b in batch]),
        "attention_mask": torch.cat([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.cat([pad(b["labels"], -100) for b in batch]),
        "pixel_values": torch.cat([b["pixel_values"] for b in batch]),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch]),
    }
    if "mm_token_type_ids" in batch[0]:
        out["mm_token_type_ids"] = torch.cat([pad(b["mm_token_type_ids"], 0) for b in batch])
    return out


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, processor, rows, coord="norm1000", data_dir="data"):
        self.p, self.rows, self.coord, self.dir = processor, rows, coord, data_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ans = objects_to_answer(r["objects"], self.coord, r["width"], r["height"])
        return encode_example(self.p, open_image(r, self.dir),
                              prompt_for(self.coord, r["width"], r["height"]), ans)


# ---------- Bước 5: LoRA / đóng băng ----------
def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, tot


def apply_setup(model, targets="llm", train_projector=False, r=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    regex = LLM_TARGETS if targets == "llm" else f"({LLM_TARGETS})|({VIT_TARGETS})"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=regex, task_type="CAUSAL_LM"))
    extra = []
    if train_projector:
        for n, p in model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS):
                p.data = p.data.float()
                p.requires_grad = True
                extra.append(n)
    tr, tot = count_params(model)
    print(f"LoRA modules={len(matched)}  projector tensors={len(extra)}  "
          f"trainable={tr:,} / {tot:,}", flush=True)
    return model, matched, extra


def projector_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if any(k in n for k in PROJECTOR_KEYS)}


def load_trained(adapter_dir, model_id=MODEL_ID):
    """Nạp model gốc 4-bit → adapter LoRA → tham số projector đã huấn luyện (nếu có)."""
    from peft import PeftModel

    model, processor = load_model(model_id)
    model = PeftModel.from_pretrained(model, adapter_dir)
    proj = Path(adapter_dir) / "projector.pt"
    if proj.exists():
        sd = torch.load(proj, map_location="cpu")
        params = dict(model.named_parameters())
        for n, v in sd.items():
            params[n].data.copy_(v.to(params[n].dtype))
        print(f"nạp lại {len(sd)} tensor projector", flush=True)
    return model, processor


# ---------- Bước 6: vòng huấn luyện ----------
def train(model, processor, rows, out_dir, coord="norm1000", lr=2e-4, batch_size=2,
          grad_accum=4, max_steps=300, seed=42, save_every=50, log_every=10, data_dir="data"):
    torch.manual_seed(seed)
    random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor.tokenizer.padding_side = "right"
    ds = TrainSet(processor, rows, coord, data_dir)
    pad_id = processor.tokenizer.pad_token_id
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g,
                                     num_workers=4, collate_fn=lambda b: collate(b, pad_id),
                                     drop_last=True, persistent_workers=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, int(0.05 * max_steps))

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0, step, losses, history = time.time(), 0, [], []
    it = iter(dl)
    while step < max_steps:
        acc = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=DTYPE):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: {loss.item()}")
            scaler.scale(loss).backward()
            acc += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        losses.append(acc)
        if step % log_every == 0 or step == 1:
            rec = {"step": step, "loss": sum(losses[-log_every:]) / len(losses[-log_every:]),
                   "lr": sched.get_last_lr()[0], "grad_norm": float(gn),
                   "min": (time.time() - t0) / 60}
            history.append(rec)
            print(json.dumps(rec), flush=True)
        if step % save_every == 0 or step == max_steps:
            save_trained(model, out_dir)
    minutes = (time.time() - t0) / 60
    tr, _ = count_params(model)
    run = {"trainable_params": tr,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
           "train_minutes": round(minutes, 2), "steps": max_steps, "n_train_images": len(rows),
           "gpu": torch.cuda.get_device_name(0), "history": history}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    return run


def save_trained(model, out_dir):
    model.save_pretrained(out_dir)
    proj = projector_state(model)
    if any(p.requires_grad for n, p in model.named_parameters() if n in proj):
        torch.save(proj, Path(out_dir) / "projector.pt")
