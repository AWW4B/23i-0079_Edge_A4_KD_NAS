import json, os, random, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
ONNX = os.path.join(ROOT, "onnx_models")
RESULTS = os.path.join(ROOT, "results")


def seed_everything(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def evaluate(model, data, bs=500, amp=True):
    model.eval()
    correct = total = 0
    for xb, yb in data.batches(bs, shuffle=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(xb)
        out = out[0] if isinstance(out, (tuple, list)) else out
        correct += (out.argmax(1) == yb).sum().item(); total += len(yb)
    return 100.0 * correct / total


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def ensure_dirs():
    for d in (CKPT, ONNX, RESULTS):
        os.makedirs(d, exist_ok=True)
