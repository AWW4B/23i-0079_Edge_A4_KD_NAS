"""Baseline pipeline: train a student from scratch with hard labels only (NO distillation).

    python q1_kd/train_baseline.py --arch mobilenetv2 --epochs 60
    python q1_kd/train_baseline.py --arch all
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from common.data import GPUData
from common.utils import CKPT, RESULTS, ensure_dirs, save_json, seed_everything, count_params, evaluate
from q1_kd.models import STUDENTS, build_student
from q1_kd.trainer import fit


def train_one(arch, epochs, bs, lr, seed):
    seed_everything(seed)
    train, test = GPUData("cifar100", True), GPUData("cifar100", False)
    model = build_student(arch)
    print(f"[baseline] {arch}: {count_params(model) / 1e6:.2f}M params")
    t0 = time.time()
    hist = fit(model, train, test, epochs, lr, bs, label_smoothing=0.0)
    acc = evaluate(model, test)
    torch.save({"arch": arch, "state_dict": model.state_dict(), "acc": acc, "epochs": epochs, "num_classes": 100, "mode": "baseline"},
               os.path.join(CKPT, f"baseline_{arch}.pth"))
    save_json({"model": arch, "mode": "baseline", "test_acc": acc, "epochs": epochs, "params": count_params(model),
               "train_time_s": time.time() - t0, "history": hist}, os.path.join(RESULTS, f"baseline_{arch}.json"))
    print(f"[baseline] {arch} final test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=list(STUDENTS) + ["all"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ensure_dirs()
    for arch in (STUDENTS if a.arch == "all" else [a.arch]):
        train_one(arch, a.epochs, a.bs, a.lr, a.seed)
