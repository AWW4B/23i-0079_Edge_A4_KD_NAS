"""Train the ResNet-50 teacher on CIFAR-100.

    python q1_kd/train_teacher.py --epochs 40
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from common.data import GPUData
from common.utils import CKPT, RESULTS, ensure_dirs, save_json, seed_everything, count_params, evaluate
from q1_kd.models import ResNet50Teacher
from q1_kd.trainer import fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ensure_dirs(); seed_everything(a.seed)
    train, test = GPUData("cifar100", True), GPUData("cifar100", False)
    model = ResNet50Teacher()
    print(f"ResNet-50 teacher: {count_params(model) / 1e6:.2f}M params")
    t0 = time.time()
    hist = fit(model, train, test, a.epochs, a.lr, a.bs, label_smoothing=0.1)
    acc = evaluate(model, test)
    torch.save({"arch": "resnet50", "state_dict": model.state_dict(), "acc": acc, "epochs": a.epochs, "num_classes": 100},
               os.path.join(CKPT, "teacher_resnet50.pth"))
    save_json({"model": "resnet50", "test_acc": acc, "epochs": a.epochs, "train_time_s": time.time() - t0, "history": hist},
              os.path.join(RESULTS, "teacher_resnet50.json"))
    print(f"Teacher final test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
