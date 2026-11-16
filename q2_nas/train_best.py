"""Train the best architecture found by the NAS from scratch on the full CIFAR-10 training set, evaluate on the
CIFAR-10 test set, save the checkpoint and export to ONNX.

    python q2_nas/train_best.py --epochs 60
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from common.data import GPUData
from common.utils import CKPT, ONNX, RESULTS, ensure_dirs, save_json, seed_everything, count_params, evaluate
from q1_kd.trainer import fit
from q2_nas.generator import build_model
from q2_nas.hardware import export_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch-file", default=os.path.join(RESULTS, "best_architecture.json"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ensure_dirs(); seed_everything(a.seed)
    cfg = json.load(open(a.arch_file))["architecture"]
    model = build_model(cfg)
    print("architecture:", json.dumps(cfg))
    print(f"params: {count_params(model) / 1e6:.3f}M")
    train, test = GPUData("cifar10", True), GPUData("cifar10", False)
    t0 = time.time()
    hist = fit(model, train, test, a.epochs, a.lr, a.bs, label_smoothing=0.1)
    acc = evaluate(model, test)
    torch.save({"arch": "nas_best", "config": cfg, "state_dict": model.state_dict(), "acc": acc, "epochs": a.epochs,
                "num_classes": 10}, os.path.join(CKPT, "nas_best.pth"))
    export_onnx(model, os.path.join(ONNX, "nas_best.onnx"))
    save_json({"model": "nas_best", "test_acc": acc, "epochs": a.epochs, "params": count_params(model),
               "train_time_s": time.time() - t0, "history": hist}, os.path.join(RESULTS, "nas_best_training.json"))
    print(f"NAS-best final CIFAR-10 test accuracy: {acc:.2f}%  -> checkpoints/nas_best.pth, onnx_models/nas_best.onnx")


if __name__ == "__main__":
    main()
