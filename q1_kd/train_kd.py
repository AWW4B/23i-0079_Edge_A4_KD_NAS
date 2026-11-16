"""Knowledge-distillation pipeline: student learns from the ResNet-50 teacher.

Loss = (1-alpha)*CE + alpha*T^2*KL(soft targets) + beta*feature-distillation (intermediate features)

    python q1_kd/train_kd.py --arch mobilenetv2 --epochs 60 --T 4 --alpha 0.7 --beta 1.0
    python q1_kd/train_kd.py --arch all
    python q1_kd/train_kd.py --arch squeezenet --no-feat      # logit-only KD (ablation)
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from common.data import GPUData
from common.utils import CKPT, RESULTS, ensure_dirs, save_json, seed_everything, count_params, evaluate
from q1_kd.kd import DistillationLoss
from q1_kd.models import STUDENTS, TEACHER_CH, build_student, load_checkpoint
from q1_kd.trainer import fit


def train_one(arch, a):
    seed_everything(a.seed)
    train, test = GPUData("cifar100", True), GPUData("cifar100", False)
    teacher, tck = load_checkpoint(os.path.join(CKPT, "teacher_resnet50.pth"), "cuda")
    teacher = teacher.to(memory_format=torch.channels_last)
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[kd] teacher acc {tck['acc']:.2f}%")
    model = build_student(arch)
    kd = DistillationLoss(a.T, a.alpha, a.beta, not a.no_feat, model.feature_channels(), TEACHER_CH)
    print(f"[kd] {arch}: {count_params(model) / 1e6:.2f}M params  T={a.T} alpha={a.alpha} beta={a.beta} feat={not a.no_feat}")
    t0 = time.time()
    hist = fit(model, train, test, a.epochs, a.lr, a.bs, teacher=teacher, kd_loss=kd)
    acc = evaluate(model, test)
    tag = "kd" if not a.no_feat else "kdlogit"
    torch.save({"arch": arch, "state_dict": model.state_dict(), "acc": acc, "epochs": a.epochs, "num_classes": 100,
                "mode": tag, "T": a.T, "alpha": a.alpha, "beta": a.beta}, os.path.join(CKPT, f"{tag}_{arch}.pth"))
    save_json({"model": arch, "mode": tag, "test_acc": acc, "epochs": a.epochs, "params": count_params(model), "T": a.T,
               "alpha": a.alpha, "beta": a.beta, "train_time_s": time.time() - t0, "history": hist},
              os.path.join(RESULTS, f"{tag}_{arch}.json"))
    print(f"[kd] {arch} final test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=list(STUDENTS) + ["all"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--no-feat", action="store_true", help="disable feature-based distillation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ensure_dirs()
    for arch in (STUDENTS if a.arch == "all" else [a.arch]):
        train_one(arch, a)
