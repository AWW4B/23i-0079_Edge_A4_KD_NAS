"""Generate the figures used in the technical report from results/*.json|csv  ->  report/figures/*.png"""
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)
ARCHS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]


def load(n):
    p = os.path.join(RES, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def curves():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, arch in zip(ax, ARCHS):
        for mode, c in (("baseline", "tab:red"), ("kd", "tab:green")):
            r = load(f"{mode}_{arch}")
            if r:
                a.plot([h["test_acc"] for h in r["history"]], c=c, label=f"{mode} ({r['test_acc']:.2f}%)")
        a.set_title(arch); a.set_xlabel("epoch"); a.set_ylabel("CIFAR-100 test acc (%)"); a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1_student_curves.png"), dpi=130); plt.close()
    t = load("teacher_resnet50")
    if t:
        plt.figure(figsize=(5, 3.5)); plt.plot([h["test_acc"] for h in t["history"]]); plt.grid(alpha=.3)
        plt.title(f"ResNet-50 teacher ({t['test_acc']:.2f}%)"); plt.xlabel("epoch"); plt.ylabel("test acc (%)")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1_teacher_curve.png"), dpi=130); plt.close()


def kd_bars():
    rows = []
    for arch in ARCHS:
        b, k = load(f"baseline_{arch}"), load(f"kd_{arch}")
        if b and k:
            rows.append((arch, b["test_acc"], k["test_acc"]))
    if not rows:
        return
    plt.figure(figsize=(6, 4))
    for i, (n, b, k) in enumerate(rows):
        plt.bar(i - .2, b, .4, color="tab:red", label="baseline" if i == 0 else None)
        plt.bar(i + .2, k, .4, color="tab:green", label="KD" if i == 0 else None)
        plt.text(i - .2, b + .2, f"{b:.1f}", ha="center"); plt.text(i + .2, k + .2, f"{k:.1f}", ha="center")
    plt.xticks(range(len(rows)), [r[0] for r in rows]); plt.ylabel("CIFAR-100 test acc (%)"); plt.legend()
    plt.ylim(min(min(r[1:]) for r in rows) - 5, max(max(r[1:]) for r in rows) + 3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1_baseline_vs_kd.png"), dpi=130); plt.close()


def nas():
    p = os.path.join(RES, "nas_history.csv")
    if not os.path.exists(p):
        return
    rows = list(csv.DictReader(open(p)))
    acc = [r for r in rows if r["status"] == "accepted"]
    rej = [r for r in rows if r["status"] == "rejected"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    sc = ax[0].scatter([float(r["latency_ms"]) for r in acc], [float(r["val_acc"]) for r in acc],
                       c=[int(r["generation"]) for r in acc], cmap="viridis")
    plt.colorbar(sc, ax=ax[0], label="generation")
    ax[0].set_xlabel("CPU latency (ms, 1 thread)"); ax[0].set_ylabel("proxy val acc (%)"); ax[0].set_title("Accuracy vs latency"); ax[0].grid(alpha=.3)
    ax[1].scatter([float(r["model_size_mb"]) for r in acc], [float(r["val_acc"]) for r in acc], c="tab:blue")
    ax[1].set_xlabel("model size (MB)"); ax[1].set_ylabel("proxy val acc (%)"); ax[1].set_title("Accuracy vs size"); ax[1].grid(alpha=.3)
    gens = sorted({int(r["generation"]) for r in acc})
    best, cur = [], 0
    for g in gens:
        cur = max([cur] + [float(r["val_acc"]) for r in acc if int(r["generation"]) == g]); best.append(cur)
    ax[2].plot(gens, best, "o-"); ax[2].set_xlabel("generation"); ax[2].set_ylabel("best val acc so far (%)")
    ax[2].set_title(f"Search progress ({len(acc)} trained, {len(rej)} rejected)"); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "q2_nas_search.png"), dpi=130); plt.close()
    t = load("nas_best_training")
    if t:
        plt.figure(figsize=(5, 3.5)); plt.plot([h["test_acc"] for h in t["history"]]); plt.grid(alpha=.3)
        plt.title(f"NAS-best, CIFAR-10 ({t['test_acc']:.2f}%)"); plt.xlabel("epoch"); plt.ylabel("test acc (%)")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "q2_best_curve.png"), dpi=130); plt.close()


if __name__ == "__main__":
    curves(); kd_bars(); nas()
    print("figures written to", FIG)
