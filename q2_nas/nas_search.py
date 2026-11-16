"""Hardware-aware evolutionary Neural Architecture Search on CIFAR-10.

Pipeline for every candidate:
  1. build the network from its configuration (generator.py)
  2. cheap pre-filter on parameter count
  3. export to ONNX and MEASURE real CPU latency / storage size / peak memory (hardware.py)
  4. REJECT the candidate if any deployment constraint is violated (no training budget is wasted on it)
  5. otherwise train it briefly on CIFAR-10 (GPU) and evaluate on a held-out validation split
  6. fitness = val_acc - lat_weight*latency/max_latency - size_weight*size/max_size   (resource-aware)

Evolution: population -> tournament selection -> crossover + mutation -> (mu + lambda) survivor selection.
Outputs (results/): nas_history.csv, nas_history.json, best_architecture.json

    python q2_nas/nas_search.py --population 12 --generations 8 --children 6
"""
import argparse, csv, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from common.data import GPUData
from common.utils import RESULTS, ensure_dirs, save_json, count_params, evaluate
from q2_nas.generator import build_model
from q2_nas.hardware import DEFAULT_CONSTRAINTS, measure_hardware, violations
from q2_nas.search_space import sample_architecture, mutate, crossover, arch_key, space_size


def quick_train(model, train, val, epochs, bs=256, lr=0.1):
    dev = "cuda"
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    scaler = torch.amp.GradScaler()
    for _ in range(epochs):
        model.train()
        for xb, yb in train.batches(bs):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(xb.contiguous(memory_format=torch.channels_last)).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    return evaluate(model, val)


class Searcher:
    def __init__(self, a):
        self.a = a
        self.cons = {"max_latency_ms": a.max_latency, "max_size_mb": a.max_size, "max_peak_memory_mb": a.max_memory,
                     "max_params_m": a.max_params}
        full = GPUData("cifar10", True)
        g = torch.Generator().manual_seed(a.seed)
        perm = torch.randperm(len(full), generator=g).to(full.device)
        self.val = full.view(perm[:5000], False)                       # held-out validation split (never the test set)
        self.train = full.view(perm[5000:5000 + a.train_subset], True)
        self.rng = random.Random(a.seed)
        self.cache, self.history, self.n_eval = {}, [], 0

    def evaluate(self, arch, generation):
        key = arch_key(arch)
        if key in self.cache:
            return self.cache[key]
        self.n_eval += 1
        rec = {"id": self.n_eval, "generation": generation, "arch": arch, "resolution": arch["resolution"],
               "depth_total": sum(s["depth"] for s in arch["stages"]), "n_stages": len(arch["stages"])}
        model = build_model(arch)
        rec["params_m"] = count_params(model) / 1e6
        v = violations(None, self.cons, rec["params_m"])
        hw = None
        if not v:
            hw = measure_hardware(model)
            rec.update(hw)
            v = violations(hw, self.cons)
        rec["violations"] = "; ".join(v)
        if v:
            rec.update(status="rejected", val_acc=None, fitness=None)
            print(f"  #{rec['id']:3d} REJECTED  {rec['violations']}")
        else:
            t0 = time.time()
            acc = quick_train(model, self.train, self.val, self.a.epochs)
            fit = acc - self.a.lat_weight * hw["latency_ms"] / self.cons["max_latency_ms"] \
                      - self.a.size_weight * hw["model_size_mb"] / self.cons["max_size_mb"]
            rec.update(status="accepted", val_acc=acc, fitness=fit, train_time_s=time.time() - t0)
            print(f"  #{rec['id']:3d} acc {acc:5.2f}  lat {hw['latency_ms']:.2f}ms  size {hw['model_size_mb']:.2f}MB  "
                  f"mem {hw['peak_memory_mb']:.1f}MB  fit {fit:.2f}")
        self.history.append(rec)
        self.cache[key] = rec
        self.dump()
        return rec

    def tournament(self, pop, k=3):
        return max(self.rng.sample(pop, min(k, len(pop))), key=lambda r: r["fitness"])

    def dump(self):
        flat = []
        for r in self.history:
            row = {k: v for k, v in r.items() if k != "arch"}
            row["arch"] = arch_key(r["arch"])
            flat.append(row)
        cols = sorted({k for r in flat for k in r}, key=lambda c: (c in ("arch", "violations"), c))
        with open(os.path.join(RESULTS, "nas_history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(flat)
        save_json({"constraints": self.cons, "args": vars(self.a), "history": self.history}, os.path.join(RESULTS, "nas_history.json"))

    def run(self):
        print(f"Search space size ~ {space_size():.2e} architectures; constraints: {self.cons}")
        print("== initial population ==")
        pop, tries = [], 0
        while len(pop) < self.a.population and tries < self.a.population * 40:
            tries += 1
            r = self.evaluate(sample_architecture(self.rng), 0)
            if r["status"] == "accepted" and r not in pop:
                pop.append(r)
        for g in range(1, self.a.generations + 1):
            print(f"== generation {g}/{self.a.generations}  best fitness so far {max(p['fitness'] for p in pop):.2f} ==")
            kids = []
            for _ in range(self.a.children):
                for _try in range(20):
                    p1 = self.tournament(pop)
                    child = crossover(p1["arch"], self.tournament(pop)["arch"], self.rng) if self.rng.random() < 0.3 else p1["arch"]
                    child = mutate(child, self.rng)
                    if arch_key(child) not in self.cache:
                        break
                r = self.evaluate(child, g)
                if r["status"] == "accepted":
                    kids.append(r)
            pop = sorted(pop + kids, key=lambda r: -r["fitness"])[:self.a.population]   # (mu + lambda) selection
        best = max((r for r in self.history if r["status"] == "accepted"), key=lambda r: r["fitness"])
        save_json({"architecture": best["arch"], "val_acc_search": best["val_acc"], "latency_ms": best["latency_ms"],
                   "model_size_mb": best["model_size_mb"], "peak_memory_mb": best["peak_memory_mb"],
                   "params_m": best["params_m"], "fitness": best["fitness"], "constraints": self.cons},
                  os.path.join(RESULTS, "best_architecture.json"))
        n_rej = sum(r["status"] == "rejected" for r in self.history)
        print(f"\nBest: acc {best['val_acc']:.2f}  lat {best['latency_ms']:.2f}ms  size {best['model_size_mb']:.2f}MB "
              f"| evaluated {len(self.history)} (rejected {n_rej}) -> results/best_architecture.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=5, help="proxy-training epochs per candidate")
    ap.add_argument("--train-subset", type=int, default=20000)
    ap.add_argument("--max-latency", type=float, default=DEFAULT_CONSTRAINTS["max_latency_ms"])
    ap.add_argument("--max-size", type=float, default=DEFAULT_CONSTRAINTS["max_size_mb"])
    ap.add_argument("--max-memory", type=float, default=DEFAULT_CONSTRAINTS["max_peak_memory_mb"])
    ap.add_argument("--max-params", type=float, default=DEFAULT_CONSTRAINTS["max_params_m"])
    ap.add_argument("--lat-weight", type=float, default=3.0)
    ap.add_argument("--size-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ensure_dirs()
    Searcher(a).run()
