"""Run the complete pipeline (both questions) end-to-end, logging each step to logs/.

    python run_all.py                 # everything
    python run_all.py --only q1       # or q2 / bench / report
    python run_all.py --quick         # tiny smoke-test settings (a few minutes, for checking the code runs)
"""
import argparse, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(name, args):
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log = os.path.join(ROOT, "logs", name + ".log")
    print(f"\n=== {name}: python {' '.join(args)}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, "-u", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line); f.flush()
            if not line.startswith("epoch") or int(line.split()[1].split("/")[0]) % 5 == 0:  # console: every 5th epoch
                print("   ", line.rstrip(), flush=True)
        p.wait()
    print(f"=== {name} finished in {(time.time() - t0) / 60:.1f} min (exit {p.returncode}) -> {log}", flush=True)
    if p.returncode:
        sys.exit(p.returncode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["q1", "q2", "bench", "report"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    E = (lambda full, quick: str(quick if a.quick else full))
    if a.only in (None, "q1"):
        run("q1_teacher", ["q1_kd/train_teacher.py", "--epochs", E(20, 1)])
        run("q1_kd_mobilenetv2", ["q1_kd/train_kd.py", "--arch", "mobilenetv2", "--epochs", E(15, 1)])
        run("q1_kd_efficientnet_lite0", ["q1_kd/train_kd.py", "--arch", "efficientnet_lite0", "--epochs", E(15, 1)])
        # SqueezeNet is tiny (0.77M params): a light feature-loss weight works better (see report ablation)
        run("q1_kd_squeezenet", ["q1_kd/train_kd.py", "--arch", "squeezenet", "--epochs", E(15, 1), "--beta", "0.1"])
        run("q1_baseline", ["q1_kd/train_baseline.py", "--arch", "all", "--epochs", E(15, 1)])
        run("q1_export_onnx", ["q1_kd/export_onnx.py"])
    if a.only in (None, "q2"):
        if a.quick:
            run("q2_search", ["q2_nas/nas_search.py", "--population", "4", "--generations", "1", "--children", "2",
                              "--epochs", "1", "--train-subset", "5000"])
            run("q2_train_best", ["q2_nas/train_best.py", "--epochs", "1"])
        else:
            run("q2_search", ["q2_nas/nas_search.py", "--population", "10", "--generations", "5", "--children", "5", "--epochs", "3", "--train-subset", "15000"])
            run("q2_train_best", ["q2_nas/train_best.py", "--epochs", "30"])
    if a.only in (None, "bench"):
        run("benchmark", ["benchmarks/benchmark.py"])
    if a.only in (None, "report"):
        run("figures", ["report/make_figures.py"])
