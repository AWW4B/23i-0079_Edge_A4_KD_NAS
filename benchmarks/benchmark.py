"""Automated benchmark of every exported ONNX model on CPU (ONNX Runtime).

For each model in onnx_models/: test accuracy (full test set, ONNX Runtime), CPU latency (batch 1, 1 and 4 threads),
throughput (batch 64), model size, parameter count and peak inference memory. Writes results/benchmark_results.csv
and a Markdown table results/benchmark_results.md.

    python benchmarks/benchmark.py
"""
import csv, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import onnx
import onnxruntime as ort
from common.prepare_data import load_numpy

ONNX_DIR, RES_DIR = os.path.join(ROOT, "onnx_models"), os.path.join(ROOT, "results")
STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}


def dataset_of(name):
    return "cifar10" if name.startswith("nas") else "cifar100"


def load_test(name):
    xr, y = load_numpy(name, False)
    mean, std = (np.array(s, np.float32).reshape(1, 3, 1, 1) for s in STATS[name])
    x = (xr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0 - mean) / std
    return x.astype(np.float32), y


def accuracy(path, x, y, bs=250):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    correct = 0
    for i in range(0, len(x), bs):
        correct += (sess.run(None, {inp: x[i:i + bs]})[0].argmax(1) == y[i:i + bs]).sum()
    return 100.0 * correct / len(x)


def throughput(path, bs=64, runs=20):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    x = np.random.randn(bs, 3, 32, 32).astype(np.float32)
    for _ in range(3):
        sess.run(None, {inp: x})
    t = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {inp: x})
    return bs * runs / (time.perf_counter() - t)


def worker(path, threads):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "q2_nas", "measure_cpu.py"), path, "--threads", str(threads),
                          "--runs", "200"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    names = sorted(f[:-5] for f in os.listdir(ONNX_DIR) if f.endswith(".onnx"))
    cache, rows = {}, []
    for n in names:
        path = os.path.join(ONNX_DIR, n + ".onnx")
        ds = dataset_of(n)
        if ds not in cache:
            cache[ds] = load_test(ds)
        m = onnx.load(path)
        params = sum(int(np.prod(i.dims)) for i in m.graph.initializer)
        w1, w4 = worker(path, 1), worker(path, 4)
        row = {"model": n, "dataset": ds, "type": n.split("_")[0], "onnx_acc_%": round(float(accuracy(path, *cache[ds])), 2),
               "params_M": round(params / 1e6, 3), "size_MB": round(w1["model_size_mb"], 3),
               "latency_1thread_ms": round(w1["latency_ms"], 3), "latency_p95_1thread_ms": round(w1["latency_p95_ms"], 3),
               "latency_4threads_ms": round(w4["latency_ms"], 3), "throughput_b64_img_s": round(throughput(path), 1),
               "peak_memory_MB": round(w1["peak_memory_mb"], 1)}
        rows.append(row)
        print(row)
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "benchmark_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(RES_DIR, "benchmark_results.md"), "w") as f:
        f.write("| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")
    print("saved results/benchmark_results.csv")


if __name__ == "__main__":
    main()
