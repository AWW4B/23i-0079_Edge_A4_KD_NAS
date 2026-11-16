"""Stand-alone CPU measurement worker (ONNX Runtime). Runs in its own process so that the memory it reports is
the *inference* memory of this model only (not the training process' memory).

    python q2_nas/measure_cpu.py model.onnx [--threads 1] [--runs 100]

Prints one JSON line: latency (ms, median/mean/p95, batch=1), peak inference memory (MB), model size (MB).
Peak memory = peak working set (Windows) / max RSS (Linux, macOS) reached during inference minus the baseline
measured right after importing onnxruntime.
"""
import argparse, json, os, sys, time
import numpy as np


def _peak_mb():
    import psutil
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)          # Windows
    if peak is None:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        except Exception:
            peak = mi.rss
    return peak / 1e6


def measure(path, threads=1, runs=100, warmup=20, shape=(1, 3, 32, 32)):
    import onnxruntime as ort
    base = _peak_mb()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(*shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"latency_ms": float(np.median(ts)), "latency_mean_ms": float(ts.mean()), "latency_p95_ms": float(np.percentile(ts, 95)),
            "peak_memory_mb": float(max(_peak_mb() - base, 0.0)), "model_size_mb": os.path.getsize(path) / 1e6,
            "threads": threads}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=100)
    a = ap.parse_args()
    print(json.dumps(measure(a.onnx, a.threads, a.runs)))
