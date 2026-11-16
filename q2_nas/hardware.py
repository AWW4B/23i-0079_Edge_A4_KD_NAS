"""Hardware-aware evaluation: measures ACTUAL CPU latency, storage size and peak inference memory of a candidate
and checks it against the deployment constraints."""
import json, os, subprocess, sys, tempfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# Deployment constraints for the target edge device (CPU, single thread, batch size 1)
DEFAULT_CONSTRAINTS = {
    "max_latency_ms": 3.0,       # median ONNX-Runtime CPU latency
    "max_size_mb": 1.0,          # ONNX file size on disk (fp32)
    "max_peak_memory_mb": 20.0,  # peak inference memory growth over the bare ORT process
    "max_params_m": 0.5,         # million parameters (cheap pre-filter before any export)
}


def export_onnx(model, path, opset=17):
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"], opset_version=opset,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)


def measure_hardware(model, threads=1, runs=100):
    """Export to ONNX in a temp file and measure it in a fresh process (clean memory + latency numbers)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cand.onnx")
        export_onnx(model, path)
        out = subprocess.run([sys.executable, os.path.join(HERE, "measure_cpu.py"), path, "--threads", str(threads),
                              "--runs", str(runs)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def violations(hw, constraints, params_m=None):
    """Return a list of violated constraints (empty list => architecture is feasible)."""
    v = []
    if params_m is not None and params_m > constraints["max_params_m"]:
        v.append(f"params {params_m:.2f}M > {constraints['max_params_m']}M")
    if hw is not None:
        if hw["latency_ms"] > constraints["max_latency_ms"]:
            v.append(f"latency {hw['latency_ms']:.2f}ms > {constraints['max_latency_ms']}ms")
        if hw["model_size_mb"] > constraints["max_size_mb"]:
            v.append(f"size {hw['model_size_mb']:.2f}MB > {constraints['max_size_mb']}MB")
        if hw["peak_memory_mb"] > constraints["max_peak_memory_mb"]:
            v.append(f"memory {hw['peak_memory_mb']:.1f}MB > {constraints['max_peak_memory_mb']}MB")
    return v
