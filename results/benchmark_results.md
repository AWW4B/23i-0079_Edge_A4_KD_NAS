| model | dataset | type | onnx_acc_% | params_M | size_MB | latency_1thread_ms | latency_p95_1thread_ms | latency_4threads_ms | throughput_b64_img_s | peak_memory_MB |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_efficientnet_lite0 | cifar100 | baseline | 65.13 | 3.478 | 13.955 | 1.833 | 2.211 | 0.919 | 1570.2 | 42.3 |
| baseline_mobilenetv2 | cifar100 | baseline | 64.54 | 2.335 | 9.385 | 1.337 | 1.623 | 0.712 | 2177.4 | 32.6 |
| baseline_squeezenet | cifar100 | baseline | 57.32 | 0.774 | 3.11 | 0.375 | 0.44 | 0.304 | 6869.5 | 16.3 |
| kd_efficientnet_lite0 | cifar100 | kd | 67.4 | 3.478 | 13.955 | 1.841 | 2.366 | 0.859 | 1680.9 | 42.3 |
| kd_mobilenetv2 | cifar100 | kd | 66.2 | 2.335 | 9.385 | 1.318 | 1.554 | 0.727 | 2072.7 | 32.8 |
| kd_squeezenet | cifar100 | kd | 60.75 | 0.774 | 3.11 | 0.378 | 0.449 | 0.259 | 6853.6 | 15.7 |
| nas_best | cifar10 | nas | 85.38 | 0.053 | 0.219 | 0.382 | 0.402 | 0.203 | 6496.5 | 12.7 |
