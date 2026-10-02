"""On-device inference latency for the neural HLR model (MLX)."""

import json
import os
import random
import statistics
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
from mlx_neural_hlr import MLXNeuralHLRModel  # noqa: E402

N_WARMUP = 50
N_TRIALS = 2000
BATCH_SIZES = [1, 32, 256, 1024]
RESULTS_OUT = os.path.join(os.path.dirname(__file__), 'neural_mlx_results.json')


def percentile(values, p):
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * p / 100))
    return values[idx]


def bench_batch(model, batch_size, n_trials):
    lexeme_keys = list(model.lexeme_vocab.keys())
    rights = mx.array([random.randint(0, 20) for _ in range(batch_size)], dtype=mx.float32)
    wrongs = mx.array([random.randint(0, 20) for _ in range(batch_size)], dtype=mx.float32)
    ts = mx.array([random.uniform(0.01, 30.0) for _ in range(batch_size)], dtype=mx.float32)
    lexeme_idx = model.lexeme_indices(random.choices(lexeme_keys, k=batch_size))

    for _ in range(N_WARMUP):
        model.forward(rights, wrongs, ts, lexeme_idx)

    latencies_ms = []
    for _ in range(n_trials):
        start = time.perf_counter()
        model.forward(rights, wrongs, ts, lexeme_idx)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return latencies_ms


def main():
    model = MLXNeuralHLRModel()

    print(f'device: {mx.default_device()}')
    print(f'{"batch":>6}  {"mean_ms":>10}  {"median_ms":>10}  {"p95_ms":>10}  {"p99_ms":>10}  {"per_item_us":>12}')
    results = {}
    for batch_size in BATCH_SIZES:
        n_trials = N_TRIALS if batch_size <= 32 else N_TRIALS // 4
        latencies_ms = bench_batch(model, batch_size, n_trials)
        mean_ms = statistics.mean(latencies_ms)
        median_ms = statistics.median(latencies_ms)
        p95_ms = percentile(latencies_ms, 95)
        p99_ms = percentile(latencies_ms, 99)
        per_item_us = (mean_ms * 1000) / batch_size
        results[batch_size] = {'mean_ms': mean_ms, 'median_ms': median_ms,
                               'p95_ms': p95_ms, 'p99_ms': p99_ms, 'per_item_us': per_item_us}
        print(f'{batch_size:>6}  {mean_ms:>10.4f}  {median_ms:>10.4f}  {p95_ms:>10.4f}  {p99_ms:>10.4f}  {per_item_us:>12.2f}')

    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'saved {RESULTS_OUT}')


if __name__ == '__main__':
    main()
