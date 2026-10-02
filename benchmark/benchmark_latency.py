"""On-device inference latency for the MLX HLR model: single-instance and batched."""

import os
import statistics
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
from load_hlr_weights import load_weights  # noqa: E402
from mlx_hlr import MLXHLRModel  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
WEIGHTS_FILE = os.path.join(REPO_ROOT, 'halflife-regression', 'results',
                             'hlr.settles.acl16.learning_traces.13m.weights')

N_WARMUP = 50
N_TRIALS = 2000
BATCH_SIZES = [1, 32, 256, 1024]


def percentile(values, p):
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * p / 100))
    return values[idx]


def bench_batch(model, batch_size, n_trials):
    import random
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
    weights = load_weights(WEIGHTS_FILE)
    model = MLXHLRModel(weights)

    print(f'device: {mx.default_device()}')
    print(f'{"batch":>6}  {"mean_ms":>10}  {"median_ms":>10}  {"p95_ms":>10}  {"p99_ms":>10}  {"per_item_us":>12}')
    for batch_size in BATCH_SIZES:
        n_trials = N_TRIALS if batch_size <= 32 else N_TRIALS // 4
        latencies_ms = bench_batch(model, batch_size, n_trials)
        mean_ms = statistics.mean(latencies_ms)
        median_ms = statistics.median(latencies_ms)
        p95_ms = percentile(latencies_ms, 95)
        p99_ms = percentile(latencies_ms, 99)
        per_item_us = (mean_ms * 1000) / batch_size
        print(f'{batch_size:>6}  {mean_ms:>10.4f}  {median_ms:>10.4f}  {p95_ms:>10.4f}  {p99_ms:>10.4f}  {per_item_us:>12.2f}')


if __name__ == '__main__':
    main()
