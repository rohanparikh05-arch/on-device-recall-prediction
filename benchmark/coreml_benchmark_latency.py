"""On-device inference latency for the Core ML HLR model: single-instance and batched."""

import os
import statistics
import sys
import time

import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
from load_hlr_weights import load_weights  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
WEIGHTS_FILE = os.path.join(REPO_ROOT, 'halflife-regression', 'results',
                             'hlr.settles.acl16.learning_traces.13m.weights')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'HLR.mlpackage')

N_WARMUP = 20
N_TRIALS = 500
BATCH_SIZES = [1, 32, 256, 1024]


def percentile(values, p):
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * p / 100))
    return values[idx]


def bench_batch(mlmodel, lexeme_vocab_size, batch_size, n_trials):
    inputs = {
        'right': np.random.randint(0, 20, size=batch_size).astype(np.float32),
        'wrong': np.random.randint(0, 20, size=batch_size).astype(np.float32),
        't': np.random.uniform(0.01, 30.0, size=batch_size).astype(np.float32),
        'lexeme_idx': np.random.randint(0, lexeme_vocab_size, size=batch_size).astype(np.int32),
    }

    for _ in range(N_WARMUP):
        mlmodel.predict(inputs)

    latencies_ms = []
    for _ in range(n_trials):
        start = time.perf_counter()
        mlmodel.predict(inputs)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return latencies_ms


def main():
    weights = load_weights(WEIGHTS_FILE)
    vocab_size = len(weights['lexeme_weights'])
    mlmodel = ct.models.MLModel(MODEL_PATH)

    print(f'{"batch":>6}  {"mean_ms":>10}  {"median_ms":>10}  {"p95_ms":>10}  {"p99_ms":>10}  {"per_item_us":>12}')
    for batch_size in BATCH_SIZES:
        n_trials = N_TRIALS if batch_size <= 32 else N_TRIALS // 4
        latencies_ms = bench_batch(mlmodel, vocab_size, batch_size, n_trials)
        mean_ms = statistics.mean(latencies_ms)
        median_ms = statistics.median(latencies_ms)
        p95_ms = percentile(latencies_ms, 95)
        p99_ms = percentile(latencies_ms, 99)
        per_item_us = (mean_ms * 1000) / batch_size
        print(f'{batch_size:>6}  {mean_ms:>10.4f}  {median_ms:>10.4f}  {p95_ms:>10.4f}  {p99_ms:>10.4f}  {per_item_us:>12.2f}')


if __name__ == '__main__':
    main()
