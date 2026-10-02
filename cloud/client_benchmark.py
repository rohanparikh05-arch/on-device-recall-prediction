"""
Time requests to the inference server over loopback. Loopback adds almost no
network delay, so this measures the JSON, HTTP and process overhead a cloud
call pays on top of inference. compose_estimate.py adds real network
round-trip times to these numbers.

Start the server first:
    python cloud/server.py
then:
    python cloud/client_benchmark.py
"""

import argparse
import http.client
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark'))
from mlx_neural_hlr import VOCAB_FILE  # noqa: E402

N_WARMUP = 20
N_TRIALS = 300
BATCH_SIZES = [1, 32, 256, 1024]
RESULTS_OUT = os.path.join(os.path.dirname(__file__), 'loopback_results.json')


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p / 100))]


def make_payload(batch_size, vocab_size):
    return json.dumps({
        'right': [random.randint(0, 20) for _ in range(batch_size)],
        'wrong': [random.randint(0, 20) for _ in range(batch_size)],
        't': [random.uniform(0.01, 30.0) for _ in range(batch_size)],
        'lexeme_idx': [random.randrange(vocab_size) for _ in range(batch_size)],
    }).encode()


def bench_batch(conn, payload, n_trials):
    headers = {'Content-Type': 'application/json'}
    round_trip_ms = []
    server_ms = []
    for _ in range(n_trials):
        start = time.perf_counter()
        conn.request('POST', '/predict', body=payload, headers=headers)
        resp = json.loads(conn.getresponse().read())
        round_trip_ms.append((time.perf_counter() - start) * 1000)
        server_ms.append(resp['server_inference_ms'])
    return round_trip_ms, server_ms


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--port', type=int, default=8765)
    args = argparser.parse_args()

    with open(VOCAB_FILE) as f:
        vocab_size = len(json.load(f)) + 1

    conn = http.client.HTTPConnection('127.0.0.1', args.port)
    results = {}

    print(f'{"batch":>6}  {"payload_kb":>10}  {"rtt_mean":>9}  {"rtt_med":>9}  {"rtt_p95":>9}  '
          f'{"rtt_p99":>9}  {"srv_infer":>9}  {"overhead":>9}')
    for batch_size in BATCH_SIZES:
        payload = make_payload(batch_size, vocab_size)
        bench_batch(conn, payload, N_WARMUP)
        round_trip_ms, server_ms = bench_batch(conn, payload, N_TRIALS)

        mean_rtt = statistics.mean(round_trip_ms)
        mean_srv = statistics.mean(server_ms)
        results[batch_size] = {
            'payload_bytes': len(payload),
            'loopback_mean_ms': mean_rtt,
            'loopback_median_ms': statistics.median(round_trip_ms),
            'loopback_p95_ms': percentile(round_trip_ms, 95),
            'loopback_p99_ms': percentile(round_trip_ms, 99),
            'server_inference_mean_ms': mean_srv,
            'overhead_mean_ms': mean_rtt - mean_srv,
        }
        r = results[batch_size]
        print(f'{batch_size:>6}  {len(payload) / 1024:>10.1f}  {mean_rtt:>9.4f}  '
              f'{r["loopback_median_ms"]:>9.4f}  {r["loopback_p95_ms"]:>9.4f}  '
              f'{r["loopback_p99_ms"]:>9.4f}  {mean_srv:>9.4f}  {r["overhead_mean_ms"]:>9.4f}')

    conn.close()
    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved {RESULTS_OUT}')


if __name__ == '__main__':
    main()
