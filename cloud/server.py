"""
Small HTTP server for the neural HLR model, standing in for a cloud prediction
API. client_benchmark.py calls it over loopback to measure what JSON and HTTP
add on top of inference, and compose_estimate.py adds published network
round-trip times to that.

Run: python cloud/server.py [--port 8765]
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark'))
from mlx_neural_hlr import MLXNeuralHLRModel  # noqa: E402

model = None


class PredictHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'  # keep-alive, as a real API client would use

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        req = json.loads(body)

        start = time.perf_counter()
        p, h = model.forward(
            mx.array(req['right'], dtype=mx.float32),
            mx.array(req['wrong'], dtype=mx.float32),
            mx.array(req['t'], dtype=mx.float32),
            mx.array(req['lexeme_idx'], dtype=mx.uint32),
        )
        inference_ms = (time.perf_counter() - start) * 1000

        payload = json.dumps({
            'p': p.tolist(),
            'h': h.tolist(),
            'server_inference_ms': inference_ms,
        }).encode()

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass  # per-request logging to stderr would distort the timings


def main():
    global model
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--port', type=int, default=8765)
    args = argparser.parse_args()

    model = MLXNeuralHLRModel()
    # one warm-up so the first real request doesn't pay lazy-init cost
    model.forward(
        mx.array([1.0], dtype=mx.float32), mx.array([1.0], dtype=mx.float32),
        mx.array([1.0], dtype=mx.float32), mx.array([0], dtype=mx.uint32),
    )

    server = ThreadingHTTPServer(('127.0.0.1', args.port), PredictHandler)
    print(f'serving neural HLR on http://127.0.0.1:{args.port}/predict', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
