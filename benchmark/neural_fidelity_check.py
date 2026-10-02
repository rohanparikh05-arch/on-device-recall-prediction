"""
Reload the trained neural model from disk through the benchmark wrapper and
print its metrics on the test users in the first 2M rows. This checks that the
saved weights and vocab load correctly. The numbers come from a subsample, so
they differ slightly from the full test set metrics in model/eval_metrics.json.
"""

import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from mlx_neural_hlr import MLXNeuralHLRModel  # noqa: E402
from data_split import load_split  # noqa: E402
from metrics import evaluate  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
SAMPLE_LINES = 2_000_000


def main():
    model = MLXNeuralHLRModel()

    _, test, _ = load_split(DATA_FILE, max_lines=SAMPLE_LINES)
    lexeme_idx = model.lexeme_indices(test['lexeme_key'])
    p_pred, h_pred = model.forward(
        mx.array(test['right'], dtype=mx.float32),
        mx.array(test['wrong'], dtype=mx.float32),
        mx.array(test['t'], dtype=mx.float32),
        lexeme_idx,
    )
    metrics = evaluate(test['p'], p_pred.tolist(), test['h'], h_pred.tolist())
    print(f'n = {len(test["p"])}')
    print(metrics)


if __name__ == '__main__':
    main()
