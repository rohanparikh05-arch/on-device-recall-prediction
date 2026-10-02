"""
Score the linear baseline (weights trained by halflife-regression/experiment.py)
and the neural model on the same user-grouped test set. Scoring both on one
test set separates the effect of the model change from the effect of the split.
"""

import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark'))
from load_hlr_weights import load_weights as load_linear_weights  # noqa: E402
from mlx_hlr import MLXHLRModel  # noqa: E402
from mlx_neural_hlr import MLXNeuralHLRModel  # noqa: E402

from data_split import load_split  # noqa: E402
from metrics import evaluate  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
LINEAR_WEIGHTS = os.path.join(REPO_ROOT, 'halflife-regression', 'results',
                               'hlr.settles.acl16.learning_traces.13m.weights')


def main():
    _, test, _ = load_split(DATA_FILE)  # full data, same deterministic user split
    right = mx.array(test['right'], dtype=mx.float32)
    wrong = mx.array(test['wrong'], dtype=mx.float32)
    t = mx.array(test['t'], dtype=mx.float32)

    linear_weights = load_linear_weights(LINEAR_WEIGHTS)
    linear_model = MLXHLRModel(linear_weights)
    lin_idx = linear_model.lexeme_indices(test['lexeme_key'])
    lin_p, lin_h = linear_model.forward(right, wrong, t, lin_idx)
    lin_metrics = evaluate(test['p'], lin_p.tolist(), test['h'], lin_h.tolist())

    neural_model = MLXNeuralHLRModel()
    neu_idx = neural_model.lexeme_indices(test['lexeme_key'])
    neu_p, neu_h = neural_model.forward(right, wrong, t, neu_idx)
    neu_metrics = evaluate(test['p'], neu_p.tolist(), test['h'], neu_h.tolist())

    print(f'n = {len(test["p"])} (same user-grouped test set for both)')
    print('linear HLR (baseline weights, trained on row-order split):', lin_metrics)
    print('neural HLR (trained on this user-grouped split):          ', neu_metrics)


if __name__ == '__main__':
    main()
