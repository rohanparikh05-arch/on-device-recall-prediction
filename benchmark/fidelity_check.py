"""
Check that the MLX version of the linear HLR model gives the same predictions
as the original experiment.py model on real rows, within TOLERANCE.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'halflife-regression'))
import experiment  # noqa: E402

from load_hlr_weights import load_weights  # noqa: E402
from mlx_hlr import MLXHLRModel  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
WEIGHTS_FILE = os.path.join(REPO_ROOT, 'halflife-regression', 'results',
                             'hlr.settles.acl16.learning_traces.13m.weights')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
SAMPLE_SIZE = 5000
TOLERANCE = 1e-4


def main():
    weights = load_weights(WEIGHTS_FILE)

    ref_model = experiment.SpacedRepetitionModel(method='hlr')
    for k, v in weights['raw'].items():
        ref_model.weights[k] = v

    _, testset = experiment.read_data(DATA_FILE, 'hlr', max_lines=SAMPLE_SIZE)
    sample = testset if len(testset) < 1000 else testset[:1000]
    print(f'fidelity-checking on {len(sample)} real test instances...')

    mlx_model = MLXHLRModel(weights)

    rights, wrongs, ts, lexeme_keys = [], [], [], []
    ref_p, ref_h = [], []
    for inst in sample:
        p, h = ref_model.predict(inst)
        ref_p.append(p)
        ref_h.append(h)
        fv = dict(inst.fv)
        # invert the sqrt(1+right)/sqrt(1+wrong) feature transform back to raw counts
        rights.append(fv['right'] ** 2 - 1)
        wrongs.append(fv['wrong'] ** 2 - 1)
        ts.append(inst.t)
        lexeme_keys.append(inst.fv[-1][0])

    import mlx.core as mx
    lexeme_idx = mlx_model.lexeme_indices(lexeme_keys)
    mlx_p, mlx_h = mlx_model.forward(
        mx.array(rights, dtype=mx.float32),
        mx.array(wrongs, dtype=mx.float32),
        mx.array(ts, dtype=mx.float32),
        lexeme_idx,
    )
    mlx_p = mlx_p.tolist()
    mlx_h = mlx_h.tolist()

    max_diff_p = max(abs(a - b) for a, b in zip(ref_p, mlx_p))
    max_diff_h = max(abs(a - b) for a, b in zip(ref_h, mlx_h))
    mean_diff_p = sum(abs(a - b) for a, b in zip(ref_p, mlx_p)) / len(sample)

    print(f'max |ref_p - mlx_p|  = {max_diff_p:.8f}')
    print(f'mean |ref_p - mlx_p| = {mean_diff_p:.8f}')
    print(f'max |ref_h - mlx_h|  = {max_diff_h:.8f}')

    ok = max_diff_p < TOLERANCE and max_diff_h < TOLERANCE
    print('PASS' if ok else 'FAIL', f'(tolerance={TOLERANCE})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
