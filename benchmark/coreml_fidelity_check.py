"""
Check that the Core ML linear HLR model gives the same predictions as the
original experiment.py model on real rows, within TOLERANCE. Run it in the
Core ML environment.
"""

import os
import sys

import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'halflife-regression'))
import experiment  # noqa: E402

from load_hlr_weights import load_weights  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
WEIGHTS_FILE = os.path.join(REPO_ROOT, 'halflife-regression', 'results',
                             'hlr.settles.acl16.learning_traces.13m.weights')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'HLR.mlpackage')
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

    rights, wrongs, ts, lexeme_idx = [], [], [], []
    ref_p, ref_h = [], []
    for inst in sample:
        p, h = ref_model.predict(inst)
        ref_p.append(p)
        ref_h.append(h)
        fv = dict(inst.fv)
        rights.append(fv['right'] ** 2 - 1)
        wrongs.append(fv['wrong'] ** 2 - 1)
        ts.append(inst.t)
        key = inst.fv[-1][0]
        lexeme_idx.append(weights['lexeme_vocab'].get(key, weights['oov_index']))

    mlmodel = ct.models.MLModel(MODEL_PATH)
    out = mlmodel.predict({
        'right': np.array(rights, dtype=np.float32),
        'wrong': np.array(wrongs, dtype=np.float32),
        't': np.array(ts, dtype=np.float32),
        'lexeme_idx': np.array(lexeme_idx, dtype=np.int32),
    })
    coreml_p = out['p']
    coreml_h = out['h']

    max_diff_p = float(np.max(np.abs(np.array(ref_p) - coreml_p)))
    mean_diff_p = float(np.mean(np.abs(np.array(ref_p) - coreml_p)))
    max_diff_h = float(np.max(np.abs(np.array(ref_h) - coreml_h)))

    print(f'max |ref_p - coreml_p|  = {max_diff_p:.8f}')
    print(f'mean |ref_p - coreml_p| = {mean_diff_p:.8f}')
    print(f'max |ref_h - coreml_h|  = {max_diff_h:.8f}')

    ok = max_diff_p < TOLERANCE and max_diff_h < TOLERANCE
    print('PASS' if ok else 'FAIL', f'(tolerance={TOLERANCE})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
