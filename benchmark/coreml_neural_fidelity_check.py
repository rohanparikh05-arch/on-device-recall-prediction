"""
Check that the Core ML neural model matches a numpy version of the same forward
pass on 2,000 test rows, within TOLERANCE. Run it in the Core ML environment;
numpy stands in for MLX, which is not installed there.
"""

import json
import math
import os
import sys

import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from coreml_neural_hlr import load_mlx_safetensors_as_numpy, WEIGHTS_FILE, VOCAB_FILE  # noqa: E402
from data_split import load_split  # noqa: E402
from metrics import evaluate  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'NeuralHLR.mlpackage')
SAMPLE_LINES = 2_000_000
TOLERANCE = 1e-3

MIN_HALF_LIFE = 15.0 / (24 * 60)
MAX_HALF_LIFE = 274.0
LOG2_MIN_HALF_LIFE = math.log2(MIN_HALF_LIFE)
LOG2_MAX_HALF_LIFE = math.log2(MAX_HALF_LIFE)


def numpy_reference_forward(weights, right, wrong, t, lexeme_idx):
    # numpy copy of the forward pass in model/neural_hlr.py
    right_feat = np.sqrt(1.0 + right)
    wrong_feat = np.sqrt(1.0 + wrong)
    emb = weights['lexeme_emb.weight'][lexeme_idx]
    x = np.concatenate([right_feat[:, None], wrong_feat[:, None], emb], axis=1)
    h1 = np.maximum(0.0, x @ weights['fc1.weight'].T + weights['fc1.bias'])
    dp = (h1 @ weights['fc2.weight'].T + weights['fc2.bias'])[:, 0]
    dp = np.clip(dp, LOG2_MIN_HALF_LIFE, LOG2_MAX_HALF_LIFE)
    h = 2.0 ** dp
    p = np.clip(2.0 ** (-t / h), 0.0001, 0.9999)
    return p, h


def main():
    weights = load_mlx_safetensors_as_numpy(WEIGHTS_FILE)
    with open(VOCAB_FILE) as f:
        vocab = json.load(f)
    oov_index = len(vocab)

    _, test, _ = load_split(DATA_FILE, max_lines=SAMPLE_LINES)
    sample = {k: v[:2000] for k, v in test.items()}
    right = np.array(sample['right'], dtype=np.float32)
    wrong = np.array(sample['wrong'], dtype=np.float32)
    t = np.array(sample['t'], dtype=np.float32)
    lexeme_idx = np.array([vocab.get(k, oov_index) for k in sample['lexeme_key']], dtype=np.int32)

    ref_p, ref_h = numpy_reference_forward(weights, right, wrong, t, lexeme_idx)

    mlmodel = ct.models.MLModel(MODEL_PATH)
    out = mlmodel.predict({'right': right, 'wrong': wrong, 't': t, 'lexeme_idx': lexeme_idx})
    coreml_p, coreml_h = out['p'], out['h']

    max_diff_p = float(np.max(np.abs(ref_p - coreml_p)))
    max_diff_h = float(np.max(np.abs(ref_h - coreml_h)))
    print(f'max |ref_p - coreml_p| = {max_diff_p:.8f}')
    print(f'max |ref_h - coreml_h| = {max_diff_h:.8f}')

    metrics_coreml = evaluate(sample['p'], coreml_p.tolist(), sample['h'], coreml_h.tolist())
    print('metrics on this 2000-row sample via Core ML:', metrics_coreml)

    ok = max_diff_p < TOLERANCE and max_diff_h < TOLERANCE
    print('PASS' if ok else 'FAIL', f'(tolerance={TOLERANCE})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
