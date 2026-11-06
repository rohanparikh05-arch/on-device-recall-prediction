"""
Build iOS versions of both Core ML models and write the rows the phone uses to
check its predictions. Run it in the Core ML environment.
"""

import json
import os
import sys

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(HERE, '..')
REPO_ROOT = os.path.join(BENCHMARK, '..')
sys.path.insert(0, BENCHMARK)
sys.path.insert(0, os.path.join(REPO_ROOT, 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'halflife-regression'))
import experiment  # noqa: E402
import coreml_hlr  # noqa: E402
import coreml_neural_hlr  # noqa: E402
from coreml_neural_fidelity_check import numpy_reference_forward  # noqa: E402
from data_split import load_split  # noqa: E402
from load_hlr_weights import load_weights  # noqa: E402

DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
LINEAR_WEIGHTS = os.path.join(REPO_ROOT, 'halflife-regression', 'results', 'hlr.user_split.weights')
OUT_DIR = os.path.join(HERE, 'HLRBench')


def use_exp2():
    # On iOS 26.6 (iPhone 17, CPU and default compute units, and the simulator's
    # CPU path), Core ML ran pow(x=2.0, y=d) as d**2. The builders in
    # coreml_hlr.py and coreml_neural_hlr.py only use pow with a base of 2, so
    # swap in exp2 for the phone without changing what the laptop models use.
    original_pow = mb.pow

    def pow_base_two(x, y, **kwargs):
        # the converter calls pow internally too, so only touch a literal 2.0
        if isinstance(x, np.floating) and float(x) == 2.0:
            return mb.exp2(x=y, **kwargs)
        return original_pow(x=x, y=y, **kwargs)
    mb.pow = pow_base_two


def convert(prog):
    return ct.convert(
        prog,
        source='milinternal',
        convert_to='mlprogram',
        inputs=[
            ct.TensorType(name='right', shape=ct.Shape(shape=(ct.RangeDim(1, 4096),))),
            ct.TensorType(name='wrong', shape=ct.Shape(shape=(ct.RangeDim(1, 4096),))),
            ct.TensorType(name='t', shape=ct.Shape(shape=(ct.RangeDim(1, 4096),))),
            ct.TensorType(name='lexeme_idx', shape=ct.Shape(shape=(ct.RangeDim(1, 4096),)), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name='p'), ct.TensorType(name='h')],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,
    )


def main():
    neural_weights = coreml_neural_hlr.load_mlx_safetensors_as_numpy(coreml_neural_hlr.WEIGHTS_FILE)
    with open(coreml_neural_hlr.VOCAB_FILE) as f:
        neural_vocab = json.load(f)
    linear_weights = load_weights(LINEAR_WEIGHTS)

    use_exp2()
    convert(coreml_neural_hlr.build_program(neural_weights)).save(os.path.join(OUT_DIR, 'NeuralHLR.mlpackage'))
    convert(coreml_hlr.build_program(linear_weights)).save(os.path.join(OUT_DIR, 'LinearHLR.mlpackage'))

    # the same 2,000 rows as coreml_neural_fidelity_check.py
    _, test, _ = load_split(DATA_FILE, max_lines=2_000_000)
    right = np.array(test['right'][:2000], dtype=np.float32)
    wrong = np.array(test['wrong'][:2000], dtype=np.float32)
    t = np.array(test['t'][:2000], dtype=np.float32)
    keys = test['lexeme_key'][:2000]

    neural_idx = np.array([neural_vocab.get(k, len(neural_vocab)) for k in keys], dtype=np.int32)
    neural_p, neural_h = numpy_reference_forward(neural_weights, right, wrong, t, neural_idx)

    # the linear reference is experiment.py's own predict()
    reference = experiment.SpacedRepetitionModel(method='hlr')
    for k, v in linear_weights['raw'].items():
        reference.weights[k] = v
    linear_idx, linear_p, linear_h = [], [], []
    for r, w, days, key in zip(test['right'][:2000], test['wrong'][:2000], test['t'][:2000], keys):
        fv = [('right', (1 + r) ** 0.5), ('wrong', (1 + w) ** 0.5), ('bias', 1.0), (key, 1.0)]
        inst = experiment.Instance(0.5, days, fv, 1.0, 0.5, '', 0, 0, 0, '', '')
        p, h = reference.predict(inst)
        linear_p.append(p)
        linear_h.append(h)
        linear_idx.append(linear_weights['lexeme_vocab'].get(key, linear_weights['oov_index']))

    rows = {
        'right': right.tolist(),
        'wrong': wrong.tolist(),
        't': t.tolist(),
        'neural': {'idx': neural_idx.tolist(), 'p': neural_p.tolist(), 'h': neural_h.tolist(),
                   'table_size': len(neural_vocab) + 1},
        'linear': {'idx': linear_idx, 'p': linear_p, 'h': linear_h,
                   'table_size': len(linear_weights['lexeme_weights'])},
    }
    with open(os.path.join(OUT_DIR, 'fidelity_rows.json'), 'w') as f:
        json.dump(rows, f)
    print('wrote models and fidelity_rows.json to', OUT_DIR)


if __name__ == '__main__':
    main()
