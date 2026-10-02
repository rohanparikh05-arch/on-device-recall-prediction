"""
Convert both models to Core ML at float16 and at float32 and print the largest
error in h and p against the reference implementation (Table I in the paper).

The linear model is checked against experiment.py on the same rows as
coreml_fidelity_check.py. The neural model is checked against the numpy forward
pass on the same rows as coreml_neural_fidelity_check.py. Run it in the Core ML
environment.
"""

import json
import os
import sys

import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'halflife-regression'))
import experiment  # noqa: E402
import coreml_hlr  # noqa: E402
import coreml_neural_hlr  # noqa: E402
from load_hlr_weights import load_weights  # noqa: E402
from coreml_neural_fidelity_check import numpy_reference_forward  # noqa: E402
from data_split import load_split  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')


def convert(prog, precision):
    inputs = [
        ct.TensorType(name='right', shape=ct.Shape(shape=(ct.RangeDim(1, 4096),))),
        ct.TensorType(name='wrong', shape=ct.Shape(shape=(ct.RangeDim(1, 4096),))),
        ct.TensorType(name='t', shape=ct.Shape(shape=(ct.RangeDim(1, 4096),))),
        ct.TensorType(name='lexeme_idx', shape=ct.Shape(shape=(ct.RangeDim(1, 4096),)), dtype=np.int32),
    ]
    return ct.convert(
        prog,
        source='milinternal',
        convert_to='mlprogram',
        inputs=inputs,
        outputs=[ct.TensorType(name='p'), ct.TensorType(name='h')],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=precision,
    )


def linear_sample():
    weights = load_weights(coreml_hlr.WEIGHTS_FILE)
    ref_model = experiment.SpacedRepetitionModel(method='hlr')
    for k, v in weights['raw'].items():
        ref_model.weights[k] = v

    _, testset = experiment.read_data(DATA_FILE, 'hlr', max_lines=5000)
    sample = testset if len(testset) < 1000 else testset[:1000]

    rights, wrongs, ts, lexeme_idx, ref_p, ref_h = [], [], [], [], [], []
    for inst in sample:
        p, h = ref_model.predict(inst)
        ref_p.append(p)
        ref_h.append(h)
        fv = dict(inst.fv)
        rights.append(fv['right'] ** 2 - 1)
        wrongs.append(fv['wrong'] ** 2 - 1)
        ts.append(inst.t)
        lexeme_idx.append(weights['lexeme_vocab'].get(inst.fv[-1][0], weights['oov_index']))

    inputs = {
        'right': np.array(rights, dtype=np.float32),
        'wrong': np.array(wrongs, dtype=np.float32),
        't': np.array(ts, dtype=np.float32),
        'lexeme_idx': np.array(lexeme_idx, dtype=np.int32),
    }
    return lambda: coreml_hlr.build_program(weights), inputs, np.array(ref_p), np.array(ref_h)


def neural_sample():
    weights = coreml_neural_hlr.load_mlx_safetensors_as_numpy(coreml_neural_hlr.WEIGHTS_FILE)
    with open(coreml_neural_hlr.VOCAB_FILE) as f:
        vocab = json.load(f)
    oov_index = len(vocab)

    _, test, _ = load_split(DATA_FILE, max_lines=2_000_000)
    sample = {k: v[:2000] for k, v in test.items()}
    inputs = {
        'right': np.array(sample['right'], dtype=np.float32),
        'wrong': np.array(sample['wrong'], dtype=np.float32),
        't': np.array(sample['t'], dtype=np.float32),
        'lexeme_idx': np.array([vocab.get(k, oov_index) for k in sample['lexeme_key']], dtype=np.int32),
    }
    ref_p, ref_h = numpy_reference_forward(weights, inputs['right'], inputs['wrong'],
                                           inputs['t'], inputs['lexeme_idx'])
    return lambda: coreml_neural_hlr.build_program(weights), inputs, ref_p, ref_h


def main():
    print(f'{"model":12s} {"precision":10s} {"max |dh| (days)":>16s} {"max |dp|":>12s}')
    for name, make_sample in [('Linear HLR', linear_sample), ('Neural HLR', neural_sample)]:
        build_program, inputs, ref_p, ref_h = make_sample()
        for label, precision in [('float16', ct.precision.FLOAT16),
                                 ('float32', ct.precision.FLOAT32)]:
            # ct.convert rewrites the program in place, so build a fresh one
            # for each precision
            out = convert(build_program(), precision).predict(inputs)
            max_dh = np.max(np.abs(ref_h - out['h']))
            max_dp = np.max(np.abs(ref_p - out['p']))
            print(f'{name:12s} {label:10s} {max_dh:16.6f} {max_dp:12.8f}')


if __name__ == '__main__':
    main()
