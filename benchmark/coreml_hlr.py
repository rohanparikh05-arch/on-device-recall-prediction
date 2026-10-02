"""
Build a Core ML package for the linear HLR model with the MIL builder. The
model is a few scalar ops over a lexeme weight table, so there is no PyTorch
or TensorFlow model to convert from.

Needs the Core ML environment (Python 3.12). See the README.
"""

import os
import sys

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil import get_new_symbol

sys.path.insert(0, os.path.dirname(__file__))
from load_hlr_weights import load_weights  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
WEIGHTS_FILE = os.path.join(REPO_ROOT, 'halflife-regression', 'results',
                             'hlr.settles.acl16.learning_traces.13m.weights')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'HLR.mlpackage')

MIN_HALF_LIFE = 15.0 / (24 * 60)
MAX_HALF_LIFE = 274.0


def build_program(weights):
    lexeme_table_np = np.array(weights['lexeme_weights'], dtype=np.float32)
    w_bias = np.float32(weights['w_bias'])
    w_right = np.float32(weights['w_right'])
    w_wrong = np.float32(weights['w_wrong'])

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(get_new_symbol(),), dtype=types.fp32),   # right
        mb.TensorSpec(shape=(get_new_symbol(),), dtype=types.fp32),   # wrong
        mb.TensorSpec(shape=(get_new_symbol(),), dtype=types.fp32),   # t
        mb.TensorSpec(shape=(get_new_symbol(),), dtype=types.int32),  # lexeme_idx
    ])
    def hlr_prog(right, wrong, t, lexeme_idx):
        right_feat = mb.sqrt(x=mb.add(x=right, y=np.float32(1.0)))
        wrong_feat = mb.sqrt(x=mb.add(x=wrong, y=np.float32(1.0)))
        lexeme_table = mb.const(val=lexeme_table_np, name='lexeme_table')
        lex_w = mb.gather(x=lexeme_table, indices=lexeme_idx, axis=0)

        term_right = mb.mul(x=right_feat, y=w_right)
        term_wrong = mb.mul(x=wrong_feat, y=w_wrong)
        dp = mb.add(x=term_right, y=term_wrong)
        dp = mb.add(x=dp, y=w_bias)
        dp = mb.add(x=dp, y=lex_w)

        h_raw = mb.pow(x=np.float32(2.0), y=dp)
        h = mb.clip(x=h_raw, alpha=np.float32(MIN_HALF_LIFE), beta=np.float32(MAX_HALF_LIFE), name='h')

        neg_t_over_h = mb.real_div(x=mb.mul(x=t, y=np.float32(-1.0)), y=h)
        p_raw = mb.pow(x=np.float32(2.0), y=neg_t_over_h)
        p = mb.clip(x=p_raw, alpha=np.float32(0.0001), beta=np.float32(0.9999), name='p')

        return p, h

    return hlr_prog


def main():
    weights = load_weights(WEIGHTS_FILE)
    prog = build_program(weights)

    mlmodel = ct.convert(
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
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.save(OUT_PATH)
    print(f'saved {OUT_PATH}')


if __name__ == '__main__':
    main()
