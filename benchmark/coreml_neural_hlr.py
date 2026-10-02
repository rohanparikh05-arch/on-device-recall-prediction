"""
Build a Core ML package for the trained neural HLR model with the MIL builder.
The weights are read from model/neural_hlr.safetensors as numpy arrays, so
this file does not import MLX.

Needs the Core ML environment (Python 3.12). See the README.
"""

import json
import math
import os

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil import get_new_symbol

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
WEIGHTS_FILE = os.path.join(MODEL_DIR, 'neural_hlr.safetensors')
VOCAB_FILE = os.path.join(MODEL_DIR, 'vocab.json')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'NeuralHLR.mlpackage')

MIN_HALF_LIFE = 15.0 / (24 * 60)
MAX_HALF_LIFE = 274.0
LOG2_MIN_HALF_LIFE = math.log2(MIN_HALF_LIFE)
LOG2_MAX_HALF_LIFE = math.log2(MAX_HALF_LIFE)


def load_mlx_safetensors_as_numpy(path):
    # safetensors is an 8-byte header length, a JSON header and the raw bytes,
    # so it can be read without MLX, which the Core ML environment lacks
    import struct
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()
    arrays = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        dtype = {'F32': np.float32}[meta['dtype']]
        start, end = meta['data_offsets']
        arr = np.frombuffer(data[start:end], dtype=dtype).reshape(meta['shape'])
        arrays[name] = arr
    return arrays


def build_program(weights):
    lexeme_table_np = weights['lexeme_emb.weight']         # [V+1, emb_dim]
    fc1_w = weights['fc1.weight']                            # [hidden, emb_dim+2]
    fc1_b = weights['fc1.bias']                               # [hidden]
    fc2_w = weights['fc2.weight']                             # [1, hidden]
    fc2_b = weights['fc2.bias']                               # [1]

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(get_new_symbol(),), dtype=types.fp32),   # right
        mb.TensorSpec(shape=(get_new_symbol(),), dtype=types.fp32),   # wrong
        mb.TensorSpec(shape=(get_new_symbol(),), dtype=types.fp32),   # t
        mb.TensorSpec(shape=(get_new_symbol(),), dtype=types.int32),  # lexeme_idx
    ])
    def neural_hlr_prog(right, wrong, t, lexeme_idx):
        right_feat = mb.sqrt(x=mb.add(x=right, y=np.float32(1.0)))
        wrong_feat = mb.sqrt(x=mb.add(x=wrong, y=np.float32(1.0)))
        right_feat = mb.expand_dims(x=right_feat, axes=[1])   # [N, 1]
        wrong_feat = mb.expand_dims(x=wrong_feat, axes=[1])   # [N, 1]

        lexeme_table = mb.const(val=lexeme_table_np, name='lexeme_table')
        emb = mb.gather(x=lexeme_table, indices=lexeme_idx, axis=0)  # [N, emb_dim]

        x = mb.concat(values=[right_feat, wrong_feat, emb], axis=1)  # [N, emb_dim+2]

        h1 = mb.linear(x=x, weight=fc1_w, bias=fc1_b)   # [N, hidden]
        h1 = mb.relu(x=h1)
        dp = mb.linear(x=h1, weight=fc2_w, bias=fc2_b)  # [N, 1]
        dp = mb.squeeze(x=dp, axes=[1])                 # [N]

        dp = mb.clip(x=dp, alpha=np.float32(LOG2_MIN_HALF_LIFE), beta=np.float32(LOG2_MAX_HALF_LIFE))
        h = mb.pow(x=np.float32(2.0), y=dp, name='h')

        neg_t_over_h = mb.real_div(x=mb.mul(x=t, y=np.float32(-1.0)), y=h)
        p_raw = mb.pow(x=np.float32(2.0), y=neg_t_over_h)
        p = mb.clip(x=p_raw, alpha=np.float32(0.0001), beta=np.float32(0.9999), name='p')

        return p, h

    return neural_hlr_prog


def main():
    weights = load_mlx_safetensors_as_numpy(WEIGHTS_FILE)
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
        compute_precision=ct.precision.FLOAT32,  # float16 is too coarse: h = 2**dp magnifies rounding error
    )
    mlmodel.save(OUT_PATH)
    print(f'saved {OUT_PATH}')


if __name__ == '__main__':
    main()
