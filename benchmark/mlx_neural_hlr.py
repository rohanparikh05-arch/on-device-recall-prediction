"""MLX on-device wrapper for the trained neural HLR model (see model/train.py)."""

import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from neural_hlr import NeuralHLR  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
WEIGHTS_FILE = os.path.join(MODEL_DIR, 'neural_hlr.safetensors')
VOCAB_FILE = os.path.join(MODEL_DIR, 'vocab.json')


class MLXNeuralHLRModel:
    def __init__(self, weights_file=WEIGHTS_FILE, vocab_file=VOCAB_FILE):
        with open(vocab_file) as f:
            self.lexeme_vocab = json.load(f)
        self.oov_index = len(self.lexeme_vocab)

        self.model = NeuralHLR(vocab_size=len(self.lexeme_vocab))
        self.model.load_weights(weights_file)
        self.model.eval()

    def lexeme_indices(self, lexeme_keys):
        idx = [self.lexeme_vocab.get(k, self.oov_index) for k in lexeme_keys]
        return mx.array(idx, dtype=mx.uint32)

    def forward(self, right, wrong, t, lexeme_idx):
        p, h = self.model(right, wrong, t, lexeme_idx)
        mx.eval(p, h)
        return p, h
