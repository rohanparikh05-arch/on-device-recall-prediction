"""
Neural half-life regression. Recall is still p = 2^(-t/h), as in the linear
baseline. What changes is how the half-life exponent is computed: a small MLP
over a learned lexeme embedding and the right/wrong history counts, in place
of a linear weighted sum.
"""

import math

import mlx.core as mx
import mlx.nn as nn

MIN_HALF_LIFE = 15.0 / (24 * 60)
MAX_HALF_LIFE = 274.0
LOG2_MIN_HALF_LIFE = math.log2(MIN_HALF_LIFE)
LOG2_MAX_HALF_LIFE = math.log2(MAX_HALF_LIFE)
EMB_DIM = 8
HIDDEN = 16


class NeuralHLR(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMB_DIM, hidden=HIDDEN):
        super().__init__()
        self.lexeme_emb = nn.Embedding(vocab_size + 1, emb_dim)  # last index = OOV
        self.fc1 = nn.Linear(emb_dim + 2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def dp(self, right, wrong, lexeme_idx):
        right_feat = mx.sqrt(1.0 + right)[:, None]
        wrong_feat = mx.sqrt(1.0 + wrong)[:, None]
        emb = self.lexeme_emb(lexeme_idx)
        x = mx.concatenate([right_feat, wrong_feat, emb], axis=1)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)[:, 0]

    def __call__(self, right, wrong, t, lexeme_idx):
        dp = self.dp(right, wrong, lexeme_idx)
        # Clamp the exponent before exponentiating. Clamping 2**dp afterwards
        # lets it overflow to inf in the forward pass, and inf times the clip's
        # zero gradient mask gives nan in the backward pass.
        dp = mx.clip(dp, LOG2_MIN_HALF_LIFE, LOG2_MAX_HALF_LIFE)
        h = 2.0 ** dp
        p = mx.clip(2.0 ** (-t / h), 0.0001, 0.9999)
        return p, h
