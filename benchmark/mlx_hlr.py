"""MLX reimplementation of experiment.py's 'hlr' method, for on-device timing."""

import mlx.core as mx

MIN_HALF_LIFE = 15.0 / (24 * 60)   # 15 minutes, in days
MAX_HALF_LIFE = 274.0              # 9 months, in days


class MLXHLRModel:
    def __init__(self, weights):
        self.w_bias = weights['w_bias']
        self.w_right = weights['w_right']
        self.w_wrong = weights['w_wrong']
        self.lexeme_vocab = weights['lexeme_vocab']
        self.oov_index = weights['oov_index']
        self.lexeme_table = mx.array(weights['lexeme_weights'], dtype=mx.float32)

    def lexeme_indices(self, lexeme_keys):
        idx = [self.lexeme_vocab.get(k, self.oov_index) for k in lexeme_keys]
        return mx.array(idx, dtype=mx.uint32)

    def forward(self, right, wrong, t, lexeme_idx):
        # right, wrong, t: mx.array float32 [N]  (raw historical counts / days)
        # lexeme_idx: mx.array uint32 [N], from lexeme_indices()
        right_feat = mx.sqrt(1.0 + right)
        wrong_feat = mx.sqrt(1.0 + wrong)
        lex_w = self.lexeme_table[lexeme_idx]
        dp = right_feat * self.w_right + wrong_feat * self.w_wrong + self.w_bias + lex_w
        h = mx.clip(2.0 ** dp, MIN_HALF_LIFE, MAX_HALF_LIFE)
        p = mx.clip(2.0 ** (-t / h), 0.0001, 0.9999)
        mx.eval(p, h)
        return p, h
