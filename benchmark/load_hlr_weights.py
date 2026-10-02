"""
Read the weights file written by halflife-regression/experiment.py
(SpacedRepetitionModel.dump_weights) into bias, right and wrong scalars plus a
lexeme weight table that MLX and Core ML can index.

Standard library only, since both the MLX and the Core ML environments use it.
"""


def load_weights(weights_file):
    w_bias = 0.0
    w_right = 0.0
    w_wrong = 0.0
    lexeme_vocab = {}     # lexeme feature key ("<learning_language>:<lexeme_string>") -> index
    lexeme_weights = []   # index -> weight
    raw = {}              # original string key -> weight, for the reference model

    with open(weights_file) as f:
        for line in f:
            key, val_str = line.rstrip('\n').split('\t')
            val = float(val_str)
            raw[key] = val
            if key == 'bias':
                w_bias = val
            elif key == 'right':
                w_right = val
            elif key == 'wrong':
                w_wrong = val
            else:
                lexeme_vocab[key] = len(lexeme_weights)
                lexeme_weights.append(val)

    oov_index = len(lexeme_weights)
    lexeme_weights.append(0.0)  # unseen lexeme -> weight 0, matches defaultdict(float) in experiment.py

    return {
        'w_bias': w_bias,
        'w_right': w_right,
        'w_wrong': w_wrong,
        'lexeme_vocab': lexeme_vocab,
        'oov_index': oov_index,
        'lexeme_weights': lexeme_weights,
        'raw': raw,
    }
