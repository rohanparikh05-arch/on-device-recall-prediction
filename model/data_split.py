"""
Load the HLR dataset with the train/test split done by user, never by row.

Splitting by row puts the same learner on both sides of the split and
inflates test metrics. Each user_id is hashed, so a given user always lands in
the same partition and the split is reproducible without storing a user list.
"""

import csv
import gzip
import hashlib
import math
import sys

MIN_HALF_LIFE = 15.0 / (24 * 60)
MAX_HALF_LIFE = 274.0
TEST_FRACTION = 0.1


def pclip(p):
    return min(max(p, 0.0001), .9999)


def hclip(h):
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def is_test_user(user_id, test_fraction=TEST_FRACTION):
    digest = hashlib.md5(user_id.encode('utf-8')).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # uniform in [0, 1]
    return bucket < test_fraction


def load_split(input_file, max_lines=None):
    """Return (train, test, vocab).

    train and test are dicts of parallel lists keyed by right, wrong, t, p, h,
    lexeme_key, user_id and lexeme_idx. vocab maps lexeme_key to an embedding
    index and is built from training rows only. Lexemes first seen in test get
    the out-of-vocabulary index len(vocab).
    """
    train = {'right': [], 'wrong': [], 't': [], 'p': [], 'h': [], 'lexeme_key': [], 'user_id': []}
    test = {'right': [], 'wrong': [], 't': [], 'p': [], 'h': [], 'lexeme_key': [], 'user_id': []}
    vocab = {}

    sys.stderr.write('reading data (user-grouped split)...')
    with gzip.open(input_file, 'rt') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_lines is not None and i >= max_lines:
                break
            p = pclip(float(row['p_recall']))
            t = float(row['delta']) / (60 * 60 * 24)
            h = hclip(-t / math.log(p, 2))
            seen = int(row['history_seen'])
            right = int(row['history_correct'])
            wrong = seen - right
            lexeme_key = sys.intern('%s:%s' % (row['learning_language'], row['lexeme_string']))
            # intern so 12.9M rows share one string object per user instead of
            # holding a separate copy on every row
            user_id = sys.intern(row['user_id'])

            bucket = test if is_test_user(user_id) else train
            bucket['right'].append(right)
            bucket['wrong'].append(wrong)
            bucket['t'].append(t)
            bucket['p'].append(p)
            bucket['h'].append(h)
            bucket['lexeme_key'].append(lexeme_key)
            bucket['user_id'].append(user_id)
            if bucket is train and lexeme_key not in vocab:
                vocab[lexeme_key] = len(vocab)

            if i % 1000000 == 0:
                sys.stderr.write('%d...' % i)
    sys.stderr.write('done!\n')

    oov_index = len(vocab)
    train['lexeme_idx'] = [vocab[k] for k in train['lexeme_key']]
    test['lexeme_idx'] = [vocab.get(k, oov_index) for k in test['lexeme_key']]

    sys.stderr.write('|train| = %d\n' % len(train['p']))
    sys.stderr.write('|test|  = %d\n' % len(test['p']))
    sys.stderr.write('vocab size = %d (+1 OOV)\n' % len(vocab))

    return train, test, vocab
