"""
Check whether the linear and neural models would schedule the same words.

A session is all rows sharing a user_id and timestamp in the test set. For each
session with two or more words, each model picks the word with the lowest
predicted recall, which is the word a scheduler would review first. We report
how often two models pick the same word, how often they order a pair of words
the same way, and how often each model's pick is the word with the lowest
observed recall in that session.
"""

import argparse
import csv
import gzip
import os
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark'))
from load_hlr_weights import load_weights  # noqa: E402
from mlx_hlr import MLXHLRModel  # noqa: E402
from mlx_neural_hlr import MLXNeuralHLRModel  # noqa: E402

from data_split import pclip, is_test_user  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
LINEAR_WEIGHTS = os.path.join(REPO_ROOT, 'halflife-regression', 'results', 'hlr.user_split.weights')


def load_test_rows():
    cols = {'right': [], 'wrong': [], 't': [], 'p': [], 'lexeme_key': [], 'user_id': [], 'ts': []}
    test_user = {}
    with gzip.open(DATA_FILE, 'rt') as f:
        for row in csv.DictReader(f):
            user_id = row['user_id']
            if user_id not in test_user:
                test_user[user_id] = is_test_user(user_id)
            if not test_user[user_id]:
                continue
            seen = int(row['history_seen'])
            right = int(row['history_correct'])
            cols['right'].append(right)
            cols['wrong'].append(seen - right)
            cols['t'].append(float(row['delta']) / (60 * 60 * 24))
            cols['p'].append(pclip(float(row['p_recall'])))
            cols['lexeme_key'].append('%s:%s' % (row['learning_language'], row['lexeme_string']))
            cols['user_id'].append(user_id)
            cols['ts'].append(int(row['timestamp']))
    return cols


def predict(model, rows):
    right = mx.array(rows['right'], dtype=mx.float32)
    wrong = mx.array(rows['wrong'], dtype=mx.float32)
    t = mx.array(rows['t'], dtype=mx.float32)
    p, _ = model.forward(right, wrong, t, model.lexeme_indices(rows['lexeme_key']))
    return np.array(p.tolist(), dtype=np.float64)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--linear-weights', nargs='+', default=[LINEAR_WEIGHTS])
    args = argparser.parse_args()

    # the predictions are small enough that the CPU is plenty
    mx.set_default_device(mx.cpu)

    rows = load_test_rows()
    observed = np.array(rows['p'])
    preds = {}
    for path in args.linear_weights:
        preds['linear:' + os.path.basename(path)] = predict(MLXHLRModel(load_weights(path)), rows)
    preds['neural'] = predict(MLXNeuralHLRModel(), rows)

    _, user_codes = np.unique(np.array(rows['user_id']), return_inverse=True)
    timestamps = np.array(rows['ts'], dtype=np.int64)
    order = np.lexsort((timestamps, user_codes))
    new_session = np.r_[True, (np.diff(user_codes[order]) != 0) | (np.diff(timestamps[order]) != 0)]
    starts = np.flatnonzero(new_session)
    ends = np.r_[starts[1:], len(order)]
    sizes = ends - starts
    print(f'test rows = {len(order)}, sessions = {len(starts)}, '
          f'sessions with 2+ words = {(sizes >= 2).sum()}, median size = {int(np.median(sizes))}')

    names = list(preds)
    picks = {name: [] for name in names}
    pair_signs = {name: [] for name in names}
    hits = {name: 0 for name in names}
    sessions_with_spread = 0
    # per session where observed recall varies: its user, each model's hit,
    # and the chance that a uniformly random pick hits
    spread_users, spread_hits, random_hit = [], {name: [] for name in names}, []
    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        idx = order[start:end]
        obs = observed[idx]
        has_spread = obs.min() < obs.max()
        sessions_with_spread += has_spread
        if has_spread:
            spread_users.append(user_codes[idx[0]])
            random_hit.append(np.mean(obs == obs.min()))
        upper = np.triu_indices(len(idx), k=1)
        for name in names:
            pred = preds[name][idx]
            pick = int(np.argmin(pred))
            picks[name].append(pick)
            pair_signs[name].append(np.sign(pred[:, None] - pred[None, :])[upper])
            if has_spread:
                hit = obs[pick] == obs.min()
                hits[name] += hit
                spread_hits[name].append(hit)

    n_sessions = len(picks[names[0]])
    print(f'\nsame first pick, out of {n_sessions} sessions with 2+ words:')
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = np.mean(np.array(picks[a]) == np.array(picks[b]))
            sa = np.concatenate(pair_signs[a])
            sb = np.concatenate(pair_signs[b])
            both_ordered = (sa != 0) & (sb != 0)
            pair_agree = np.mean(sa[both_ordered] == sb[both_ordered])
            print(f'  {a} vs {b}: first pick {same:.4f}, pair order {pair_agree:.4f} '
                  f'({both_ordered.sum()} pairs)')

    print(f'\npick has the lowest observed recall, out of {sessions_with_spread} sessions '
          f'where observed recall varies:')
    for name in names:
        print(f'  {name}: {hits[name] / sessions_with_spread:.4f}')
    print(f'  random pick (expected): {np.mean(random_hit):.4f}')

    # paired bootstrap by user on the hit-rate gap, neural minus the first linear model
    users = np.array(spread_users)
    _, user_index = np.unique(users, return_inverse=True)
    n_users = user_index.max() + 1
    sessions_per_user = np.bincount(user_index, minlength=n_users)
    neural_hits = np.bincount(user_index, weights=np.array(spread_hits['neural'], dtype=float), minlength=n_users)
    linear_hits = np.bincount(user_index, weights=np.array(spread_hits[names[0]], dtype=float), minlength=n_users)
    rng = np.random.default_rng(0)
    gaps = np.empty(1000)
    for i in range(1000):
        sample = rng.integers(0, n_users, size=n_users)
        total = sessions_per_user[sample].sum()
        gaps[i] = (neural_hits[sample].sum() - linear_hits[sample].sum()) / total
    point = (neural_hits.sum() - linear_hits.sum()) / sessions_per_user.sum()
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f'\nhit-rate gap, neural minus {names[0]}: {point:+.4f} '
          f'(95% CI by user: [{lo:+.4f}, {hi:+.4f}], {n_users} users)')


if __name__ == '__main__':
    main()
