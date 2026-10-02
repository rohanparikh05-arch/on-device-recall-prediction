"""
Paired bootstrap comparing the linear HLR baseline with the neural model on
the user-grouped test set.

Resampling is by user. Rows from one user are correlated, so resampling rows
would understate the variance and overstate significance. Both models are
scored on the same resample in each iteration, so each interval is for the
difference between them.
"""

import argparse
import os
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark'))
from load_hlr_weights import load_weights as load_linear_weights  # noqa: E402
from mlx_hlr import MLXHLRModel  # noqa: E402
from mlx_neural_hlr import MLXNeuralHLRModel  # noqa: E402

from data_split import load_split  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
LINEAR_WEIGHTS = os.path.join(REPO_ROOT, 'halflife-regression', 'results',
                               'hlr.settles.acl16.learning_traces.13m.weights')


def rankdata_avg(a):
    # average ranks for ties, same convention as R's rank() / evaluation.r
    sorter = np.argsort(a, kind='stable')
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a), dtype=np.intp)
    a_sorted = a[sorter]
    is_new_value = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = is_new_value.cumsum()[inv]
    group_start = np.r_[np.flatnonzero(is_new_value), len(a)]
    return 0.5 * (group_start[dense] + group_start[dense - 1] + 1)


def pearson(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    return float(xc.dot(yc) / np.sqrt(xc.dot(xc) * yc.dot(yc)))


def auc_score(labels, scores):
    # rank-based (Mann-Whitney U) AUC, matching evaluation.r's roc(round(p) ~ pp)
    ranks = rankdata_avg(scores)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--resamples', type=int, default=1000)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('-x', dest='max_lines', type=int, default=None)
    args = argparser.parse_args()

    _, test, _ = load_split(DATA_FILE, max_lines=args.max_lines)

    right = mx.array(test['right'], dtype=mx.float32)
    wrong = mx.array(test['wrong'], dtype=mx.float32)
    t = mx.array(test['t'], dtype=mx.float32)

    linear_model = MLXHLRModel(load_linear_weights(LINEAR_WEIGHTS))
    lin_p, lin_h = linear_model.forward(right, wrong, t,
                                        linear_model.lexeme_indices(test['lexeme_key']))
    neural_model = MLXNeuralHLRModel()
    neu_p, neu_h = neural_model.forward(right, wrong, t,
                                        neural_model.lexeme_indices(test['lexeme_key']))

    # predictions are computed once; the bootstrap only reshuffles which rows
    # are counted, so no re-inference (and no retraining) is needed
    p = np.array(test['p'], dtype=np.float64)
    h = np.array(test['h'], dtype=np.float64)
    lin_p = np.array(lin_p.tolist(), dtype=np.float64)
    lin_h = np.array(lin_h.tolist(), dtype=np.float64)
    neu_p = np.array(neu_p.tolist(), dtype=np.float64)
    neu_h = np.array(neu_h.tolist(), dtype=np.float64)

    lin_abs_err = np.abs(p - lin_p)
    neu_abs_err = np.abs(p - neu_p)
    # round-half-to-even, same convention as R's round() in evaluation.r
    labels = np.round(p).astype(np.int64)

    # group row indices by user: rows_by_user holds row indices sorted by
    # user, with user u's rows at [offsets[u] : offsets[u] + counts[u]]
    _, user_codes = np.unique(np.array(test['user_id']), return_inverse=True)
    n_users = user_codes.max() + 1
    rows_by_user = np.argsort(user_codes, kind='stable')
    counts = np.bincount(user_codes, minlength=n_users)
    offsets = np.r_[0, np.cumsum(counts)[:-1]]

    print(f'test rows = {len(p)}, test users = {n_users}')
    print(f'running {args.resamples} user-level resamples...')

    rng = np.random.default_rng(args.seed)
    mae_diffs = np.empty(args.resamples)
    rho_diffs = np.empty(args.resamples)
    auc_diffs = np.empty(args.resamples)

    for i in range(args.resamples):
        sampled = rng.integers(0, n_users, size=n_users)
        sel_counts = counts[sampled]
        sel_offsets = offsets[sampled]
        # ragged gather: expand each sampled user into its own row indices
        starts = np.repeat(np.cumsum(sel_counts) - sel_counts, sel_counts)
        flat = np.repeat(sel_offsets, sel_counts) + (np.arange(sel_counts.sum()) - starts)
        rows = rows_by_user[flat]

        mae_diffs[i] = lin_abs_err[rows].mean() - neu_abs_err[rows].mean()

        h_ranks = rankdata_avg(h[rows])
        rho_lin = pearson(h_ranks, rankdata_avg(lin_h[rows]))
        rho_neu = pearson(h_ranks, rankdata_avg(neu_h[rows]))
        rho_diffs[i] = rho_neu - rho_lin

        resample_labels = labels[rows]
        auc_diffs[i] = (auc_score(resample_labels, neu_p[rows])
                        - auc_score(resample_labels, lin_p[rows]))

        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{args.resamples}', flush=True)

    # point estimates on the full test set (not the bootstrap mean)
    mae_lin = lin_abs_err.mean()
    mae_neu = neu_abs_err.mean()
    h_ranks_full = rankdata_avg(h)
    rho_lin_full = pearson(h_ranks_full, rankdata_avg(lin_h))
    rho_neu_full = pearson(h_ranks_full, rankdata_avg(neu_h))
    auc_lin_full = auc_score(labels, lin_p)
    auc_neu_full = auc_score(labels, neu_p)

    for name, point, lo_hi, lin_val, neu_val in [
        ('MAE', mae_lin - mae_neu, np.percentile(mae_diffs, [2.5, 97.5]), mae_lin, mae_neu),
        ('CORh (Spearman)', rho_neu_full - rho_lin_full,
         np.percentile(rho_diffs, [2.5, 97.5]), rho_lin_full, rho_neu_full),
        ('AUC', auc_neu_full - auc_lin_full,
         np.percentile(auc_diffs, [2.5, 97.5]), auc_lin_full, auc_neu_full),
    ]:
        lo, hi = lo_hi
        excludes_zero = (lo > 0) or (hi < 0)
        print(f'\n{name}')
        print(f'  linear HLR = {lin_val:.4f}, neural HLR = {neu_val:.4f}')
        print(f'  {name} improved by {point:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])')
        print(f'  excludes zero: {"YES" if excludes_zero else "NO"}')


if __name__ == '__main__':
    main()
