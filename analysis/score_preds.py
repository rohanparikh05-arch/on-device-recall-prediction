"""
Score an experiment.py .preds file with the three metrics from evaluation.r.

Usage: python analysis/score_preds.py halflife-regression/results/<method>.<...>.preds
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from metrics import evaluate  # noqa: E402


def main(preds_file, halflife_meaningful=True):
    p, pp, h, hh = [], [], [], []
    with open(preds_file) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            p.append(float(row['p']))
            pp.append(float(row['pp']))
            h.append(float(row['h']))
            hh.append(float(row['hh']))
    m = evaluate(p, pp, h, hh)
    name = os.path.basename(preds_file).split('.')[0]
    corh = f'{m["spearman_h"]:.4f}' if halflife_meaningful else 'n/a'
    print(f'{name:10s} n={len(p):<9} MAE={m["mae"]:.4f}  AUC={m["auc"]:.4f}  CORh={corh}')


if __name__ == '__main__':
    # experiment.py's lr method returns a random half-life, so CORh means
    # nothing for it. Match the prefix: 'lr.' is also a substring of 'hlr.'.
    main(sys.argv[1], halflife_meaningful=not os.path.basename(sys.argv[1]).startswith('lr.'))
