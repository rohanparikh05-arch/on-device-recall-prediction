"""
Train experiment.py's linear HLR model on the user-grouped training split, so
the linear baseline and the neural model are trained on the same learners and
scored on learners neither has seen.

The model and training loop are experiment.py's own; only the choice of
training rows changes.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'halflife-regression'))
import experiment  # noqa: E402

from data_split import is_test_user  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
WEIGHTS_OUT = os.path.join(REPO_ROOT, 'halflife-regression', 'results', 'hlr.user_split.weights')


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-x', dest='max_lines', type=int, default=None)
    argparser.add_argument('--out', default=WEIGHTS_OUT)
    argparser.add_argument('--seed', type=int, default=0)
    args = argparser.parse_args()

    # experiment.py shuffles the training rows with the global random module;
    # seed it so this run can be repeated exactly
    random.seed(args.seed)

    # read_data splits by file position; put the two halves back together and
    # split by user instead
    first_part, last_part = experiment.read_data(DATA_FILE, 'hlr', max_lines=args.max_lines)
    rows = first_part + last_part
    del first_part, last_part
    trainset = [inst for inst in rows if not is_test_user(inst.uid)]
    del rows
    sys.stderr.write('|train| = %d (user-grouped)\n' % len(trainset))

    model = experiment.SpacedRepetitionModel(method='hlr')
    model.train(trainset)
    model.dump_weights(args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
