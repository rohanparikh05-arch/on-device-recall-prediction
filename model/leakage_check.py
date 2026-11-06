"""
Count how much the row-ordered training split overlaps our user-grouped test
set, and how many test rows use a word the neural model never saw in training.
"""

import csv
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_split import is_test_user  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_FILE = os.path.join(REPO_ROOT, 'data', 'settles.acl16.learning_traces.13m.csv.gz')
VOCAB_FILE = os.path.join(os.path.dirname(__file__), 'vocab.json')


def main():
    with open(VOCAB_FILE) as f:
        vocab = json.load(f)

    row_users = []
    test_user = {}
    unseen = []
    with gzip.open(DATA_FILE, 'rt') as f:
        for row in csv.DictReader(f):
            user = sys.intern(row['user_id'])
            row_users.append(user)
            if user not in test_user:
                test_user[user] = is_test_user(user)
            if test_user[user]:
                key = '%s:%s' % (row['learning_language'], row['lexeme_string'])
                if key not in vocab:
                    unseen.append(key)

    # experiment.py trains on the first 90% of rows in file order
    cutoff = int(0.9 * len(row_users))
    test_rows = sum(1 for user in row_users if test_user[user])
    overlap = [user for user in row_users[:cutoff] if test_user[user]]
    print(f'test rows: {test_rows}, test users: {sum(test_user.values())}')
    print(f'test rows inside the row-ordered training split: {len(overlap)} ({len(overlap) / test_rows:.1%})')
    print(f'test users with rows in that split: {len(set(overlap))}')
    print(f'test rows with a word missing from the vocabulary: {len(unseen)}, covering {len(set(unseen))} words')


if __name__ == '__main__':
    main()
