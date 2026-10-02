import argparse
import json
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from data_split import load_split
from neural_hlr import NeuralHLR
from metrics import evaluate

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data',
                          'settles.acl16.learning_traces.13m.csv.gz')
WEIGHTS_OUT = os.path.join(os.path.dirname(__file__), 'neural_hlr.safetensors')
VOCAB_OUT = os.path.join(os.path.dirname(__file__), 'vocab.json')
METRICS_OUT = os.path.join(os.path.dirname(__file__), 'eval_metrics.json')

HLWT = 0.01  # weight on the half-life loss term, matches experiment.py's default


def to_mx(d):
    return {
        'right': mx.array(d['right'], dtype=mx.float32),
        'wrong': mx.array(d['wrong'], dtype=mx.float32),
        't': mx.array(d['t'], dtype=mx.float32),
        'p': mx.array(d['p'], dtype=mx.float32),
        'h': mx.array(d['h'], dtype=mx.float32),
        'lexeme_idx': mx.array(d['lexeme_idx'], dtype=mx.uint32),
    }


def loss_fn(model, batch):
    p_pred, h_pred = model(batch['right'], batch['wrong'], batch['t'], batch['lexeme_idx'])
    slp = mx.mean((batch['p'] - p_pred) ** 2)
    slh = mx.mean((batch['h'] - h_pred) ** 2)
    return slp + HLWT * slh


def evaluate_test(model, test_mx, batch_size=65536):
    n = test_mx['p'].shape[0]
    all_p, all_h = [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        p_pred, h_pred = model(
            test_mx['right'][start:end], test_mx['wrong'][start:end],
            test_mx['t'][start:end], test_mx['lexeme_idx'][start:end],
        )
        all_p.extend(p_pred.tolist())
        all_h.extend(h_pred.tolist())
    return evaluate(test_mx['p'].tolist(), all_p, test_mx['h'].tolist(), all_h)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-x', dest='max_lines', type=int, default=None)
    argparser.add_argument('--epochs', type=int, default=8)
    argparser.add_argument('--batch-size', type=int, default=8192)
    argparser.add_argument('--lr', type=float, default=0.001)
    args = argparser.parse_args()

    train, test, vocab = load_split(DATA_FILE, max_lines=args.max_lines)
    train_mx = to_mx(train)
    test_mx = to_mx(test)

    model = NeuralHLR(vocab_size=len(vocab))
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    n_train = train_mx['p'].shape[0]
    print(f'training on {n_train} instances, {args.epochs} epochs, batch_size={args.batch_size}')

    for epoch in range(args.epochs):
        order = list(range(n_train))
        random.shuffle(order)
        order = mx.array(order, dtype=mx.uint32)

        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            idx = order[start:end]
            batch = {k: v[idx] for k, v in train_mx.items()}
            loss, grads = loss_and_grad_fn(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            running_loss += loss.item()
            n_batches += 1

        metrics = evaluate_test(model, test_mx)
        print(f'epoch {epoch + 1}/{args.epochs}  '
              f'train_loss={running_loss / n_batches:.5f}  '
              f'time={time.time() - epoch_start:.1f}s  '
              f'test: mae={metrics["mae"]:.4f} auc={metrics["auc"]:.4f} spearman_h={metrics["spearman_h"]:.4f}')

    final_metrics = evaluate_test(model, test_mx)
    print('final:', final_metrics)

    model.save_weights(WEIGHTS_OUT)
    with open(VOCAB_OUT, 'w') as f:
        json.dump(vocab, f)
    with open(METRICS_OUT, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f'saved weights to {WEIGHTS_OUT}, vocab to {VOCAB_OUT}, metrics to {METRICS_OUT}')


if __name__ == '__main__':
    main()
