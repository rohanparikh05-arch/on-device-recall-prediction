"""
MAE, AUC and Spearman correlation, defined the same way as sr_evaluate() in
halflife-regression/evaluation.r:
  MAE      = mean(|p - pp|)
  AUC      = roc(round(p) ~ pp), with pp as the score and round(p) as the label
  Spearman = cor.test(h, hh, method='s')
"""


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    mx_ = sum(x) / n
    my_ = sum(y) / n
    num = sum((x[i] - mx_) * (y[i] - my_) for i in range(n))
    dx = sum((x[i] - mx_) ** 2 for i in range(n))
    dy = sum((y[i] - my_) ** 2 for i in range(n))
    return num / (dx * dy) ** 0.5


def spearman(x, y):
    return pearson(rank(x), rank(y))


def auc(labels, scores):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    ranks = rank(scores)
    sum_ranks_pos = sum(r for r, lab in zip(ranks, labels) if lab == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mae(x, y):
    return sum(abs(x[i] - y[i]) for i in range(len(x))) / len(x)


def evaluate(p, pp, h, hh):
    labels = [round(pi) for pi in p]
    return {
        'mae': mae(p, pp),
        'auc': auc(labels, pp),
        'spearman_h': spearman(h, hh),
    }
