# On-device recall prediction

Code, trained weights and raw measurements for the paper "On-Device Recall
Prediction for Language Learning: Comparable Accuracy Without a Cloud
Round-Trip" by Rohan Parikh, Dhanvanth Lakshman and Viraj Gadeela, to appear at
IEEE CCNC 2027.

We train a small neural half-life regression (HLR) model on Duolingo's public
dataset of 12.9 million learning traces, convert it to Core ML and MLX, and
compare its on-device latency with a cloud round-trip built from measured
protocol overhead plus published network latency.

## Layout

| Directory | Contents |
|---|---|
| `halflife-regression/` | Duolingo's original HLR code (Settles and Meeder, ACL 2016), ported to Python 3, plus the linear model weights we trained with it |
| `model/` | User-grouped data split, the neural model, training, metrics, the paired bootstrap, and the trained weights |
| `benchmark/` | MLX and Core ML versions of both models, fidelity checks, the precision check, latency benchmarks, and our latency results |
| `cloud/` | The inference server, the loopback client, the cloud estimate, and our loopback results |
| `analysis/` | Baseline scoring and the two figures |
| `data/` | Empty. The dataset goes here (see below) |

## Setup

You need two Python environments. coremltools 9.0 has no compiled wheel for
Python 3.14, and on 3.14 pip falls back to a pure-Python package that can build
a model but fails when you call `predict()`. We used Python 3.12 for Core ML
and Python 3.14 for everything else.

```
python3.14 -m venv .venv
.venv/bin/pip install -r requirements-mlx.txt

python3.12 -m venv .venv-coreml
.venv-coreml/bin/pip install -r requirements-coreml.txt
```

MLX and Core ML both require Apple silicon. Our measurements come from a
MacBook with an Apple M5 Pro.

## Data

The dataset belongs to Duolingo and is not included here. Download
`settles.acl16.learning_traces.13m.csv.gz` (361 MB) from
[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME)
and save it as `data/settles.acl16.learning_traces.13m.csv.gz`. Dataverse
blocked our scripted downloads, so you may need to use a browser.

## Reproducing the paper

Run everything from the repository root unless a command says otherwise.
Scripts that start with `coreml_` and `precision_check.py` use `.venv-coreml`;
the rest use `.venv`.

The repository ships the exact weights behind the paper's numbers. Both
`experiment.py` and `model/train.py` shuffle the training data without a seed,
so retraining gives slightly different weights and metrics. The scoring and
benchmark scripts below use the shipped weights, so they should match the paper
exactly, apart from latency, which depends on your machine.

Some scripts overwrite files that ship with the repository:

- `experiment.py -m hlr` overwrites `halflife-regression/results/hlr.settles.acl16.learning_traces.13m.weights`
- `model/train.py` overwrites `model/neural_hlr.safetensors`, `model/vocab.json` and `model/eval_metrics.json`
- the two neural latency benchmarks overwrite `benchmark/neural_*_results.json`
- `cloud/client_benchmark.py` overwrites `cloud/loopback_results.json`

`git checkout -- <file>` restores the originals.

### Table I: Core ML conversion error by precision

```
.venv-coreml/bin/python benchmark/precision_check.py
```

Core ML converts ML programs to float16 by default. The half-life is
h = 2^dp, so a small rounding error in dp becomes a large error in h, up to
0.79 days for the neural model. Both conversion scripts pin float32 for this
reason.

### Table II: baseline reproduction

`experiment.py` writes to a `results/` folder in the current directory, so run
it from `halflife-regression/`. The HLR run took 8 min 21 s on our machine.

```
cd halflife-regression
../.venv/bin/python experiment.py ../data/settles.acl16.learning_traces.13m.csv.gz
../.venv/bin/python experiment.py -m leitner ../data/settles.acl16.learning_traces.13m.csv.gz
../.venv/bin/python experiment.py -m pimsleur ../data/settles.acl16.learning_traces.13m.csv.gz
../.venv/bin/python experiment.py -m lr ../data/settles.acl16.learning_traces.13m.csv.gz
cd ..
.venv/bin/python analysis/score_preds.py halflife-regression/results/hlr.settles.acl16.learning_traces.13m.preds
```

Repeat the last line for the `leitner`, `pimsleur` and `lr` prediction files.
`score_preds.py` computes the three metrics the same way as `sr_evaluate()` in
`halflife-regression/evaluation.r`, so R is optional. Logistic regression gave
an AUC of 0.486 in one of our runs and 0.487 in the other; the paper discusses
why it falls below chance.

### Table III and Fig. 1: linear versus neural on the user-grouped split

```
.venv/bin/python model/compare_on_user_split.py
.venv/bin/python model/bootstrap_significance.py
```

The first script scores both models on the same test set. The second runs
1,000 paired bootstrap resamples, drawn by user, for the difference in each
metric.

The split hashes each `user_id` with MD5 and puts about 10% of users in the
test set, so no learner appears on both sides. The linear weights in
`halflife-regression/results/` come from experiment.py, which trains on the
first 90% of rows in file order. That portion contains rows from 10,817 of the
11,618 users in our test set, so the linear model has seen most of the test
learners during training and the neural model has seen none of them.

To retrain the neural model (about 78 seconds on our machine):

```
.venv/bin/python model/train.py
```

The defaults are the settings used in the paper: 8 epochs, learning rate
0.001, batch size 8192.

### Table IV: on-device latency

```
.venv/bin/python benchmark/neural_fidelity_check.py
.venv/bin/python benchmark/neural_benchmark_latency.py

.venv-coreml/bin/python benchmark/coreml_neural_hlr.py
.venv-coreml/bin/python benchmark/coreml_neural_fidelity_check.py
.venv-coreml/bin/python benchmark/coreml_neural_benchmark_latency.py
```

`neural_fidelity_check.py` reloads the saved MLX model and prints its metrics
on the test users in the first 2M rows, which catches a bad weights or vocab
file. `coreml_neural_hlr.py` builds `benchmark/NeuralHLR.mlpackage`, and
`coreml_neural_fidelity_check.py` compares its output with a numpy copy of the
forward pass on 2,000 rows.

`fidelity_check.py`, `benchmark_latency.py` and their `coreml_` counterparts do
the same for the linear model. We used them while building the harness; the
paper reports latency for the neural model only.

### Tables V and VI, Fig. 2: cloud round-trip

Start the server in one terminal and run the client in another:

```
.venv/bin/python cloud/server.py
.venv/bin/python cloud/client_benchmark.py
```

The client measures round-trips over localhost, which isolates JSON and HTTP
overhead from network transit (Table V). Then:

```
.venv/bin/python cloud/compose_estimate.py
```

adds published network round-trip times to the loopback measurements and
compares the totals with on-device latency (Table VI). Each round-trip time in
the script is listed with its source: the FCC's Thirteenth Measuring Broadband
America Fixed Broadband Report for fixed broadband, and Sharma et al.,
arXiv:2607.03993, for 5G.

### Figures

```
.venv/bin/python analysis/make_figures.py
```

This writes `fig1_bootstrap` and `fig2_latency` to `analysis/` as PNG and PDF.
The plotted values are typed into the script from the results above.

## License

Our code is under the MIT License (see `LICENSE`). Everything in
`halflife-regression/` is Duolingo's, under their MIT License in
`halflife-regression/LICENSE.md`. Our only changes to `experiment.py` are the
ones needed to run it on Python 3: text-mode file handles, `dict.items()` and
`sys.intern()`. The dataset is Duolingo's and is distributed through Dataverse.

If you use the dataset or the original HLR code, please cite:

> B. Settles and B. Meeder. A Trainable Spaced Repetition Model for Language
> Learning. In Proceedings of the Association for Computational Linguistics
> (ACL), pages 1848-1858, 2016.
