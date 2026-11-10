# Predicting Word Recall on the Device Instead of the Cloud

Language apps decide which word you review next by predicting whether you still
remember it. This repository runs that prediction on a laptop and an iPhone and
compares the speed and accuracy with a cloud round-trip.

It holds the code, trained weights and raw measurements for the paper
"On-Device Recall Prediction for Language Learning: Comparable Accuracy Without
a Cloud Round-Trip" by Rohan Parikh, Viraj Gadeela and Dhanvanth Lakshman, to
appear at IEEE CCNC 2027.

We train a small neural half-life regression (HLR) model on Duolingo's public
dataset of 12.9 million learning traces, convert it to Core ML and MLX, and
measure it on both devices. The cloud figure combines measured protocol overhead
with published network latency.

## Layout

| Directory | Contents |
|---|---|
| `halflife-regression/` | Duolingo's original HLR code (Settles and Meeder, ACL 2016), ported to Python 3, plus our linear model weights: one set trained on the original row-ordered split and five on the user-grouped split |
| `model/` | User-grouped data split, the neural model, training for both models, metrics, the paired bootstrap, the scheduler test, and the trained weights |
| `benchmark/` | MLX and Core ML versions of both models, fidelity checks, the precision check, latency and memory benchmarks, and our laptop results |
| `benchmark/ios/` | The iPhone test app, a Mac build of the same harness, and our raw phone and Mac results |
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

MLX and Core ML both require Apple silicon. Our laptop measurements come from a
MacBook with an Apple M5 Pro, and our phone measurements from an iPhone 17 on
iOS 26.6.1. Building the phone app also needs Xcode.

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

The repository ships the exact weights behind the paper's numbers.
`experiment.py` and `model/train.py` shuffle the training data without a seed,
so retraining with them gives slightly different weights and metrics.
`model/train_linear_user_split.py` takes a seed, though retraining may not match
our files exactly on a different machine. The scoring and benchmark scripts
below use the shipped weights, so they should match the paper apart from
latency and memory, which depend on your machine.

Some scripts overwrite files that ship with the repository:

- `experiment.py -m hlr` overwrites `halflife-regression/results/hlr.settles.acl16.learning_traces.13m.weights`
- `model/train.py` overwrites `model/neural_hlr.safetensors`, `model/vocab.json` and `model/eval_metrics.json`
- `model/train_linear_user_split.py` overwrites `halflife-regression/results/hlr.user_split.weights` unless you pass `--out`
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
test set, so no learner appears on both sides. Both models train on the same
11,551,576 rows from the remaining users.

To retrain them:

```
.venv/bin/python model/train.py
.venv/bin/python model/train_linear_user_split.py --seed 0
```

The neural defaults are the settings used in the paper: 8 epochs, learning rate
0.001, batch size 8192. That run takes about 78 seconds on our machine. Table
III uses `hlr.user_split.weights`, which is seed 0. Seeds 1 to 4 are
`hlr.user_split.seed1.weights` and so on, and the paper reports the range across
all five. Pass `--out` to write a new seed to its own file. To score one of them:

```
.venv/bin/python model/compare_on_user_split.py --linear-weights halflife-regression/results/hlr.user_split.seed1.weights
.venv/bin/python model/bootstrap_significance.py --linear-weights halflife-regression/results/hlr.user_split.seed1.weights
```

### Leakage in the row-ordered split

`experiment.py` trains on the first 90% of rows in file order. That portion
contains 90.1% of our test rows and rows from 10,817 of our 11,618 test users.
This script prints those counts, along with the 182 test rows and 108 words the
neural model never saw in training:

```
.venv/bin/python model/leakage_check.py
```

To score the row-ordered weights on our test set:

```
.venv/bin/python model/compare_on_user_split.py --linear-weights halflife-regression/results/hlr.settles.acl16.learning_traces.13m.weights
```

### Scheduler test

Each model picks the word with the lowest predicted recall in each test session,
which is the word a scheduler would review first. This script reports how often
the models agree and how often each pick has the lowest observed recall:

```
.venv/bin/python model/scheduler_agreement.py --linear-weights \
    halflife-regression/results/hlr.user_split.weights \
    halflife-regression/results/hlr.user_split.seed{1,2,3,4}.weights
```

The bootstrap interval on the gap compares the neural model with the first file
listed.

### Table IV: on-device latency on the laptop

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
the same for the linear model.

### Linear versus neural: speed and memory

These run both models in one session, so the two sets of timings share the same
machine conditions:

```
.venv/bin/python benchmark/latency_both_models_mlx.py
.venv-coreml/bin/python benchmark/latency_both_models_coreml.py
```

For memory, run one model per process:

```
.venv-coreml/bin/python benchmark/coreml_memory.py benchmark/HLR_user_split.mlpackage
.venv-coreml/bin/python benchmark/coreml_memory.py benchmark/NeuralHLR.mlpackage
```

The 0.61 MB and 0.08 MB weight sizes in the paper are the `weight.bin` files
inside the two packages.

### Table IV: on-device latency on the iPhone

Our raw results are in `benchmark/ios/results/`. Table IV reports the mean of
`iphone17_exp2_run1.json` and `iphone17_exp2_run2.json`, using the `all` compute
units setting. The `mac_exp2_*` files are the same app code running on the laptop.

To run it yourself, build the phone models and test rows, which needs the dataset:

```
.venv-coreml/bin/python benchmark/ios/export_ios_assets.py
```

Then open `benchmark/ios/HLRBench.xcodeproj` in Xcode, set your own signing team
and bundle ID, and run it on a phone. The app times both models, compares their
predictions with the reference values, and writes `results.json` to its Documents
folder. Launch arguments `mem-neural` and `mem-linear` run the memory test instead.

Check the fidelity numbers before you trust the timings. On iOS 26.6, Core ML
evaluated our `pow(2, d)` operation as `d**2`, so the first phone build returned
a half-life of 57.52 days where the reference gives 191.90. The laptop gave the
right answer from the same model. `iphone17_pow_FAILED_fidelity.json` and
`iphone17_pow_diag.json` record that failure. `export_ios_assets.py` now uses
Core ML's `exp2` operation for the phone models.

To run the same harness on a Mac:

```
cd benchmark/ios
mkdir -p build/mac
xcrun coremlcompiler compile HLRBench/NeuralHLR.mlpackage build/mac
xcrun coremlcompiler compile HLRBench/LinearHLR.mlpackage build/mac
xcrun swiftc -O HLRBench/Bench.swift mac/main.swift -o build/mac_bench
./build/mac_bench build/mac HLRBench/fidelity_rows.json results.json
```

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
