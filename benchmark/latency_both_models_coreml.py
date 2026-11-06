"""
Time the linear and neural Core ML models in one session, so both run under the
same machine conditions. Run coreml_neural_hlr.py first. This script builds the
linear model from the user-split weights.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coreml_hlr  # noqa: E402
import coreml_benchmark_latency as linear  # noqa: E402
import coreml_neural_benchmark_latency as neural  # noqa: E402

WEIGHTS = os.path.join(HERE, '..', 'halflife-regression', 'results', 'hlr.user_split.weights')
MODEL = os.path.join(HERE, 'HLR_user_split.mlpackage')

coreml_hlr.WEIGHTS_FILE = WEIGHTS
coreml_hlr.OUT_PATH = MODEL
coreml_hlr.main()
linear.WEIGHTS_FILE = WEIGHTS
linear.MODEL_PATH = MODEL

for round_number in (1, 2):
    print(f'round {round_number}: Core ML linear')
    linear.main()
    print(f'round {round_number}: Core ML neural')
    neural.main()
