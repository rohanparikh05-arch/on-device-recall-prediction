"""
Time the linear and neural MLX models in one session, so both run under the
same machine conditions.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark_latency as linear  # noqa: E402
import neural_benchmark_latency as neural  # noqa: E402

linear.WEIGHTS_FILE = os.path.join(HERE, '..', 'halflife-regression', 'results', 'hlr.user_split.weights')

for round_number in (1, 2):
    print(f'round {round_number}: MLX linear')
    linear.main()
    print(f'round {round_number}: MLX neural')
    neural.main()
