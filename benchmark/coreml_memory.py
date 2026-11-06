"""
How much a Core ML model adds to peak resident memory. Run once per model, each
in a new process, because peak memory never goes back down.

    python benchmark/coreml_memory.py benchmark/NeuralHLR.mlpackage
"""

import resource
import sys

import numpy as np
import coremltools as ct


def peak_mb():
    # ru_maxrss is in bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


before = peak_mb()
model = ct.models.MLModel(sys.argv[1])
loaded = peak_mb()
inputs = {'right': np.array([3.0], np.float32), 'wrong': np.array([1.0], np.float32),
          't': np.array([2.0], np.float32), 'lexeme_idx': np.array([9000], np.int32)}
for _ in range(100):
    model.predict(inputs)
after = peak_mb()
print(f'{sys.argv[1]}: before load {before:.1f} MB, after load {loaded:.1f} MB, '
      f'after 100 predictions {after:.1f} MB, added {after - before:.1f} MB')
