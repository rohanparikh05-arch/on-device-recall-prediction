"""
Estimate a cloud round-trip as measured loopback time plus a published network
round-trip time, and compare it with measured on-device latency.

Every round-trip figure in RTT_SCENARIOS is listed with its source. If you add
a scenario, add its source too.
"""

import json
import os

HERE = os.path.dirname(__file__)
LOOPBACK = os.path.join(HERE, 'loopback_results.json')
MLX_RESULTS = os.path.join(HERE, '..', 'benchmark', 'neural_mlx_results.json')
COREML_RESULTS = os.path.join(HERE, '..', 'benchmark', 'neural_coreml_results.json')

# (label, round-trip time in ms, source)
# The FCC rows are last-mile latency to nearby measurement servers, so they
# are a lower bound on a real cloud round-trip. The 5G rows were measured to
# actual cloud regions.
RTT_SCENARIOS = [
    ('fixed fiber, idle (best)', 7.0,
     'FCC Thirteenth MBA Fixed Broadband Report (2023), Chart 6: fiber idle latency 7-14 ms'),
    ('fixed cable, idle (worst)', 24.0,
     'FCC Thirteenth MBA Fixed Broadband Report (2023), Chart 6: cable idle latency 12-24 ms'),
    ('5G to mobile edge (MEC)', 44.62,
     'Sharma et al., arXiv:2607.03993, Fig. 5: MEC average 44.62 ms over 5G'),
    ('5G to nearest cloud region', 47.69,
     'Sharma et al., arXiv:2607.03993, Fig. 5: NO-East average 47.69 ms'),
    ('5G to distant cloud region', 78.49,
     'Sharma et al., arXiv:2607.03993, Fig. 5: SE-Central average 78.49 ms'),
]


def main():
    with open(LOOPBACK) as f:
        loopback = json.load(f)
    with open(MLX_RESULTS) as f:
        mlx_results = json.load(f)
    with open(COREML_RESULTS) as f:
        coreml_results = json.load(f)

    for batch in ['1', '32', '256', '1024']:
        on_device_mlx = mlx_results[batch]['mean_ms']
        on_device_coreml = coreml_results[batch]['mean_ms']
        best_on_device = min(on_device_mlx, on_device_coreml)
        lb = loopback[batch]['loopback_mean_ms']

        print(f'\n=== batch size {batch} '
              f'(payload {loopback[batch]["payload_bytes"] / 1024:.1f} KB) ===')
        print(f'  on-device MLX      : {on_device_mlx:8.3f} ms')
        print(f'  on-device Core ML  : {on_device_coreml:8.3f} ms')
        print(f'  server inference   : {loopback[batch]["server_inference_mean_ms"]:8.3f} ms')
        print(f'  loopback round-trip: {lb:8.3f} ms  (inference + JSON + HTTP, no network)')
        print(f'  {"scenario":<28} {"cloud total":>12} {"vs on-device":>14}')
        for label, rtt, _source in RTT_SCENARIOS:
            total = lb + rtt
            print(f'  {label:<28} {total:>9.2f} ms {total / best_on_device:>12.0f}x')

    print('\nsources:')
    for label, rtt, source in RTT_SCENARIOS:
        print(f'  {label} ({rtt} ms): {source}')


if __name__ == '__main__':
    main()
