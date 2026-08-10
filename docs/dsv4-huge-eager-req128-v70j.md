# DSV4 Huge Eager Req128 v70j checkpoint

This checkpoint is intentionally limited to eager prefill on four B300 GPUs.
CUDA Graph is disabled.  Each request has 16,384 cached tokens and 4,096 new
tokens; Req128 is executed as four waves of 32 requests.

## Source and placement

- Parent source commit: `f031f72bf`
- Placement preset: `req128_v70j`
- Placement generator:
  `python/sglang/benchmark/generate_dsv4_huge_static_expert_map.py`
- Runtime selection is explicit through `--init-expert-location` and
  `--ep-dispatch-algorithm static`; there is no automatic workload selection.

Only layer 42 differs from identity placement.  The preset swaps these expert
pairs: `(119, 238)`, `(18, 98)`, `(14, 74)`, `(7, 195)`, and `(25, 176)`.

## Correctness

- Req128: 128/128 first tokens match v70h; maximum absolute logprob difference
  is 0.0666957 and mean absolute difference is 0.0088154.
- Req16: 16/16 first tokens match v70h; maximum absolute logprob difference is
  0.0617614 and mean absolute difference is 0.0083917.
- Req16 does not receive a stable latency benefit, so identity placement remains
  the recommended Req16 configuration.

More aggressive cross-rank placement is intentionally excluded: six swaps in
layer 42 caused one Req128 token mismatch, and balancing layers 41 and 42 caused
three token mismatches.

## Performance

Req128 v70j samples are 5.4637, 5.4505, 5.4757, 5.4338, and 5.4332 seconds;
the median is 5.4505 seconds.  The same-code identity control median is 5.4873
seconds, so the measured improvement is 36.8 ms (0.67%).

The full four-wave nsys comparison reports:

| Metric | identity control | v70j | delta |
| --- | ---: | ---: | ---: |
| kernel launches | 32,016 | 32,016 | 0 |
| GPU envelope | 5,064.063 ms | 5,040.592 ms | -23.471 ms |
| summed GPU time | 19,689.563 ms | 19,593.525 ms | -96.038 ms |
| MoE barrier time | 1,348.923 ms | 1,275.294 ms | -73.629 ms |

The reports are stored under
`.runtime/b300_req128_eager_v48c_nsys/` as
`v48c_v70i_current_control_full4waves.*` and
`v48c_v70j_layer42_swap5_full4waves.*`.
