# DSV4 Huge final evidence manifest

- Frozen on: 2026-08-10
- Host: B300-M3
- Model: `/mnt/b300-shared/models/DeepSeek-V4-Flash`
- Accepted worktree HEAD: `20148a8c44e9d28cbe3070e009213e5712556337`
- Archive: `/mnt/b300-shared/home/gjy/sglang_huge/.runtime/archive_dsv4_huge_e2e_final_20260810`
- Main report: `dsv4-huge-b300-end-to-end-optimization-report.md`

## Accepted runtime presets

- Req16: v70h identity, commit `259eb32b8`, median 680.6 ms.
- Req128: v70j layer42 swap5, commit `20148a8c4`, median 5450.5 ms.
- Semantics: Eager Prefill; every request has 16384 cached and 4096 new tokens.

## Directory meaning

- `bench/native`: current Native five-sample endpoint logs.
- `bench/huge`: accepted Huge samples, correctness logs, and final controls.
- `nsys/eager`: curated global profiles covering the accepted optimization path.
- `nsys/graph`: preserved historical CUDA Graph comparisons.
- `ncu`: Full reports for accepted kernels and representative rejected fusions.
- `summaries`: immutable stage reports copied from the original runtime tree.
- `source`: v48c CUDA source/overlay/wheel tarball plus a Git bundle containing all refs.
- `SHA256SUMS`: hashes for every other file in this archive.

## Comparability warning

The current Native endpoint medians are supported by the raw five-sample logs, but
there is no same-window four-process Native NSYS capture. Historical Native NSYS
reports are retained as `historical_*_not_final_baseline` only. They must not be
used to explain the final 1.512x/1.527x ratios or to compare AllReduce shares.

CUDA Graph results come from the preserved historical Graph branch and are not a
single-switch comparison against the later v70h/v70j Eager implementation.

## Verify

```bash
cd /mnt/b300-shared/home/gjy/sglang_huge/.runtime/archive_dsv4_huge_e2e_final_20260810
sha256sum -c SHA256SUMS
```
