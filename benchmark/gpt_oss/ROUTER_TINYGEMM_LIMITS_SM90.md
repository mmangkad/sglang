# GPT-OSS Router TinyGEMM Limits on SM90

This note records how the current GPT-OSS router `tinygemm_bf16` limits were chosen in
`python/sglang/srt/models/gpt_oss.py`.

These limits are intentionally documented as SM90/Hopper-only heuristics for now.
They should not be treated as validated for SM100+ or newer architectures.

## Current policy

`GptOssRouterLinear` uses `flashinfer.gemm.tinygemm_bf16` only when all of the
following are true:

- input is 2D
- input is on CUDA
- device compute capability major is at least 9
- input, weight, and bias are BF16
- `skip_bias_add` is disabled
- `M = x.shape[0]` is at most the router-specific cutoff

Current cutoffs:

- `N <= 32`: `M <= 1024`
- `N > 32`: `M <= 256`

Here:

- `M` is the number of token rows in the router input
- `K` is the hidden size (`2880` for GPT-OSS)
- `N` is the router output size, i.e. number of local experts

For GPT-OSS:

- 20B router: `N = 32`
- 120B router: `N = 128`

## Why an `M` limit exists

`tinygemm_bf16` is a latency-oriented small-GEMM kernel. It is excellent for decode-sized
router calls and can also help for some medium `M` ranges, but on Hopper it regresses
once `M` gets large enough.

This means a single global "always use tinygemm on SM90+" rule is not safe.

## Hardware and software used

- GPU: NVIDIA H200 (`SM90`)
- FlashInfer local install with `flashinfer.gemm.tinygemm_bf16`
- Local model snapshots:
  - `openai/gpt-oss-20b`
  - `openai/gpt-oss-120b`

## Measurement approach

I used three levels of validation.

### 1. Synthetic GEMM sanity checks

First, I benchmarked raw `F.linear` vs `tinygemm_bf16` on GPT-OSS-like shapes:

- `K = 2880`
- `N = 32` and `N = 128`
- varying `M`

This gave the initial signal that:

- `N = 32` stays favorable for larger `M`
- `N = 128` crosses over much earlier

These synthetic results were only used as a first pass.

### 2. Router benchmarks with actual local GPT-OSS weights

The final cutoff choice came from benchmarking the actual router module shape with real
router weights from the local model snapshots.

For 20B:

- loaded the local SGLang model in-process
- benchmarked the real `GptOssRouterLinear`

For 120B:

- loading the full model only to benchmark the router was unnecessary and memory-heavy
- instead, I read an actual router weight and bias directly from the local safetensors
  snapshot and benchmarked `GptOssRouterLinear` with those tensors

This still exercised the real router dimensions and real BF16 weights.

### 3. End-to-end smoke tests

I also launched the actual local models through the server and verified generation works
with the patched router path enabled:

- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`

These smoke tests were done with reduced `max_total_tokens` and CUDA graph disabled so
the startup stayed practical while still validating real serving.

## Measured data used to choose the limits

The most important comparison is:

- "threshold256": use tinygemm only up to `M <= 256`
- "unbounded": use tinygemm for all eligible `M`

Lower latency is better.

### 20B actual router (`N = 32`)

Measured with actual local 20B router weights:

| M | threshold256 ms | unbounded ms | takeaway |
|---|----------------:|-------------:|----------|
| 256  | 0.0062 | 0.0069 | tinygemm better |
| 512  | 0.0106 | 0.0070 | `256` cutoff is too conservative |
| 768  | 0.0110 | 0.0098 | still better to keep tinygemm |
| 1024 | 0.0112 | 0.0101 | still slightly better to keep tinygemm |
| 1536 | 0.0116 | 0.0132 | crossover starts here |
| 2048 | 0.0123 | 0.0162 | fallback clearly better |
| 3072 | 0.0148 | 0.0228 | fallback clearly better |
| 4096 | 0.0136 | 0.0296 | fallback clearly better |

Conclusion for `N = 32` on SM90:

- `M = 256` is too low
- `M = 1024` still keeps the win
- somewhere between `1024` and `1536` the kernel stops helping

That is why the current 20B-side cutoff is `1024`.

### 120B actual router (`N = 128`)

Measured with actual local 120B router weights from the safetensors snapshot:

| M | threshold256 ms | unbounded ms | takeaway |
|---|----------------:|-------------:|----------|
| 128  | 0.0071 | 0.0065 | tinygemm slightly better |
| 192  | 0.0093 | 0.0099 | tinygemm still fine |
| 256  | 0.0100 | 0.0095 | roughly tied |
| 320  | 0.0107 | 0.0128 | unbounded already worse |
| 384  | 0.0108 | 0.0125 | fallback better |
| 448  | 0.0110 | 0.0157 | fallback clearly better |
| 512  | 0.0110 | 0.0160 | fallback clearly better |
| 768  | 0.0115 | 0.0220 | fallback clearly better |
| 1024 | 0.0109 | 0.0275 | fallback much better |

Conclusion for `N = 128` on SM90:

- the crossover is much earlier than for `N = 32`
- keeping tinygemm beyond `256` is clearly harmful

That is why the current 120B-side cutoff is `256`.

## Why the policy is width-specific

The router width changes the crossover materially:

- `N = 32` behaves well for much larger `M`
- `N = 128` degrades much earlier

So a single cutoff for all GPT-OSS variants is worse than using a small amount of
shape-aware policy.

## End-to-end smoke test status

Both local models launched and generated successfully with the patched code:

- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`

The smoke-test launch settings intentionally disabled CUDA graph and used reduced
`max_total_tokens` to make validation fast:

- 20B: `--max-total-tokens 4096 --chunked-prefill-size 4096 --mem-fraction-static 0.7`
- 120B: `--max-total-tokens 2048 --chunked-prefill-size 2048 --mem-fraction-static 0.6`

## Interpretation

The current policy means:

- decode-sized router calls use tinygemm frequently
- large prefill-sized router calls fall back to the default path

This is exactly what the Hopper measurements support.

## What should happen on SM100+

Nothing in this note proves the same limits are optimal on SM100+.

In fact, it is plausible that:

- `N = 32` can use a much higher limit, maybe effectively unbounded
- `N = 128` can also tolerate a higher limit than `256`

But that needs real measurement on SM100+ hardware before the SM90-derived limits are
relaxed in code.

## Recommended next step for SM100+

When SM100+ hardware is available, rerun the same router benchmarks and compare:

- current policy
- unbounded tinygemm
- a few higher cutoffs, for example `512`, `1024`, `2048`

for both:

- 20B router (`N = 32`)
- 120B router (`N = 128`)

Until that is done, the current limits should be read as:

"safe Hopper defaults, not final cross-architecture policy."
