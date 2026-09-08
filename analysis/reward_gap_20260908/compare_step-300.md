| metric | BF16 @300 | NVFP4 @300 | BF16 mean 201-300 | NVFP4 mean 201-300 | BF16 mean 1-300 | NVFP4 mean 1-300 |
|---|---|---|---|---|---|---|
| loss | 0.01618 | 0.01574 | 0.01924 | 0.007048 | 0.01318 | 0.00705 |
| grad norm | 0.1128 | 0.08105 | 0.1694 | 0.1199 | 0.1562 | 0.1154 |
| clip fraction | 0.003224 | 0.01873 | 0.003195 | 0.02052 | 0.003293 | 0.02057 |
| rollout reward (mean) | 0.1719 | 0.04688 | 0.2519 | 0.09219 | 0.1861 | 0.08776 |
| entropy | 0.2502 | 0.2315 | 0.2858 | 0.2817 | 0.2804 | 0.2782 |
| logprob diff trainer-generator (mean) | -0.0006459 | -0.002646 | -0.0006836 | -0.002914 | -0.0006762 | -0.002911 |
| logprob diff (max) | 0.6201 | 5.219 | 1.116 | 3.872 | 1.132 | 3.723 |
| response length (mean tokens) | 1435 | 1425 | 1437 | 1421 | 1422 | 1407 |
| zero-std group fraction | 0.625 | 0.75 | 0.4925 | 0.6975 | 0.5508 | 0.7071 |
| step wall (s) | 387.8 | 640.4 | 387.5 | 642.4 | 398.3 | 612.2 |
| trainer fwd+bwd (s) | 26.09 | 42.02 | 23.06 | 38.17 | 22.34 | 36.96 |

| throughput / wall-clock (steps 1-300) | BF16 | NVFP4 |
|---|---|---|
| step wall s (optim-step gaps within a launch, structured logs) | 399 (median 398, min 359, max 472, n=299) | 613 (median 603, min 551, max 799, n=299) |
| step wall s, steps 201-300 | 388 (n=100) | 644 (n=100) |
| weight pull s per generator rank per sync | 3.1 (median 2.8, min 2.4, max 61.3, n=1272) | 3.1 (median 2.9, min 2.4, max 39.1, n=1280) |
| step 1 logged | 2026-09-05 22:13 UTC | 2026-09-05 05:41 UTC |
| step 300 logged | 2026-09-07 09:24 UTC | 2026-09-07 23:50 UTC |
| wall-clock step 1 -> 300 (incl. handoffs, outages) | 35.2 h (424 s/step) | 66.1 h (796 s/step) |
| launches with training steps (tfevents files) | 5 | 9 |
