# Benchmark results

One row per sweep. A sweep is one pass over the benchmark with one model and one skill library (cold: empty; warm: learned). Success rate is the mean over rounds of trials, with the lowest and highest round. Tokens and time are always shown; dollars only where every call had a price.

| Sweep | Planner model | Library | Trials per task | Success rate | Avg steps | Avg tokens | Avg time | Avg cost | Started |
|---|---|---|---|---|---|---|---|---|---|
| [eval-openrouter-cold](eval-openrouter-cold.md) | `nvidia/nemotron-3-nano-30b-a3b` | cold | 3 | 33.3% (range 25.0%–41.7% over 3 rounds) | 12.6 | 33,377 | 108.0 s | $0.0042 | 2026-10-18T05:26:56.054233+00:00 |
| [eval-openrouter-warm](eval-openrouter-warm.md) | `nvidia/nemotron-3-nano-30b-a3b` | warm | 3 | 52.8% (range 41.7%–66.7% over 3 rounds) | 9.0 | 23,763 | 78.2 s | $0.0030 | 2026-10-18T05:45:17.772456+00:00 |
| [models-openrouter](models-openrouter.md) | `nvidia/nemotron-3-nano-30b-a3b` | cold | 2 | 41.7% (range 41.7%–41.7% over 2 rounds) | 10.8 | 32,005 | 78.2 s | $0.0042 | 2026-10-19T09:48:13.199547+00:00 |
| [models-openrouter-glimmer](models-openrouter-glimmer.md) | `meta/muse-glimmer-30b` | cold | 2 | 33.3% (range 33.3%–33.3% over 2 rounds) | 13.0 | 26,953 | 185.6 s | $0.0127 | 2026-10-19T10:17:41.545407+00:00 |
| [models-openrouter-qwen](models-openrouter-qwen.md) | `qwen/qwen3.6-35b-a3b` | cold | 2 | 33.3% (range 33.3%–33.3% over 2 rounds) | 12.6 | 39,705 | 172.5 s | $0.0174 | 2026-10-19T09:57:13.288964+00:00 |
| [modes-local](modes-local.md) | `nvidia/nemotron-3-nano-30b-a3b` | cold | 1 | 100.0% (1 round only: no range, treat as a rough guide) | 1.0 | 2,607 | 3.9 s | unknown (1 call unpriced) | 2026-10-20T06:02:59.529748+00:00 |
| [modes-nvidia](modes-nvidia.md) | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | cold | 1 | 100.0% (1 round only: no range, treat as a rough guide) | 1.0 | 2,685 | 11.0 s | unknown (1 call unpriced) | 2026-10-20T06:03:50.992158+00:00 |
| [modes-openrouter](modes-openrouter.md) | `nvidia/nemotron-3-nano-30b-a3b` | cold | 1 | 100.0% (1 round only: no range, treat as a rough guide) | 2.0 | 6,411 | 31.3 s | $0.0007 | 2026-10-20T06:03:05.786814+00:00 |

## Judge–human agreement

Over every sweep above together. Agreement is the share of hand-labelled trials where the machine's mark and a person's matched. Cohen's κ discounts the agreement the two would reach by chance alone: 1 is perfect, 0 is no better than chance.

| Measure | Value |
|---|---|
| Judge–human agreement: pass or fail | not yet measured (no human labels) |
| Judge–human agreement: kind of failure (trials both called a fail) | not yet measured (no human labels) |
