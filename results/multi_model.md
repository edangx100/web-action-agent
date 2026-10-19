# Multi-model comparison

The same benchmark on different AI models. Each arm is one sweep with an empty skill library (cold), and the arms are meant to differ only in `--profile`, which picks the models -- one for steps that read the page as text, one for steps that also send a screenshot. The checks below test that against what each sweep recorded. Success rate is the mean over rounds of trials, with the lowest and highest round. N is how many marked trials each figure rests on: with fewer trials per task than the headline cold-vs-warm sweeps, these figures are rougher, and the ranges show by how much.

## Results

| Arm | Planner model | Screenshot model | Trials marked (N) | Success rate | Avg steps | Avg tokens | Avg time | Avg cost (agent) | Code | Started |
|---|---|---|---|---|---|---|---|---|---|---|
| [openrouter](models-openrouter.md) | `nvidia/nemotron-3-nano-30b-a3b` | `nvidia/nemotron-3.5-super-vl` | 24 (12 tasks × 2) | 41.7% (range 41.7%–41.7% over 2 rounds) | 10.8 | 32,005 | 78.2 s | $0.0042 | `910e4ad8e017` | 2026-10-19T09:48:13.199547+00:00 |
| [openrouter-glimmer](models-openrouter-glimmer.md) | `meta/muse-glimmer-30b` | `meta/muse-glimmer-30b` | 24 (12 tasks × 2) | 33.3% (range 33.3%–33.3% over 2 rounds) | 13.0 | 26,953 | 185.6 s | $0.0127 | `910e4ad8e017` | 2026-10-19T10:17:41.545407+00:00 |
| [openrouter-qwen](models-openrouter-qwen.md) | `qwen/qwen3.6-35b-a3b` | `qwen/qwen3.6-35b-a3b` | 24 (12 tasks × 2) | 33.3% (range 33.3%–33.3% over 2 rounds) | 12.6 | 39,705 | 172.5 s | $0.0174 | `910e4ad8e017` | 2026-10-19T09:57:13.288964+00:00 |

## Did only the model change?

| Check | Result |
|---|---|
| Same code in every arm | yes: every trial of every arm ran commit `910e4ad8e017`, with nothing uncommitted |
| Same benchmark suite | yes: sg-gov-v1 |
| Same judge | yes: openrouter-judge |
| Empty skill library in every arm | yes: cold |

A code version is the git commit the sweep ran, with `-dirty` added when the agent's code, settings, tasks or dependencies had changes not yet committed. A sweep restarted on newer code lists both versions.

## Per task

Trials passed out of trials marked, for each arm.

| Task | openrouter | openrouter-glimmer | openrouter-qwen |
|---|---|---|---|
| cpf-employee-share | 2/2 | 2/2 | 2/2 |
| cpf-total-rate | 0/2 | 2/2 | 0/2 |
| iras-non-resident-employment-rate | 2/2 | 0/2 | 1/2 |
| iras-tax-on-first-80k | 2/2 | 2/2 | 1/2 |
| moe-ai-tong-postal-code | 0/2 | 0/2 | 0/2 |
| moe-bishan-boys-secondary | 0/2 | 0/2 | 0/2 |
| moe-bishan-primary-count | 0/2 | 0/2 | 0/2 |
| moe-queenstown-integrated-programme | 0/2 | 0/2 | 0/2 |
| mom-annual-leave | 0/2 | 0/2 | 0/2 |
| mom-deepavali-2027 | 0/2 | 0/2 | 0/2 |
| singstat-latest-cpi | 2/2 | 2/2 | 2/2 |
| singstat-total-population | 2/2 | 0/2 | 2/2 |
