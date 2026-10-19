# Results: models-openrouter-glimmer

Suite **sg-gov-v1** · planner **openrouter-glimmer** (`meta/muse-glimmer-30b`) · library **cold** · judge **openrouter-judge**

Evaluated: sweep started 2026-10-19T10:17:41.545407+00:00; last trial marked 2026-10-19T10:37:32.951519+00:00. Live websites change, so these numbers describe the sites as they were then.

## Summary

| Measure | Value |
|---|---|
| Success rate | 33.3% (range 33.3%–33.3% over 2 rounds) |
| Trials passed | 8 of 24 marked |
| Trials not yet marked | 0 (left out of every figure) |
| Average steps per trial | 13.0 |
| Average tokens per trial | 26,953 |
| Average time per trial | 185.6 s |
| Average cost per trial (agent) | $0.0127 |
| Cost of judging (not the agent) | $0.1050 over 18 calls |
| Judge–human agreement: pass or fail | not yet measured (no human labels) |
| Judge–human agreement: kind of failure (trials both called a fail) | not yet measured (no human labels) |

Agreement is the share of hand-labelled trials where the machine's mark and a person's matched. Cohen's κ discounts the agreement the two would reach by chance alone: 1 is perfect, 0 is no better than chance.

## Per task

| Task | Passed | Avg steps | Avg tokens | Avg time | Avg cost |
|---|---|---|---|---|---|
| cpf-employee-share | 2/2 | 1.0 | 2,528 | 9.8 s | $0.0010 |
| cpf-total-rate | 2/2 | 3.0 | 6,162 | 28.3 s | $0.0027 |
| iras-non-resident-employment-rate | 0/2 | 25.0 | 50,083 | 232.3 s | $0.0211 |
| iras-tax-on-first-80k | 2/2 | 6.0 | 15,008 | 91.1 s | $0.0069 |
| moe-ai-tong-postal-code | 0/2 | 23.5 | 42,560 | 301.3 s | $0.0208 |
| moe-bishan-boys-secondary | 0/2 | 12.5 | 32,693 | 331.5 s | $0.0186 |
| moe-bishan-primary-count | 0/2 | 13.0 | 34,195 | 334.4 s | $0.0190 |
| moe-queenstown-integrated-programme | 0/2 | 16.5 | 38,088 | 305.4 s | $0.0194 |
| mom-annual-leave | 0/2 | 25.0 | 47,144 | 254.2 s | $0.0197 |
| mom-deepavali-2027 | 0/2 | 25.0 | 43,794 | 275.5 s | $0.0185 |
| singstat-latest-cpi | 2/2 | 1.0 | 2,478 | 24.0 s | $0.0013 |
| singstat-total-population | 0/2 | 4.0 | 8,700 | 39.1 s | $0.0038 |

## Failures by kind

| grounding | planning | navigation | site_change |
|---|---|---|---|
| 2 | 10 | 4 | 0 |

## Every verdict

Each row links to the run folder the verdict was made from.

| Task | Trial | Verdict | Kind of failure | Steps | Reason | Run |
|---|---|---|---|---|---|---|
| cpf-employee-share | 1 | pass |  | 1 | every check passed | [run](../runs/models-openrouter-glimmer/cpf-employee-share/trial-1) |
| cpf-employee-share | 2 | pass |  | 1 | every check passed | [run](../runs/models-openrouter-glimmer/cpf-employee-share/trial-2) |
| cpf-total-rate | 1 | pass |  | 3 | every check passed | [run](../runs/models-openrouter-glimmer/cpf-total-rate/trial-1) |
| cpf-total-rate | 2 | pass |  | 3 | every check passed | [run](../runs/models-openrouter-glimmer/cpf-total-rate/trial-2) |
| iras-non-resident-employment-rate | 1 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-glimmer/iras-non-resident-employment-rate/trial-1) |
| iras-non-resident-employment-rate | 2 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-glimmer/iras-non-resident-employment-rate/trial-2) |
| iras-tax-on-first-80k | 1 | pass |  | 5 | every check passed | [run](../runs/models-openrouter-glimmer/iras-tax-on-first-80k/trial-1) |
| iras-tax-on-first-80k | 2 | pass |  | 7 | every check passed | [run](../runs/models-openrouter-glimmer/iras-tax-on-first-80k/trial-2) |
| moe-ai-tong-postal-code | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-glimmer/moe-ai-tong-postal-code/trial-1) |
| moe-ai-tong-postal-code | 2 | fail | planning | 22 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-glimmer/moe-ai-tong-postal-code/trial-2) |
| moe-bishan-boys-secondary | 1 | fail | planning | 14 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-glimmer/moe-bishan-boys-secondary/trial-1) |
| moe-bishan-boys-secondary | 2 | fail | planning | 11 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-glimmer/moe-bishan-boys-secondary/trial-2) |
| moe-bishan-primary-count | 1 | fail | grounding | 14 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-glimmer/moe-bishan-primary-count/trial-1) |
| moe-bishan-primary-count | 2 | fail | grounding | 12 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-glimmer/moe-bishan-primary-count/trial-2) |
| moe-queenstown-integrated-programme | 1 | fail | planning | 14 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-glimmer/moe-queenstown-integrated-programme/trial-1) |
| moe-queenstown-integrated-programme | 2 | fail | planning | 19 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-glimmer/moe-queenstown-integrated-programme/trial-2) |
| mom-annual-leave | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-glimmer/mom-annual-leave/trial-1) |
| mom-annual-leave | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-glimmer/mom-annual-leave/trial-2) |
| mom-deepavali-2027 | 1 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-glimmer/mom-deepavali-2027/trial-1) |
| mom-deepavali-2027 | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-glimmer/mom-deepavali-2027/trial-2) |
| singstat-latest-cpi | 1 | pass |  | 1 | The screenshot clearly displays a 2.3% year-on-year change for CPI in Aug 2026, which matches the agent's answer exactly. | [run](../runs/models-openrouter-glimmer/singstat-latest-cpi/trial-1) |
| singstat-latest-cpi | 2 | pass |  | 1 | The screenshot clearly displays a 2.3% year-on-year change for CPI in Aug 2026, which matches the agent's answer exactly. | [run](../runs/models-openrouter-glimmer/singstat-latest-cpi/trial-2) |
| singstat-total-population | 1 | fail | planning | 4 | the agent did not finish the task (navigation_dead_end) | [run](../runs/models-openrouter-glimmer/singstat-total-population/trial-1) |
| singstat-total-population | 2 | fail | navigation | 4 | the agent did not finish the task (navigation_dead_end) | [run](../runs/models-openrouter-glimmer/singstat-total-population/trial-2) |
