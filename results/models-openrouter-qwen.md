# Results: models-openrouter-qwen

Suite **sg-gov-v1** · planner **openrouter-qwen** (`qwen/qwen3.6-35b-a3b`) · library **cold** · judge **openrouter-judge**

Evaluated: sweep started 2026-10-19T09:57:13.288964+00:00; last trial marked 2026-10-19T10:17:40.562257+00:00. Live websites change, so these numbers describe the sites as they were then.

## Summary

| Measure | Value |
|---|---|
| Success rate | 33.3% (range 33.3%–33.3% over 2 rounds) |
| Trials passed | 8 of 24 marked |
| Trials not yet marked | 0 (left out of every figure) |
| Average steps per trial | 12.6 |
| Average tokens per trial | 39,705 |
| Average time per trial | 172.5 s |
| Average cost per trial (agent) | $0.0174 |
| Cost of judging (not the agent) | $0.1081 over 20 calls |
| Judge–human agreement: pass or fail | not yet measured (no human labels) |
| Judge–human agreement: kind of failure (trials both called a fail) | not yet measured (no human labels) |

Agreement is the share of hand-labelled trials where the machine's mark and a person's matched. Cohen's κ discounts the agreement the two would reach by chance alone: 1 is perfect, 0 is no better than chance.

## Per task

| Task | Passed | Avg steps | Avg tokens | Avg time | Avg cost |
|---|---|---|---|---|---|
| cpf-employee-share | 2/2 | 1.0 | 2,594 | 4.0 s | $0.0005 |
| cpf-total-rate | 0/2 | 25.0 | 74,978 | 194.4 s | $0.0215 |
| iras-non-resident-employment-rate | 1/2 | 13.5 | 46,008 | 199.3 s | $0.0228 |
| iras-tax-on-first-80k | 1/2 | 5.0 | 12,902 | 29.7 s | $0.0033 |
| moe-ai-tong-postal-code | 0/2 | 2.0 | 6,320 | 59.8 s | $0.0034 |
| moe-bishan-boys-secondary | 0/2 | 14.5 | 65,134 | 338.5 s | $0.0371 |
| moe-bishan-primary-count | 0/2 | 17.5 | 67,145 | 340.8 s | $0.0335 |
| moe-queenstown-integrated-programme | 0/2 | 22.0 | 66,568 | 318.6 s | $0.0270 |
| mom-annual-leave | 0/2 | 25.0 | 63,212 | 255.6 s | $0.0265 |
| mom-deepavali-2027 | 0/2 | 22.5 | 64,429 | 313.9 s | $0.0319 |
| singstat-latest-cpi | 2/2 | 1.0 | 2,380 | 5.0 s | $0.0006 |
| singstat-total-population | 2/2 | 2.0 | 4,789 | 9.8 s | $0.0012 |

## Failures by kind

| grounding | planning | navigation | site_change |
|---|---|---|---|
| 2 | 13 | 1 | 0 |

## Every verdict

Each row links to the run folder the verdict was made from.

| Task | Trial | Verdict | Kind of failure | Steps | Reason | Run |
|---|---|---|---|---|---|---|
| cpf-employee-share | 1 | pass |  | 1 | every check passed | [run](../runs/models-openrouter-qwen/cpf-employee-share/trial-1) |
| cpf-employee-share | 2 | pass |  | 1 | every check passed | [run](../runs/models-openrouter-qwen/cpf-employee-share/trial-2) |
| cpf-total-rate | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-qwen/cpf-total-rate/trial-1) |
| cpf-total-rate | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-qwen/cpf-total-rate/trial-2) |
| iras-non-resident-employment-rate | 1 | fail | navigation | 16 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-qwen/iras-non-resident-employment-rate/trial-1) |
| iras-non-resident-employment-rate | 2 | pass |  | 11 | every check passed | [run](../runs/models-openrouter-qwen/iras-non-resident-employment-rate/trial-2) |
| iras-tax-on-first-80k | 1 | pass |  | 5 | every check passed | [run](../runs/models-openrouter-qwen/iras-tax-on-first-80k/trial-1) |
| iras-tax-on-first-80k | 2 | fail | planning | 5 | answer_matches failed: answer '$4,600' matches none of ['$3,350', '3,350', '3350', 'S$3,350'] | [run](../runs/models-openrouter-qwen/iras-tax-on-first-80k/trial-2) |
| moe-ai-tong-postal-code | 1 | fail | planning | 2 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/models-openrouter-qwen/moe-ai-tong-postal-code/trial-1) |
| moe-ai-tong-postal-code | 2 | fail | planning | 2 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/models-openrouter-qwen/moe-ai-tong-postal-code/trial-2) |
| moe-bishan-boys-secondary | 1 | fail | planning | 16 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-qwen/moe-bishan-boys-secondary/trial-1) |
| moe-bishan-boys-secondary | 2 | fail | planning | 13 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-qwen/moe-bishan-boys-secondary/trial-2) |
| moe-bishan-primary-count | 1 | fail | planning | 20 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-qwen/moe-bishan-primary-count/trial-1) |
| moe-bishan-primary-count | 2 | fail | grounding | 15 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-qwen/moe-bishan-primary-count/trial-2) |
| moe-queenstown-integrated-programme | 1 | fail | planning | 22 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-qwen/moe-queenstown-integrated-programme/trial-1) |
| moe-queenstown-integrated-programme | 2 | fail | grounding | 22 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-qwen/moe-queenstown-integrated-programme/trial-2) |
| mom-annual-leave | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-qwen/mom-annual-leave/trial-1) |
| mom-annual-leave | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-qwen/mom-annual-leave/trial-2) |
| mom-deepavali-2027 | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter-qwen/mom-deepavali-2027/trial-1) |
| mom-deepavali-2027 | 2 | fail | planning | 20 | the agent did not finish the task (run_timeout) | [run](../runs/models-openrouter-qwen/mom-deepavali-2027/trial-2) |
| singstat-latest-cpi | 1 | pass |  | 1 | The SingStat home page clearly displays 2.3% for CPI in Aug 2026 (YoY), matching the agent's answer. | [run](../runs/models-openrouter-qwen/singstat-latest-cpi/trial-1) |
| singstat-latest-cpi | 2 | pass |  | 1 | The SingStat home page clearly displays 2.3% for CPI in Aug 2026 (YoY), matching the agent's answer. | [run](../runs/models-openrouter-qwen/singstat-latest-cpi/trial-2) |
| singstat-total-population | 1 | pass |  | 2 | The agent navigated to the Population tab which clearly shows a Total Population of 6.21 m as at Jun 2026, and accurately reported both the figure and year. | [run](../runs/models-openrouter-qwen/singstat-total-population/trial-1) |
| singstat-total-population | 2 | pass |  | 2 | The final page displays the total population as 6.21 m as at Jun 2026, which matches the agent's reported figure and year. | [run](../runs/models-openrouter-qwen/singstat-total-population/trial-2) |
