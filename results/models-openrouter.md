# Results: models-openrouter

Suite **sg-gov-v1** · planner **openrouter** (`nvidia/nemotron-3-nano-30b-a3b`) · library **cold** · judge **openrouter-judge**

Evaluated: sweep started 2026-10-19T09:48:13.199547+00:00; last trial marked 2026-10-19T09:57:12.174494+00:00. Live websites change, so these numbers describe the sites as they were then.

## Summary

| Measure | Value |
|---|---|
| Success rate | 41.7% (range 41.7%–41.7% over 2 rounds) |
| Trials passed | 10 of 24 marked |
| Trials not yet marked | 0 (left out of every figure) |
| Average steps per trial | 10.8 |
| Average tokens per trial | 32,005 |
| Average time per trial | 78.2 s |
| Average cost per trial (agent) | $0.0042 |
| Cost of judging (not the agent) | $0.0991 over 18 calls |
| Judge–human agreement: pass or fail | not yet measured (no human labels) |
| Judge–human agreement: kind of failure (trials both called a fail) | not yet measured (no human labels) |

Agreement is the share of hand-labelled trials where the machine's mark and a person's matched. Cohen's κ discounts the agreement the two would reach by chance alone: 1 is perfect, 0 is no better than chance.

## Per task

| Task | Passed | Avg steps | Avg tokens | Avg time | Avg cost |
|---|---|---|---|---|---|
| cpf-employee-share | 2/2 | 1.0 | 2,698 | 3.3 s | $0.0002 |
| cpf-total-rate | 0/2 | 18.5 | 44,334 | 46.9 s | $0.0048 |
| iras-non-resident-employment-rate | 2/2 | 9.0 | 21,814 | 28.2 s | $0.0021 |
| iras-tax-on-first-80k | 2/2 | 5.0 | 26,725 | 68.1 s | $0.0042 |
| moe-ai-tong-postal-code | 0/2 | 14.0 | 39,176 | 105.2 s | $0.0056 |
| moe-bishan-boys-secondary | 0/2 | 21.5 | 76,760 | 213.7 s | $0.0112 |
| moe-bishan-primary-count | 0/2 | 10.0 | 35,216 | 108.9 s | $0.0050 |
| moe-queenstown-integrated-programme | 0/2 | 12.5 | 42,093 | 168.2 s | $0.0059 |
| mom-annual-leave | 0/2 | 9.5 | 27,873 | 112.3 s | $0.0031 |
| mom-deepavali-2027 | 0/2 | 25.0 | 58,684 | 70.7 s | $0.0071 |
| singstat-latest-cpi | 2/2 | 1.5 | 4,029 | 6.5 s | $0.0004 |
| singstat-total-population | 2/2 | 2.0 | 4,658 | 6.2 s | $0.0004 |

## Failures by kind

| grounding | planning | navigation | site_change |
|---|---|---|---|
| 4 | 10 | 0 | 0 |

## Every verdict

Each row links to the run folder the verdict was made from.

| Task | Trial | Verdict | Kind of failure | Steps | Reason | Run |
|---|---|---|---|---|---|---|
| cpf-employee-share | 1 | pass |  | 1 | every check passed | [run](../runs/models-openrouter/cpf-employee-share/trial-1) |
| cpf-employee-share | 2 | pass |  | 1 | every check passed | [run](../runs/models-openrouter/cpf-employee-share/trial-2) |
| cpf-total-rate | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter/cpf-total-rate/trial-1) |
| cpf-total-rate | 2 | fail | grounding | 12 | the agent did not finish the task (validation_exhausted) | [run](../runs/models-openrouter/cpf-total-rate/trial-2) |
| iras-non-resident-employment-rate | 1 | pass |  | 14 | every check passed | [run](../runs/models-openrouter/iras-non-resident-employment-rate/trial-1) |
| iras-non-resident-employment-rate | 2 | pass |  | 4 | every check passed | [run](../runs/models-openrouter/iras-non-resident-employment-rate/trial-2) |
| iras-tax-on-first-80k | 1 | pass |  | 5 | every check passed | [run](../runs/models-openrouter/iras-tax-on-first-80k/trial-1) |
| iras-tax-on-first-80k | 2 | pass |  | 5 | every check passed | [run](../runs/models-openrouter/iras-tax-on-first-80k/trial-2) |
| moe-ai-tong-postal-code | 1 | fail | grounding | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter/moe-ai-tong-postal-code/trial-1) |
| moe-ai-tong-postal-code | 2 | fail | grounding | 3 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/models-openrouter/moe-ai-tong-postal-code/trial-2) |
| moe-bishan-boys-secondary | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter/moe-bishan-boys-secondary/trial-1) |
| moe-bishan-boys-secondary | 2 | fail | planning | 18 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school?area=Bishan&type=boys' does not contain 'gender=Boys' | [run](../runs/models-openrouter/moe-bishan-boys-secondary/trial-2) |
| moe-bishan-primary-count | 1 | fail | planning | 13 | answer_matches failed: answer '1' matches none of ['3', 'three'] | [run](../runs/models-openrouter/moe-bishan-primary-count/trial-1) |
| moe-bishan-primary-count | 2 | fail | planning | 7 | answer_matches failed: answer '182' matches none of ['3', 'three'] | [run](../runs/models-openrouter/moe-bishan-primary-count/trial-2) |
| moe-queenstown-integrated-programme | 1 | fail | planning | 21 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school?location=Queenstown' does not contain 'Integrated%20Programme' | [run](../runs/models-openrouter/moe-queenstown-integrated-programme/trial-1) |
| moe-queenstown-integrated-programme | 2 | fail | planning | 4 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/models-openrouter/moe-queenstown-integrated-programme/trial-2) |
| mom-annual-leave | 1 | fail | grounding | 13 | the agent did not finish the task (validation_exhausted) | [run](../runs/models-openrouter/mom-annual-leave/trial-1) |
| mom-annual-leave | 2 | fail | planning | 6 | the agent did not finish the task (validation_exhausted) | [run](../runs/models-openrouter/mom-annual-leave/trial-2) |
| mom-deepavali-2027 | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter/mom-deepavali-2027/trial-1) |
| mom-deepavali-2027 | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/models-openrouter/mom-deepavali-2027/trial-2) |
| singstat-latest-cpi | 1 | pass |  | 2 | The agent correctly retrieved the year-on-year CPI change of 2.3% for August 2026 directly from the home page indicator. | [run](../runs/models-openrouter/singstat-latest-cpi/trial-1) |
| singstat-latest-cpi | 2 | pass |  | 1 | The SingStat homepage shows '2.3%' for 'CPI in Aug 2026 (YoY)', which matches the agent's reported answer. | [run](../runs/models-openrouter/singstat-latest-cpi/trial-2) |
| singstat-total-population | 1 | pass |  | 2 | The final page displays the total population as 6.21 m as of June 2026, which matches the agent's reported answer. | [run](../runs/models-openrouter/singstat-total-population/trial-1) |
| singstat-total-population | 2 | pass |  | 2 | The agent navigated to the Population tab on the Department of Statistics homepage and correctly reported the total population figure of 6.21 m for June 2026 as displayed. | [run](../runs/models-openrouter/singstat-total-population/trial-2) |
