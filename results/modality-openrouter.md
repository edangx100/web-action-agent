# Results: modality-openrouter

Suite **sg-gov-v1** · planner **openrouter** (`nvidia/nemotron-3-nano-30b-a3b`) · library **cold** · judge **openrouter-judge**

Evaluated: sweep started 2026-10-20T11:27:27.774787+00:00; last trial marked 2026-10-20T11:35:23.797492+00:00. Live websites change, so these numbers describe the sites as they were then.

## Summary

| Measure | Value |
|---|---|
| Success rate | 45.8% (range 41.7%–50.0% over 2 rounds) |
| Trials passed | 11 of 24 marked |
| Trials not yet marked | 0 (left out of every figure) |
| Average steps per trial | 9.5 |
| Average tokens per trial | 26,036 |
| Average time per trial | 67.9 s |
| Average cost per trial (agent) | $0.0032 |
| Cost of judging (not the agent) | $0.0867 over 17 calls |
| Judge–human agreement: pass or fail | 100.0% (3 of 3 trials) · Cohen's κ 1.00 (almost perfect) · only 3 labelled, fewer than the 20 needed to rely on this: treat as a rough guide |
| Judge–human agreement: kind of failure (trials both called a fail) | 50.0% (1 of 2 trials) · Cohen's κ 0.00 (slight) · only 2 labelled, fewer than the 20 needed to rely on this: treat as a rough guide |

Agreement is the share of hand-labelled trials where the machine's mark and a person's matched. Cohen's κ discounts the agreement the two would reach by chance alone: 1 is perfect, 0 is no better than chance.

## Per task

| Task | Passed | Avg steps | Avg tokens | Avg time | Avg cost |
|---|---|---|---|---|---|
| cpf-employee-share | 2/2 | 1.0 | 2,671 | 3.5 s | $0.0002 |
| cpf-total-rate | 1/2 | 14.0 | 39,416 | 33.8 s | $0.0043 |
| iras-non-resident-employment-rate | 2/2 | 18.0 | 42,200 | 89.9 s | $0.0040 |
| iras-tax-on-first-80k | 1/2 | 5.0 | 16,262 | 36.5 s | $0.0020 |
| moe-ai-tong-postal-code | 0/2 | 2.5 | 8,192 | 52.2 s | $0.0013 |
| moe-bishan-boys-secondary | 0/2 | 9.5 | 43,078 | 173.8 s | $0.0071 |
| moe-bishan-primary-count | 1/2 | 7.5 | 23,798 | 81.9 s | $0.0034 |
| moe-queenstown-integrated-programme | 0/2 | 3.5 | 10,900 | 62.6 s | $0.0016 |
| mom-annual-leave | 0/2 | 25.0 | 66,707 | 146.5 s | $0.0078 |
| mom-deepavali-2027 | 0/2 | 25.0 | 52,363 | 125.7 s | $0.0055 |
| singstat-latest-cpi | 2/2 | 1.0 | 2,459 | 4.4 s | $0.0002 |
| singstat-total-population | 2/2 | 2.0 | 4,384 | 4.4 s | $0.0003 |

## Failures by kind

| grounding | planning | navigation | site_change |
|---|---|---|---|
| 1 | 9 | 3 | 0 |

## Every verdict

Each row links to the run folder the verdict was made from.

| Task | Trial | Verdict | Kind of failure | Steps | Reason | Run |
|---|---|---|---|---|---|---|
| cpf-employee-share | 1 | pass |  | 1 | every check passed | [run](../runs/modality-openrouter/cpf-employee-share/trial-1) |
| cpf-employee-share | 2 | pass |  | 1 | every check passed | [run](../runs/modality-openrouter/cpf-employee-share/trial-2) |
| cpf-total-rate | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter/cpf-total-rate/trial-1) |
| cpf-total-rate | 2 | pass |  | 3 | every check passed | [run](../runs/modality-openrouter/cpf-total-rate/trial-2) |
| iras-non-resident-employment-rate | 1 | pass |  | 19 | every check passed | [run](../runs/modality-openrouter/iras-non-resident-employment-rate/trial-1) |
| iras-non-resident-employment-rate | 2 | pass |  | 17 | every check passed | [run](../runs/modality-openrouter/iras-non-resident-employment-rate/trial-2) |
| iras-tax-on-first-80k | 1 | fail | grounding | 5 | answer_matches failed: answer '4,600' matches none of ['$3,350', '3,350', '3350', 'S$3,350'] | [run](../runs/modality-openrouter/iras-tax-on-first-80k/trial-1) |
| iras-tax-on-first-80k | 2 | pass |  | 5 | every check passed | [run](../runs/modality-openrouter/iras-tax-on-first-80k/trial-2) |
| moe-ai-tong-postal-code | 1 | fail | planning | 3 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/modality-openrouter/moe-ai-tong-postal-code/trial-1) |
| moe-ai-tong-postal-code | 2 | fail | planning | 2 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/modality-openrouter/moe-ai-tong-postal-code/trial-2) |
| moe-bishan-boys-secondary | 1 | fail | planning | 7 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school?location=Bishan&schooltype=boys' does not contain 'gender=Boys' | [run](../runs/modality-openrouter/moe-bishan-boys-secondary/trial-1) |
| moe-bishan-boys-secondary | 2 | fail | planning | 12 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school?location=Bishan&schoolType=boys' does not contain 'gender=Boys' | [run](../runs/modality-openrouter/moe-bishan-boys-secondary/trial-2) |
| moe-bishan-primary-count | 1 | pass |  | 3 | every check passed | [run](../runs/modality-openrouter/moe-bishan-primary-count/trial-1) |
| moe-bishan-primary-count | 2 | fail | planning | 12 | answer_matches failed: answer '1' matches none of ['3', 'three'] | [run](../runs/modality-openrouter/moe-bishan-primary-count/trial-2) |
| moe-queenstown-integrated-programme | 1 | fail | planning | 3 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/modality-openrouter/moe-queenstown-integrated-programme/trial-1) |
| moe-queenstown-integrated-programme | 2 | fail | planning | 4 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/modality-openrouter/moe-queenstown-integrated-programme/trial-2) |
| mom-annual-leave | 1 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter/mom-annual-leave/trial-1) |
| mom-annual-leave | 2 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter/mom-annual-leave/trial-2) |
| mom-deepavali-2027 | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter/mom-deepavali-2027/trial-1) |
| mom-deepavali-2027 | 2 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter/mom-deepavali-2027/trial-2) |
| singstat-latest-cpi | 1 | pass |  | 1 | The SingStat homepage shows '2.3%' for 'CPI in Aug 2026 (YoY)', which matches the agent's reported answer. | [run](../runs/modality-openrouter/singstat-latest-cpi/trial-1) |
| singstat-latest-cpi | 2 | pass |  | 1 | The SingStat homepage shows '2.3%' for 'CPI in Aug 2026 (YoY)', which matches the agent's reported answer. | [run](../runs/modality-openrouter/singstat-latest-cpi/trial-2) |
| singstat-total-population | 1 | pass |  | 2 | The agent navigated to the population tab and correctly reported the total population figure of 6.21 m as at June 2026 shown on the page. | [run](../runs/modality-openrouter/singstat-total-population/trial-1) |
| singstat-total-population | 2 | pass |  | 2 | The agent navigated to the Population tab on the home page and correctly reported Singapore's total population of 6.21 million as of June 2026 shown on the page. | [run](../runs/modality-openrouter/singstat-total-population/trial-2) |
