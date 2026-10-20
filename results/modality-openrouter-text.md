# Results: modality-openrouter-text

Suite **sg-gov-v1** · planner **openrouter-text** (`nvidia/nemotron-3-nano-30b-a3b`) · library **cold** · judge **openrouter-judge**

Evaluated: sweep started 2026-10-20T11:35:24.896427+00:00; last trial marked 2026-10-20T11:44:26.445346+00:00. Live websites change, so these numbers describe the sites as they were then.

## Summary

| Measure | Value |
|---|---|
| Success rate | 41.7% (range 41.7%–41.7% over 2 rounds) |
| Trials passed | 10 of 24 marked |
| Trials not yet marked | 0 (left out of every figure) |
| Average steps per trial | 10.6 |
| Average tokens per trial | 24,966 |
| Average time per trial | 77.1 s |
| Average cost per trial (agent) | $0.0029 |
| Cost of judging (not the agent) | $0.0957 over 18 calls |
| Judge–human agreement: pass or fail | not yet measured (no human labels) |
| Judge–human agreement: kind of failure (trials both called a fail) | not yet measured (no human labels) |

Agreement is the share of hand-labelled trials where the machine's mark and a person's matched. Cohen's κ discounts the agreement the two would reach by chance alone: 1 is perfect, 0 is no better than chance.

## Per task

| Task | Passed | Avg steps | Avg tokens | Avg time | Avg cost |
|---|---|---|---|---|---|
| cpf-employee-share | 2/2 | 1.0 | 2,818 | 4.7 s | $0.0003 |
| cpf-total-rate | 0/2 | 25.0 | 52,848 | 79.6 s | $0.0059 |
| iras-non-resident-employment-rate | 2/2 | 15.5 | 35,438 | 70.8 s | $0.0033 |
| iras-tax-on-first-80k | 2/2 | 5.0 | 15,168 | 30.4 s | $0.0018 |
| moe-ai-tong-postal-code | 0/2 | 5.5 | 14,120 | 62.1 s | $0.0017 |
| moe-bishan-boys-secondary | 0/2 | 7.0 | 26,122 | 116.5 s | $0.0041 |
| moe-bishan-primary-count | 0/2 | 11.5 | 40,334 | 164.2 s | $0.0058 |
| moe-queenstown-integrated-programme | 0/2 | 3.0 | 9,300 | 56.5 s | $0.0014 |
| mom-annual-leave | 0/2 | 25.0 | 50,198 | 187.1 s | $0.0051 |
| mom-deepavali-2027 | 0/2 | 25.0 | 45,142 | 144.0 s | $0.0043 |
| singstat-latest-cpi | 2/2 | 1.0 | 2,503 | 3.5 s | $0.0002 |
| singstat-total-population | 2/2 | 2.5 | 5,594 | 5.8 s | $0.0004 |

## Failures by kind

| grounding | planning | navigation | site_change |
|---|---|---|---|
| 1 | 10 | 3 | 0 |

## Every verdict

Each row links to the run folder the verdict was made from.

| Task | Trial | Verdict | Kind of failure | Steps | Reason | Run |
|---|---|---|---|---|---|---|
| cpf-employee-share | 1 | pass |  | 1 | every check passed | [run](../runs/modality-openrouter-text/cpf-employee-share/trial-1) |
| cpf-employee-share | 2 | pass |  | 1 | every check passed | [run](../runs/modality-openrouter-text/cpf-employee-share/trial-2) |
| cpf-total-rate | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter-text/cpf-total-rate/trial-1) |
| cpf-total-rate | 2 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter-text/cpf-total-rate/trial-2) |
| iras-non-resident-employment-rate | 1 | pass |  | 21 | every check passed | [run](../runs/modality-openrouter-text/iras-non-resident-employment-rate/trial-1) |
| iras-non-resident-employment-rate | 2 | pass |  | 10 | every check passed | [run](../runs/modality-openrouter-text/iras-non-resident-employment-rate/trial-2) |
| iras-tax-on-first-80k | 1 | pass |  | 5 | every check passed | [run](../runs/modality-openrouter-text/iras-tax-on-first-80k/trial-1) |
| iras-tax-on-first-80k | 2 | pass |  | 5 | every check passed | [run](../runs/modality-openrouter-text/iras-tax-on-first-80k/trial-2) |
| moe-ai-tong-postal-code | 1 | fail | planning | 9 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/modality-openrouter-text/moe-ai-tong-postal-code/trial-1) |
| moe-ai-tong-postal-code | 2 | fail | planning | 2 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/modality-openrouter-text/moe-ai-tong-postal-code/trial-2) |
| moe-bishan-boys-secondary | 1 | fail | planning | 7 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school?location=Bishan&schooltype=boys' does not contain 'gender=Boys' | [run](../runs/modality-openrouter-text/moe-bishan-boys-secondary/trial-1) |
| moe-bishan-boys-secondary | 2 | fail | planning | 7 | the judge failed it: The agent did not filter by the Bishan area and reported all 11 boys' secondary schools across Singapore instead of only Catholic High School (Secondary) and Raffles Institution (Secondary). | [run](../runs/modality-openrouter-text/moe-bishan-boys-secondary/trial-2) |
| moe-bishan-primary-count | 1 | fail | grounding | 9 | answer_matches failed: answer '1' matches none of ['3', 'three'] | [run](../runs/modality-openrouter-text/moe-bishan-primary-count/trial-1) |
| moe-bishan-primary-count | 2 | fail | planning | 14 | answer_matches failed: answer '182' matches none of ['3', 'three'] | [run](../runs/modality-openrouter-text/moe-bishan-primary-count/trial-2) |
| moe-queenstown-integrated-programme | 1 | fail | planning | 3 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/modality-openrouter-text/moe-queenstown-integrated-programme/trial-1) |
| moe-queenstown-integrated-programme | 2 | fail | planning | 3 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/modality-openrouter-text/moe-queenstown-integrated-programme/trial-2) |
| mom-annual-leave | 1 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter-text/mom-annual-leave/trial-1) |
| mom-annual-leave | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter-text/mom-annual-leave/trial-2) |
| mom-deepavali-2027 | 1 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter-text/mom-deepavali-2027/trial-1) |
| mom-deepavali-2027 | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/modality-openrouter-text/mom-deepavali-2027/trial-2) |
| singstat-latest-cpi | 1 | pass |  | 1 | The SingStat homepage shows '2.3%' for 'CPI in Aug 2026 (YoY)', which matches the agent's reported answer. | [run](../runs/modality-openrouter-text/singstat-latest-cpi/trial-1) |
| singstat-latest-cpi | 2 | pass |  | 1 | The SingStat homepage shows '2.3%' for 'CPI in Aug 2026 (YoY)', which matches the agent's reported answer. | [run](../runs/modality-openrouter-text/singstat-latest-cpi/trial-2) |
| singstat-total-population | 1 | pass |  | 3 | The agent navigated to the Population tab and correctly reported the total population figure of 6.21 million as of June 2026 shown on the page. | [run](../runs/modality-openrouter-text/singstat-total-population/trial-1) |
| singstat-total-population | 2 | pass |  | 2 | The agent navigated to the Population tab on the Department of Statistics homepage and correctly reported the latest figure of 6.21 m for 2026 as shown on the page. | [run](../runs/modality-openrouter-text/singstat-total-population/trial-2) |
