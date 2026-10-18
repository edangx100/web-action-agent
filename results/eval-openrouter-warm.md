# Results: eval-openrouter-warm

Suite **sg-gov-v1** · planner **openrouter** (`nvidia/nemotron-3-nano-30b-a3b`) · library **warm** · judge **openrouter-judge**

Evaluated: sweep started 2026-10-18T05:45:17.772456+00:00; last trial marked 2026-10-18T05:59:02.629772+00:00. Live websites change, so these numbers describe the sites as they were then.

## Summary

| Measure | Value |
|---|---|
| Success rate | 52.8% (range 41.7%–66.7% over 3 rounds) |
| Trials passed | 19 of 36 marked |
| Trials not yet marked | 0 (left out of every figure) |
| Average steps per trial | 9.0 |
| Average tokens per trial | 23,763 |
| Average time per trial | 78.2 s |
| Average cost per trial (agent) | $0.0030 |
| Cost of judging (not the agent) | $0.1039 over 23 calls |
| Judge–human agreement: pass or fail | not yet measured (no human labels) |
| Judge–human agreement: kind of failure (trials both called a fail) | not yet measured (no human labels) |

Agreement is the share of hand-labelled trials where the machine's mark and a person's matched. Cohen's κ discounts the agreement the two would reach by chance alone: 1 is perfect, 0 is no better than chance.

## Per task

| Task | Passed | Avg steps | Avg tokens | Avg time | Avg cost |
|---|---|---|---|---|---|
| cpf-employee-share | 3/3 | 1.0 | 2,785 | 5.0 s | $0.0002 |
| cpf-total-rate | 1/3 | 13.7 | 34,193 | 57.7 s | $0.0042 |
| iras-non-resident-employment-rate | 3/3 | 2.0 | 5,587 | 10.7 s | $0.0004 |
| iras-tax-on-first-80k | 3/3 | 2.0 | 6,749 | 18.8 s | $0.0007 |
| moe-ai-tong-postal-code | 0/3 | 2.0 | 4,911 | 60.6 s | $0.0006 |
| moe-bishan-boys-secondary | 0/3 | 16.7 | 49,358 | 237.1 s | $0.0068 |
| moe-bishan-primary-count | 2/3 | 13.7 | 42,351 | 204.6 s | $0.0056 |
| moe-queenstown-integrated-programme | 1/3 | 4.0 | 14,365 | 119.9 s | $0.0023 |
| mom-annual-leave | 0/3 | 25.0 | 59,840 | 137.5 s | $0.0071 |
| mom-deepavali-2027 | 0/3 | 25.0 | 57,925 | 74.5 s | $0.0069 |
| singstat-latest-cpi | 3/3 | 1.0 | 2,500 | 6.1 s | $0.0002 |
| singstat-total-population | 3/3 | 2.0 | 4,594 | 6.2 s | $0.0004 |

## Failures by kind

| grounding | planning | navigation | site_change |
|---|---|---|---|
| 2 | 14 | 1 | 0 |

## Every verdict

Each row links to the run folder the verdict was made from.

| Task | Trial | Verdict | Kind of failure | Steps | Reason | Run |
|---|---|---|---|---|---|---|
| cpf-employee-share | 1 | pass |  | 1 | every check passed | [run](../runs/eval-openrouter-warm/cpf-employee-share/trial-1) |
| cpf-employee-share | 2 | pass |  | 1 | every check passed | [run](../runs/eval-openrouter-warm/cpf-employee-share/trial-2) |
| cpf-employee-share | 3 | pass |  | 1 | every check passed | [run](../runs/eval-openrouter-warm/cpf-employee-share/trial-3) |
| cpf-total-rate | 1 | fail | planning | 6 | answer_matches failed: answer '28%' matches none of ['37%', '37', '37 percent'] | [run](../runs/eval-openrouter-warm/cpf-total-rate/trial-1) |
| cpf-total-rate | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-warm/cpf-total-rate/trial-2) |
| cpf-total-rate | 3 | pass |  | 10 | every check passed | [run](../runs/eval-openrouter-warm/cpf-total-rate/trial-3) |
| iras-non-resident-employment-rate | 1 | pass |  | 2 | every check passed | [run](../runs/eval-openrouter-warm/iras-non-resident-employment-rate/trial-1) |
| iras-non-resident-employment-rate | 2 | pass |  | 2 | every check passed | [run](../runs/eval-openrouter-warm/iras-non-resident-employment-rate/trial-2) |
| iras-non-resident-employment-rate | 3 | pass |  | 2 | every check passed | [run](../runs/eval-openrouter-warm/iras-non-resident-employment-rate/trial-3) |
| iras-tax-on-first-80k | 1 | pass |  | 2 | every check passed | [run](../runs/eval-openrouter-warm/iras-tax-on-first-80k/trial-1) |
| iras-tax-on-first-80k | 2 | pass |  | 2 | every check passed | [run](../runs/eval-openrouter-warm/iras-tax-on-first-80k/trial-2) |
| iras-tax-on-first-80k | 3 | pass |  | 2 | every check passed | [run](../runs/eval-openrouter-warm/iras-tax-on-first-80k/trial-3) |
| moe-ai-tong-postal-code | 1 | fail | planning | 2 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/eval-openrouter-warm/moe-ai-tong-postal-code/trial-1) |
| moe-ai-tong-postal-code | 2 | fail | planning | 2 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/eval-openrouter-warm/moe-ai-tong-postal-code/trial-2) |
| moe-ai-tong-postal-code | 3 | fail | planning | 2 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/eval-openrouter-warm/moe-ai-tong-postal-code/trial-3) |
| moe-bishan-boys-secondary | 1 | fail | planning | 6 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school?location=Bishan&schooltype=boys' does not contain 'gender=Boys' | [run](../runs/eval-openrouter-warm/moe-bishan-boys-secondary/trial-1) |
| moe-bishan-boys-secondary | 2 | fail | grounding | 19 | the agent did not finish the task (run_timeout) | [run](../runs/eval-openrouter-warm/moe-bishan-boys-secondary/trial-2) |
| moe-bishan-boys-secondary | 3 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-warm/moe-bishan-boys-secondary/trial-3) |
| moe-bishan-primary-count | 1 | fail | grounding | 13 | the agent did not finish the task (run_timeout) | [run](../runs/eval-openrouter-warm/moe-bishan-primary-count/trial-1) |
| moe-bishan-primary-count | 2 | pass |  | 5 | every check passed | [run](../runs/eval-openrouter-warm/moe-bishan-primary-count/trial-2) |
| moe-bishan-primary-count | 3 | pass |  | 23 | every check passed | [run](../runs/eval-openrouter-warm/moe-bishan-primary-count/trial-3) |
| moe-queenstown-integrated-programme | 1 | fail | planning | 3 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/eval-openrouter-warm/moe-queenstown-integrated-programme/trial-1) |
| moe-queenstown-integrated-programme | 2 | fail | planning | 6 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/eval-openrouter-warm/moe-queenstown-integrated-programme/trial-2) |
| moe-queenstown-integrated-programme | 3 | pass |  | 3 | every check passed | [run](../runs/eval-openrouter-warm/moe-queenstown-integrated-programme/trial-3) |
| mom-annual-leave | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-warm/mom-annual-leave/trial-1) |
| mom-annual-leave | 2 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-warm/mom-annual-leave/trial-2) |
| mom-annual-leave | 3 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-warm/mom-annual-leave/trial-3) |
| mom-deepavali-2027 | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-warm/mom-deepavali-2027/trial-1) |
| mom-deepavali-2027 | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-warm/mom-deepavali-2027/trial-2) |
| mom-deepavali-2027 | 3 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-warm/mom-deepavali-2027/trial-3) |
| singstat-latest-cpi | 1 | pass |  | 1 | The home page clearly displays CPI in Aug 2026 (YoY) as 2.3%, which matches the agent's answer. | [run](../runs/eval-openrouter-warm/singstat-latest-cpi/trial-1) |
| singstat-latest-cpi | 2 | pass |  | 1 | The agent correctly extracted the CPI year-on-year change of 2.3% for August 2026 displayed on the home page. | [run](../runs/eval-openrouter-warm/singstat-latest-cpi/trial-2) |
| singstat-latest-cpi | 3 | pass |  | 1 | The home page clearly displays CPI in Aug 2026 (YoY) as 2.3%, which matches the agent's answer. | [run](../runs/eval-openrouter-warm/singstat-latest-cpi/trial-3) |
| singstat-total-population | 1 | pass |  | 2 | The agent navigated to the population tab and correctly reported the total population figure of 6.21 million as of June 2026 shown on the page. | [run](../runs/eval-openrouter-warm/singstat-total-population/trial-1) |
| singstat-total-population | 2 | pass |  | 2 | The final screenshot clearly displays the latest Total Population figure of 6.21 m as of June 2026, matching the agent's report. | [run](../runs/eval-openrouter-warm/singstat-total-population/trial-2) |
| singstat-total-population | 3 | pass |  | 2 | The agent navigated to the Population tab on the SingStat home page and correctly reported the total population of 6.21 million as of June 2026. | [run](../runs/eval-openrouter-warm/singstat-total-population/trial-3) |
