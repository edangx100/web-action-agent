# Results: eval-openrouter-cold

Suite **sg-gov-v1** · planner **openrouter** (`nvidia/nemotron-3-nano-30b-a3b`) · library **cold** · judge **openrouter-judge**

Evaluated: sweep started 2026-10-18T05:26:56.054233+00:00; last trial marked 2026-10-18T05:44:26.406149+00:00. Live websites change, so these numbers describe the sites as they were then.

## Summary

| Measure | Value |
|---|---|
| Success rate | 33.3% (range 25.0%–41.7% over 3 rounds) |
| Trials passed | 12 of 36 marked |
| Trials not yet marked | 0 (left out of every figure) |
| Average steps per trial | 12.6 |
| Average tokens per trial | 33,377 |
| Average time per trial | 108.0 s |
| Average cost per trial (agent) | $0.0042 |
| Cost of judging (not the agent) | $0.1546 over 30 calls |
| Judge–human agreement: pass or fail | not yet measured (no human labels) |
| Judge–human agreement: kind of failure (trials both called a fail) | not yet measured (no human labels) |

Agreement is the share of hand-labelled trials where the machine's mark and a person's matched. Cohen's κ discounts the agreement the two would reach by chance alone: 1 is perfect, 0 is no better than chance.

## Per task

| Task | Passed | Avg steps | Avg tokens | Avg time | Avg cost |
|---|---|---|---|---|---|
| cpf-employee-share | 3/3 | 1.0 | 2,751 | 4.2 s | $0.0002 |
| cpf-total-rate | 0/3 | 25.0 | 46,759 | 69.3 s | $0.0046 |
| iras-non-resident-employment-rate | 1/3 | 20.3 | 46,737 | 91.5 s | $0.0046 |
| iras-tax-on-first-80k | 2/3 | 11.0 | 37,895 | 132.1 s | $0.0052 |
| moe-ai-tong-postal-code | 0/3 | 14.7 | 37,316 | 175.4 s | $0.0049 |
| moe-bishan-boys-secondary | 0/3 | 12.0 | 49,776 | 242.9 s | $0.0082 |
| moe-bishan-primary-count | 0/3 | 7.7 | 28,513 | 198.6 s | $0.0043 |
| moe-queenstown-integrated-programme | 0/3 | 8.0 | 31,420 | 121.6 s | $0.0050 |
| mom-annual-leave | 0/3 | 23.3 | 53,967 | 150.2 s | $0.0065 |
| mom-deepavali-2027 | 0/3 | 25.0 | 58,458 | 95.9 s | $0.0067 |
| singstat-latest-cpi | 3/3 | 1.0 | 2,567 | 7.3 s | $0.0003 |
| singstat-total-population | 3/3 | 2.0 | 4,366 | 6.5 s | $0.0003 |

## Failures by kind

| grounding | planning | navigation | site_change |
|---|---|---|---|
| 3 | 17 | 4 | 0 |

## Every verdict

Each row links to the run folder the verdict was made from.

| Task | Trial | Verdict | Kind of failure | Steps | Reason | Run |
|---|---|---|---|---|---|---|
| cpf-employee-share | 1 | pass |  | 1 | every check passed | [run](../runs/eval-openrouter-cold/cpf-employee-share/trial-1) |
| cpf-employee-share | 2 | pass |  | 1 | every check passed | [run](../runs/eval-openrouter-cold/cpf-employee-share/trial-2) |
| cpf-employee-share | 3 | pass |  | 1 | every check passed | [run](../runs/eval-openrouter-cold/cpf-employee-share/trial-3) |
| cpf-total-rate | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/cpf-total-rate/trial-1) |
| cpf-total-rate | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/cpf-total-rate/trial-2) |
| cpf-total-rate | 3 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/cpf-total-rate/trial-3) |
| iras-non-resident-employment-rate | 1 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/iras-non-resident-employment-rate/trial-1) |
| iras-non-resident-employment-rate | 2 | pass |  | 11 | every check passed | [run](../runs/eval-openrouter-cold/iras-non-resident-employment-rate/trial-2) |
| iras-non-resident-employment-rate | 3 | fail | navigation | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/iras-non-resident-employment-rate/trial-3) |
| iras-tax-on-first-80k | 1 | fail | navigation | 22 | the agent did not finish the task (run_timeout) | [run](../runs/eval-openrouter-cold/iras-tax-on-first-80k/trial-1) |
| iras-tax-on-first-80k | 2 | pass |  | 5 | every check passed | [run](../runs/eval-openrouter-cold/iras-tax-on-first-80k/trial-2) |
| iras-tax-on-first-80k | 3 | pass |  | 6 | every check passed | [run](../runs/eval-openrouter-cold/iras-tax-on-first-80k/trial-3) |
| moe-ai-tong-postal-code | 1 | fail | grounding | 15 | the agent did not finish the task (run_timeout) | [run](../runs/eval-openrouter-cold/moe-ai-tong-postal-code/trial-1) |
| moe-ai-tong-postal-code | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/moe-ai-tong-postal-code/trial-2) |
| moe-ai-tong-postal-code | 3 | fail | grounding | 4 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/primary%20school' does not contain 'keyword=Ai' | [run](../runs/eval-openrouter-cold/moe-ai-tong-postal-code/trial-3) |
| moe-bishan-boys-secondary | 1 | fail | planning | 7 | the agent did not finish the task (run_timeout) | [run](../runs/eval-openrouter-cold/moe-bishan-boys-secondary/trial-1) |
| moe-bishan-boys-secondary | 2 | fail | planning | 21 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school?area=Bishan&schooltype=boys' does not contain 'gender=Boys' | [run](../runs/eval-openrouter-cold/moe-bishan-boys-secondary/trial-2) |
| moe-bishan-boys-secondary | 3 | fail | planning | 8 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school?location=Bishan&schooltype=boys' does not contain 'gender=Boys' | [run](../runs/eval-openrouter-cold/moe-bishan-boys-secondary/trial-3) |
| moe-bishan-primary-count | 1 | fail | grounding | 7 | answer_matches failed: answer '182' matches none of ['3', 'three'] | [run](../runs/eval-openrouter-cold/moe-bishan-primary-count/trial-1) |
| moe-bishan-primary-count | 2 | fail | planning | 6 | answer_matches failed: answer '182' matches none of ['3', 'three'] | [run](../runs/eval-openrouter-cold/moe-bishan-primary-count/trial-2) |
| moe-bishan-primary-count | 3 | fail | planning | 10 | answer_matches failed: answer '182' matches none of ['3', 'three'] | [run](../runs/eval-openrouter-cold/moe-bishan-primary-count/trial-3) |
| moe-queenstown-integrated-programme | 1 | fail | planning | 4 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/eval-openrouter-cold/moe-queenstown-integrated-programme/trial-1) |
| moe-queenstown-integrated-programme | 2 | fail | planning | 11 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/eval-openrouter-cold/moe-queenstown-integrated-programme/trial-2) |
| moe-queenstown-integrated-programme | 3 | fail | planning | 9 | url_contains failed: final URL 'https://www.moe.gov.sg/schoolfinder/secondary%20school' does not contain 'Integrated%20Programme' | [run](../runs/eval-openrouter-cold/moe-queenstown-integrated-programme/trial-3) |
| mom-annual-leave | 1 | fail | navigation | 20 | the agent did not finish the task (run_timeout) | [run](../runs/eval-openrouter-cold/mom-annual-leave/trial-1) |
| mom-annual-leave | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/mom-annual-leave/trial-2) |
| mom-annual-leave | 3 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/mom-annual-leave/trial-3) |
| mom-deepavali-2027 | 1 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/mom-deepavali-2027/trial-1) |
| mom-deepavali-2027 | 2 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/mom-deepavali-2027/trial-2) |
| mom-deepavali-2027 | 3 | fail | planning | 25 | the agent did not finish the task (step_cap) | [run](../runs/eval-openrouter-cold/mom-deepavali-2027/trial-3) |
| singstat-latest-cpi | 1 | pass |  | 1 | The agent correctly extracted the CPI year-on-year change of 2.3% for August 2026 displayed on the home page. | [run](../runs/eval-openrouter-cold/singstat-latest-cpi/trial-1) |
| singstat-latest-cpi | 2 | pass |  | 1 | The home page clearly displays CPI in Aug 2026 (YoY) as 2.3%, which matches the agent's answer. | [run](../runs/eval-openrouter-cold/singstat-latest-cpi/trial-2) |
| singstat-latest-cpi | 3 | pass |  | 1 | The agent correctly extracted the latest year-on-year CPI change of 2.3% for August 2026 directly from the home page. | [run](../runs/eval-openrouter-cold/singstat-latest-cpi/trial-3) |
| singstat-total-population | 1 | pass |  | 2 | The agent navigated to the Population tab which clearly shows a total population of 6.21 m as of Jun 2026, matching its final answer. | [run](../runs/eval-openrouter-cold/singstat-total-population/trial-1) |
| singstat-total-population | 2 | pass |  | 2 | The agent navigated to the Population section and correctly reported the figure (6.21 m) and the year (2026) as displayed on the page. | [run](../runs/eval-openrouter-cold/singstat-total-population/trial-2) |
| singstat-total-population | 3 | pass |  | 2 | The agent navigated to the Population tab on the SingStat home page, which clearly displays a total population of 6.21 m as at Jun 2026, and accurately reported this figure and year. | [run](../runs/eval-openrouter-cold/singstat-total-population/trial-3) |
