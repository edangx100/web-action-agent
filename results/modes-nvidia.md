# Results: modes-nvidia

Suite **sg-gov-v1[cpf-employee-share]** · planner **nvidia** (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`) · library **cold** · judge **openrouter-judge**

Evaluated: sweep started 2026-10-20T06:03:50.992158+00:00; last trial marked 2026-10-20T06:04:05.309814+00:00. Live websites change, so these numbers describe the sites as they were then.

## Summary

| Measure | Value |
|---|---|
| Success rate | 100.0% (1 round only: no range, treat as a rough guide) |
| Trials passed | 1 of 1 marked |
| Trials not yet marked | 0 (left out of every figure) |
| Average steps per trial | 1.0 |
| Average tokens per trial | 2,685 |
| Average time per trial | 11.0 s |
| Average cost per trial (agent) | unknown (1 call unpriced) |
| Cost of judging (not the agent) | none: no judge calls |
| Judge–human agreement: pass or fail | not yet measured (no human labels) |
| Judge–human agreement: kind of failure (trials both called a fail) | not yet measured (no human labels) |

Agreement is the share of hand-labelled trials where the machine's mark and a person's matched. Cohen's κ discounts the agreement the two would reach by chance alone: 1 is perfect, 0 is no better than chance.

## Per task

| Task | Passed | Avg steps | Avg tokens | Avg time | Avg cost |
|---|---|---|---|---|---|
| cpf-employee-share | 1/1 | 1.0 | 2,685 | 11.0 s | unknown (1 call unpriced) |

## Failures by kind

| grounding | planning | navigation | site_change |
|---|---|---|---|
| 0 | 0 | 0 | 0 |

## Every verdict

Each row links to the run folder the verdict was made from.

| Task | Trial | Verdict | Kind of failure | Steps | Reason | Run |
|---|---|---|---|---|---|---|
| cpf-employee-share | 1 | pass |  | 1 | every check passed | [run](../runs/modes-nvidia/cpf-employee-share/trial-1) |
