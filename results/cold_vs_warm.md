# Cold vs. warm skill library

The same tasks on the same model, twice. **Cold**: an empty skill library. **Warm**: the library learnt from the cold pass's passing trials, each skill checked on a live page before it was kept. Success rate is the mean over rounds of trials, with the lowest and highest round. A change in success rate is in percentage points; other changes also give the relative size.

## openrouter (`nvidia/nemotron-3-nano-30b-a3b`)

Suite **sg-gov-v1** · cold [eval-openrouter-cold](eval-openrouter-cold.md), started 2026-10-18T05:26:56.054233+00:00 · warm [eval-openrouter-warm](eval-openrouter-warm.md), started 2026-10-18T05:45:17.772456+00:00

| Measure | Cold | Warm | Change |
|---|---|---|---|
| Success rate (all tasks) | 33.3% (range 25.0%–41.7% over 3 rounds) | 52.8% (range 41.7%–66.7% over 3 rounds) | +19.4 points |
| …on tasks where a skill was used (3) | 6/9 (66.7%), 11.1 steps | 9/9 (100.0%), 2.0 steps | +33.3 points; steps -9.1 (-82.0%) |
| …on tasks where none was (9) | 6/27 (22.2%), 13.1 steps | 10/27 (37.0%), 11.3 steps | +14.8 points; steps -1.7 (-13.3%) |
| Trials marked (N) | 36 (12 tasks × 3) | 36 (12 tasks × 3) | |
| Trials passed | 12 | 19 | +7 |
| Average steps per trial | 12.6 | 9.0 | -3.6 (-28.5%) |
| Average tokens per trial | 33,377 | 23,763 | -9,613.8 (-28.8%) |
| Average time per trial | 108.0 s | 78.2 s | -29.7 s (-27.5%) |
| Average cost per trial (agent) | $0.0042 | $0.0030 | -$0.0013 (-30.2%) |
| Skill library at the start | empty | 3 skills (3 active) on 2 sites | |

**Read the split before the headline.** A task counts as *a skill was used* when any of its warm trials replayed a learnt skill. Only those tasks can show the effect of learning. On the others the warm pass ran exactly as the cold one did, so their change is how much the results move between passes by chance alone. The split rows count passes over all marked trials, rather than averaging rounds.

### Per task

| Task | Skill used (warm) | Cold passed | Warm passed | Cold avg steps | Warm avg steps |
|---|---|---|---|---|---|
| cpf-employee-share | no | 3/3 | 3/3 | 1.0 | 1.0 |
| cpf-total-rate | no | 0/3 | 1/3 | 25.0 | 13.7 |
| iras-non-resident-employment-rate | yes | 1/3 | 3/3 | 20.3 | 2.0 |
| iras-tax-on-first-80k | yes | 2/3 | 3/3 | 11.0 | 2.0 |
| moe-ai-tong-postal-code | no | 0/3 | 0/3 | 14.7 | 2.0 |
| moe-bishan-boys-secondary | no | 0/3 | 0/3 | 12.0 | 16.7 |
| moe-bishan-primary-count | no | 0/3 | 2/3 | 7.7 | 13.7 |
| moe-queenstown-integrated-programme | no | 0/3 | 1/3 | 8.0 | 4.0 |
| mom-annual-leave | no | 0/3 | 0/3 | 23.3 | 25.0 |
| mom-deepavali-2027 | no | 0/3 | 0/3 | 25.0 | 25.0 |
| singstat-latest-cpi | no | 3/3 | 3/3 | 1.0 | 1.0 |
| singstat-total-population | yes | 3/3 | 3/3 | 2.0 | 2.0 |
