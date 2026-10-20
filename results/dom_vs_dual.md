# DOM only vs. DOM + screenshots

The same benchmark on the same planner model, twice. **DOM only**: the page as text alone -- the numbered list of things to click, and the page's words -- with the screenshot channel removed (a profile with no screenshot model). **DOM + screenshots**: the usual setting, where a screenshot is added only when few elements were found or the planner keeps failing on text. An image counts as sent only when the request that went to the model carried one.

## Results

| Arm | Sees the page as | Planner model | Screenshot model | Trials marked (N) | Success rate | Calls with an image | Avg steps | Avg tokens | Avg time | Avg cost (agent) | Code |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [openrouter](modality-openrouter.md) | DOM + screenshots when gated in | `nvidia/nemotron-3-nano-30b-a3b` | `nvidia/nemotron-3.5-super-vl` | 24 (12 tasks × 2) | 45.8% (range 41.7%–50.0% over 2 rounds) | 68 of 235 (in 4 trials) | 9.5 | 26,036 | 67.9 s | $0.0032 | `b9f89544ace8` |
| [openrouter-text](modality-openrouter-text.md) | DOM only | `nvidia/nemotron-3-nano-30b-a3b` | none | 24 (12 tasks × 2) | 41.7% (range 41.7%–41.7% over 2 rounds) | 0 of 256 (in 0 trials) | 10.6 | 24,966 | 77.1 s | $0.0029 | `b9f89544ace8` |

## What the screenshots bought, and what they cost

The change from text alone to text plus screenshots, per trial.

| Measure | DOM only | DOM + screenshots | Change |
|---|---|---|---|
| Success rate | 41.7% (range 41.7%–41.7% over 2 rounds) | 45.8% (range 41.7%–50.0% over 2 rounds) | +4.2 points |
| Trials passed | 10 of 24 | 11 of 24 | |
| Average steps | 10.6 | 9.5 | -1.1 (-10.2%) |
| Average tokens | 24,966 | 26,036 | +1,070.2 (+4.3%) |
| Average time | 77.1 s | 67.9 s | -9.2 s (-11.9%) |
| Average cost (agent) | $0.0029 | $0.0032 | +$0.0003 (+10.3%) |

## Did only the screenshot channel change?

| Check | Result |
|---|---|
| No image in any request of the DOM-only arm | yes: 0 of 256 calls carried an image |
| Same code in every arm | yes: every trial of every arm ran commit `b9f89544ace8`, with nothing uncommitted |
| Same planner model | yes: nvidia/nemotron-3-nano-30b-a3b |
| Same benchmark suite | yes: sg-gov-v1 |
| Same judge | yes: openrouter-judge |
| Empty skill library in every arm | yes: cold |

## Per task

Trials passed out of trials marked, and in brackets the calls that carried an image.

| Task | openrouter | openrouter-text |
|---|---|---|
| cpf-employee-share | 2/2 (0) | 2/2 (0) |
| cpf-total-rate | 1/2 (15) | 0/2 (0) |
| iras-non-resident-employment-rate | 2/2 (0) | 2/2 (0) |
| iras-tax-on-first-80k | 1/2 (0) | 2/2 (0) |
| moe-ai-tong-postal-code | 0/2 (0) | 0/2 (0) |
| moe-bishan-boys-secondary | 0/2 (0) | 0/2 (0) |
| moe-bishan-primary-count | 1/2 (0) | 0/2 (0) |
| moe-queenstown-integrated-programme | 0/2 (0) | 0/2 (0) |
| mom-annual-leave | 0/2 (34) | 0/2 (0) |
| mom-deepavali-2027 | 0/2 (19) | 0/2 (0) |
| singstat-latest-cpi | 2/2 (0) | 2/2 (0) |
| singstat-total-population | 2/2 (0) | 2/2 (0) |
