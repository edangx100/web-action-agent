# Example run: income tax on the first $80,000

One complete run, committed so you can see everything a run leaves behind
without running anything yourself. Every other run folder under `runs/` is
local scratch and is not committed.

**The task in plain words:** *"In Singapore, if your taxable income for the
year is $80,000, how much income tax do you pay?"* The agent must find the
answer on the website of IRAS (Singapore's tax office), starting from its home
page, rather than answer from memory. The answer is in a table on the IRAS
rates page: tax is charged on slices of income at rising rates, and the table's
running total for the first $80,000 is **$3,350**. The row below it, $4,600, is
the tax on the *next* $40,000, an easy row to misread. The
[README's Demo section](../../README.md#demo) shows the table.

| | |
|---|---|
| Task | Starting from the IRAS home page, find the individual income tax rates for tax residents. Using the rates that apply from Year of Assessment 2024 onwards, report the gross tax payable on the first $80,000 of chargeable income. ([task file](../../benchmarks/tasks/iras-tax-on-first-80k.yaml)) |
| Start page | https://www.iras.gov.sg/ |
| Answer | **$3,350** |
| Verdict | **pass**: the answer matched, and the run ended on the tax rates page |
| Steps | 5 (4 clicks, then `done`) |
| Time | 38 s from first step to last |
| Model calls | 5, all to `nvidia/nemotron-3-nano-30b-a3b` through OpenRouter (`openrouter` profile) |
| Tokens and cost | 16,738 tokens, $0.0022 |
| Screenshots sent to the model | none: every page gave at least 5 elements in its text form, so the text was enough |

## What happened, step by step

Each picture is the page as the agent saw it before that step, with the
numbered boxes it was given. The numbers are the ones its decision names.

| Step | Picture | The agent did | Why it worked |
|---|---|---|---|
| 1 | [step-00.png](screenshots/step-00.png) | click [4] "Taxes" | the home page's top menu |
| 2 | [step-01.png](screenshots/step-01.png) | click [8] "Individual Income Tax" | the Taxes menu opened in place, on the same page |
| 3 | [step-02.png](screenshots/step-02.png) | click [39] "Tax residency and tax rates" | the third menu column; the link is partly hidden behind a help pop-up in the picture, but the agent works from the page's text, where it is not hidden |
| 4 | [step-03.png](screenshots/step-03.png) | click [19] "Individual Income Tax rates" | the card that leads to the rates page |
| 5 | [step-04.png](screenshots/step-04.png) | `done`, answer "$3,350" | read from the rates table in the page's text |

One thing this task cannot catch: the rates page has two tables, one from
Year of Assessment 2024 onwards and one for earlier years, and both give
$3,350 for the first $80,000. So the pass shows the agent reached the right
page and read the right row, not that it chose the right table.

## The files

| File | What it holds |
|---|---|
| `run.json` | the task, start page and run id, written when the run opened |
| `config.json` | every setting the run used (profile, limits, timeouts, viewport) |
| `actions.jsonl` | one line per step: start and end times, what the page showed (address, title, numbered elements, text), the decision, what happened, whether a screenshot was sent and why, and that step's model call |
| `calls.jsonl` | the cost log: one line per model call, with tokens, time, cost and the step it belongs to |
| `outcome.json` | how the run ended: the answer, step count and cost totals |
| `verdict.json` | the benchmark's mark: each check, and whether it passed |
| `screenshots/` | the page before each step, with the numbered boxes drawn on |

Every line of `actions.jsonl` is checked against the action log schema
(`src/web_action_agent/schemas/action_log.schema.json`).

## Reading it back

The trace reader rebuilds the run from this folder alone:

```bash
uv run python -c "
from pathlib import Path
from web_action_agent.trace_reader import read_run
print(read_run(Path('runs/example')).summary())
"
```

```
  1  act    click [4] link "Taxes" -> ok (0.19s)
  2  act    click [8] link "Individual Income Tax" -> ok (0.14s)
  3  act    click [39] link "Tax residency and tax rates" -> ok (0.37s)
  4  act    click [19] link "Individual Income Tax rates View individual income tax rates that is applicable to your situation" -> ok (0.31s)
  5  done   success: $3,350
success in 5 steps: $3,350
5 calls, 16,738 tokens, $0.0022, 34.9s thinking
```

To click through it with its screenshots in Opik instead:
`uv run --extra opik waa export --to opik runs/example`.

## Where it came from

It is trial 2 of this task in the `modality-openrouter` sweep (text plus
screenshots when needed), run on 2026-10-20 at commit `b9f8954`. When it was
copied here, one thing was changed: each step's `screenshot_on_disk` path was
rewritten from that sweep's folder to `runs/example/screenshots/`, so it
points at a file that exists in a fresh clone. Nothing else was edited.
