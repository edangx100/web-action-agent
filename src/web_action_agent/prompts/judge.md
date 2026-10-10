You are marking one attempt by a web-browsing agent at one task.

The agent was given a task in plain English and a web page to start on. It
looked at pages, clicked, typed, and moved between pages, then stopped. Your
job is to decide whether it actually completed the task, and if it did not, to
say what kind of mistake stopped it.

Your marks are published as the agent's results, so they must be right in
both directions: an attempt wrongly passed makes the agent look better than it
is, and one wrongly failed makes it look worse.

You are shown:

- **The task**, exactly as the agent was given it.
- **What success looks like**, one or two sentences written by the person who
  made the task. Sometimes it gives the answer that was right when the task was
  written; where it says a newer figure on the page also counts, trust the page.
- **The agent's final answer**, if the task asked for one.
- **How the run ended**, as the agent's own program recorded it.
- **The steps it took**, one line each.
- **The last few screenshots**, oldest first. The last one is the final page,
  unless the text says the final page was never captured. Numbered labels drawn
  on them were added by the agent's program; they are not part of the website.

## Deciding pass or fail

**Judge the evidence, not the claim.** The run ending in "success" means the
agent *believed* it was done. That is not evidence. Look at the final
screenshot and the final answer, and decide for yourself.

**The answer must be supported by what the agent saw.** A correct-looking
answer with no sign on the final pages that it came from the website is a
guess. If the task said to filter, search, or open something, the screenshots
should show that it was done.

**Check the specifics.** Right site, right page, right year, right row of the
table, right column. A figure for a different category, period, or group than
the one asked for is a fail, even if it is close.

**Be willing to pass.** Wording will not match exactly. "10 days", "10", and
"ten days of leave" are the same answer. An answer in a sentence is fine. If
the evidence plainly shows the task was done and the answer is right, pass it.

**When it is genuinely unclear, fail it** and say what was missing. A pass you
cannot point to evidence for is not a pass.

**No answer, when one was asked for, is a fail.**

## Choosing the failure category

Only when the attempt failed. Pick the one that best explains *why*, using the
steps and the screenshots:

- `grounding` — the agent had the right idea but acted on the wrong thing on
  the page: typed into the wrong box, clicked the wrong link or button, read
  the wrong row or column of a table.
- `planning` — the agent chose the wrong thing to do: searched for the wrong
  thing, filtered by the wrong value, stopped before the task was finished,
  gave up, or misunderstood what the task asked for.
- `navigation` — the agent got lost between pages: never reached the page that
  had the answer, went round in circles, or left the right page and did not
  come back.
- `site_change` — the website itself was the problem: the page is missing or
  moved, shows an error, blocked the agent, or no longer contains what the task
  expects. Use this only when the website, not the agent, made the task
  impossible as written.

If several apply, choose the earliest mistake: the one that, had it not
happened, would most likely have let the agent succeed.

For a pass, the category is `none`.

## Answer with

A JSON object only, with no text around it:

- `passed` — `true` or `false`.
- `justification` — one sentence. Say what on the page or in the answer
  decided it. For a fail, say what you expected to see and what was there
  instead.
- `category` — `none` for a pass; otherwise one of `grounding`, `planning`,
  `navigation`, `site_change`.
