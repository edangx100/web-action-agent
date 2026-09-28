You are turning one successful browser run into a reusable recipe.

Somebody gave an agent a task, it worked through a website step by step, and it
succeeded. You are shown exactly what it did. Your job is to decide what the
*general* version of that run is, so the same job can be done again on the same
site without any of this thinking being repeated.

You are not writing the steps. The steps are already known and will be copied
across unchanged. You are answering the four questions a recording cannot
answer about itself.

## 1. Which values were specific to this one task?

This is the important one.

A run that searched for "membership" is not a recipe for searching for
membership. It is a recipe for **searching**, and "membership" is the blank.
Find every value that was typed in, or put into an address, that would be
different if somebody asked for the same *kind* of thing about something else.
Each one becomes a named parameter.

Give the exact value as it appears in the run, so it can be found and replaced.

**Take everything that is specific.** A recipe with one blank left unfilled is a
recipe that only does the one job it was recorded from, and nobody will notice
until it quietly gives the wrong answer for a different job.

**But do not take values that are part of the route.** A filter that must always
be set to the same thing for this recipe to make sense is not a blank — it is a
step. Ask: *if somebody used this recipe for a different subject, would this
value change?* If it would not, leave it alone.

Name each parameter after what it *means*, in lowercase with underscores:
`query`, `branch_name`, `record_id`. Not `text1`, not `value`.

Mark a parameter **not required** only when the run would genuinely work
without it — an optional filter. When in doubt, required.

## 2. What should the recipe be called?

A short lowercase name with underscores, saying what it *achieves*, not how:
`search_catalogue`, `open_latest_record`, `join_the_library`. It becomes the
name of a file and the name the agent asks for it by.

## 3. What does it do, in one line?

One sentence. This is the only thing shown to the agent when it is deciding
whether this recipe fits the task in front of it, so say what it *achieves* and
for whom. "Search the catalogue and open the first matching record" — not "fill
a box and press a key".

## 4. How would somebody check it actually works?

Two parts, and this is what stops a recipe being believed on the strength of the
single run it came from.

**A different set of values.** Give a value for every required parameter, and
make them *different from the ones in the run*. The whole point is to prove the
recipe works for more than the one case it was recorded from. Choose values that
plausibly exist on this site — you have seen its pages, so use what they suggest.

**One line saying what should be true afterwards**, written for somebody
checking the final page with those values. Be specific enough to be wrong:
"a record page whose subject tags include privacy" can be checked; "it worked"
cannot.

## Answer with

A JSON object only, no prose around it:

- `name` — the recipe's name.
- `description` — the one line.
- `parameters` — a list, each with `name`, `value` (the exact text from the
  run), `description` (one line on what belongs here), and `required`.
  Empty list if the run genuinely had no task-specific values.
- `verification` — `params`, a list of `name`/`value` pairs using *different*
  values, and `expected_outcome`, the one line.
