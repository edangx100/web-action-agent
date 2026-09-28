# Committed run folders, for the tests that need a run to have happened

A run folder is what the agent leaves behind after doing a task: what it was
asked, the page it saw at each step, what it chose, what that cost, and the
verdict. Most tests that need one produce it on the spot, by running the agent
against the practice site — which is better, because then the folder and the run
it came from can be compared.

The folders here are the exception. Induction turns *one finished run* into a
reusable skill, and the interesting question about it is what it makes of a
particular route through a particular site. That needs the same run every time,
which means a run committed to the repository rather than one produced afresh.

## `search-and-open/`

A real run, produced by the real agent against the practice site: search the
catalogue for a term, submit the form, and open the matching record.

```
  1  act    fill [7] searchbox "Search terms" with "membership" -> ok
  2  act    press Enter -> ok
  3  act    click [7] link "Membership application" -> ok
  4  done   success: Membership application
```

It is a good induction subject because of what step 3 does. "membership" was
typed, so it plainly becomes a blank. But the link that step clicks is called
"Membership application" — and that is not a value anybody typed, it is *what
the search happened to return*.

So the skill induced from this run reads as though it searches for anything and
opens whatever it finds, while actually looking for that one link by name.
Nothing in induction can see that: the step is a faithful record of what the run
did, and the result is schema-valid. It is the verification block that catches
it, by replaying with a different search term and finding no such link.

That makes this run a useful fixture twice over — once for what induction gets
right, and once for the limit that the next component exists to cover.

### What was changed before committing, and why

Two edits, both because a folder written on one machine has to read the same on
every other one:

- **The address.** The practice site binds a random free port, so every run
  records a different one. Every `http://127.0.0.1:<port>` was rewritten to
  `http://northgate.example`. Nothing else about the pages was touched — the
  elements, the text and the actions are exactly what the browser produced.
- **The screenshots.** The images were left out and their paths set to null, to
  keep a few hundred kilobytes of PNG out of the repository's history for a
  fixture that never looks at them. A complete run folder, images and all, is
  committed separately as the project's worked example.

Everything else is the file the agent wrote.
