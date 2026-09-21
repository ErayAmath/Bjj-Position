# CLAUDE.md

Guidance for Claude working in this repo. Keep the "Current state" section up to date.

## Project

Recognise BJJ positions from pose keypoints (ViCoS BJJ dataset), later derive training
statistics from the owner's own sparring videos. Learning project + portfolio. The owner is a
BJJ white belt learning ML via Karpathy's "Zero to Hero" (currently micrograd) and must be able
to explain every design decision in a job interview.

## Roles

- Claude writes most code: setup, data loading, plots, tests, infrastructure.
- The owner writes the highest-learning-value parts, reviews Claude's code and makes decisions.

## Owner coding tasks

- Max 3 per milestone, each ~20–40 min.
- For each: function signature + docstring, a `TODO(human)` marker, and a test that shows when
  it is done. Explain the concept briefly beforehand **without giving away the solution**.
- "Hinweis" → give a hint. "Lösung" → show and explain the solution. "skip" → Claude writes it.
- After the owner implements: critical senior-engineer review (correctness, edge cases,
  readability). Errors in owner code: short hint first, solution only on request.

## Teaching protocol

- Before each step: what, why, and which alternative is rejected.
- After each step: show and explain the 2–3 most important code locations.
- Before every training run / evaluation: ask the owner what result they expect; compare after.
- Suggest small experiments where instructive (random vs sequence split, with/without
  normalisation, ...).
- Explain PyTorch concepts beyond micrograd on first occurrence.
- Own bugs: fix and briefly explain the cause, unless the bug is instructive — then the owner
  gets first try.
- Owner barely knows git: explain the first commit; later show how to push to GitHub.
- End of each milestone: 5 interview questions on design decisions; owner answers, Claude
  corrects, both go into LEARNINGS.md.

## Repo rules

- Small steps; each ends runnable and with a commit with a meaningful message.
- Code, comments, README, commits in **English**; explanations to the owner in **German**.
- README is the lab notebook: decisions, results, failures.
- No external experiment services; results as CSV/JSON in `results/`.
- Raw data (`data/`) is git-ignored — never commit it.
- Frontend is plain HTML/CSS/JS (owner decided: no TypeScript, no Node toolchain).

## Commands

```bash
.venv/Scripts/python -m pytest                          # tests
.venv/Scripts/python scripts/export_frontend_data.py    # refresh frontend data
.venv/Scripts/python scripts/serve_frontend.py           # no-cache dev server, :8000
```

Environment: Windows 11, Python 3.14 venv in `.venv`, RTX 4070 (torch cu128 wheels exist for
3.14; not installed yet).

## Current state

- Milestone 1, plan proposed (steps 0–8, owner tasks A `normalize_pose`, B `train_step`,
  C `confusion_matrix`/`macro_f1`). Owner has not yet chosen C vs. alternative (video split).
- Done: step 0 (setup); pulled forward from step 1: `bjj.data` loader, `bjj.stats`
  (class counts, segment matrix, `group_segments`); static frontend in `frontend/`.
- Finding: 16 segments = camera views of 6 sequences; each sequence covers only a few classes,
  so a split by sequence drops whole classes. Split strategy is an OPEN OWNER DECISION (step 4).
- Higgsfield was requested for visuals but the account had 0.1 credits (1 image = 1 credit);
  hero uses real skeleton animations instead. A photo can be added later if credits exist.
- Frontend has an upload section (`frontend/js/upload.js`): local preview only; talks to a
  future same-origin backend (`/api/health`, `POST /api/analyses`). No backend exists yet —
  plan in README "Backend plan" (FastAPI on localhost, video never leaves the machine, LLM
  coaching chat sees only aggregated stats, API key server-side). Never fake analysis results.
- Next: step 1 remainder (EDA plots in results/figures), then step 3 = owner task A.
