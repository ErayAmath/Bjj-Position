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

- **Milestone 1 is complete.** Owner tasks A (`normalize_pose`) and B (`train_step`) are done;
  all 69 tests pass. Pipeline: pose ensemble -> tracking -> features -> MLP -> Viterbi ->
  timeline, entry points `scripts/train_position_model.py`, `scripts/analyze_round.py`,
  `scripts/eval_round.py`.
- Numbers (2026-09-26, results/): classifier 79.2 % validation / 90.4 % test; end to end on a
  held-out 2000-frame round 70.5 % (18 classes), 84.5 % (10 base positions).
- **Caveat to repeat whenever a number is quoted:** the test camera films the same sparring as
  the training cameras, so it measures a new angle, not a new roll. Leave-one-sequence-out is
  impossible here because each class lives in one sequence. Only the owner's own footage can
  answer generalisation.
- Split lives in `bjj.split`: test = one camera per sequence, validation = time blocks inside
  the training cameras with a margin (a second held-out camera cost 9 points).
- Published: https://github.com/ErayAmath/Bjj-Position, dashboard live at
  https://erayamath.github.io/Bjj-Position/ via .github/workflows/pages.yml (Pages source must
  stay on "GitHub Actions"; a workflow token cannot enable Pages itself).
- Open: interview questions for LEARNINGS.md (owner wants them after further changes), then
  the owner's own footage (M3) and M2 (time).
- Private analyses are safe by default: `export_round_to_frontend.py` writes to the git-ignored
  `frontend/data/private/round.json`; the dashboard prefers it and shows a "not published"
  badge. Only `--demo` overwrites the committed public round.
- Commercial use is blocked by licences: ViCoS dataset and the Human-Art-trained detector are
  both CC BY-NC-SA (non-commercial). A product needs own data and permissively licensed models.
- Real-footage findings (2026-09-30): pose selection by confidence tracked bystanders -> use
  `select_rolling_pair` (large + in contact); transition matrix must be rescaled to the analysis
  fps (`rescale_transitions`); runs under ~1 s are noise (`merge_short_runs`). Always render an
  overlay (`scripts/render_overlay.py`) before trusting a timeline.
- Label semantics (measured 2026-10-03, see README): the 1/2 suffix names the TOP athlete for
  mount/side control/turtle, the athlete CONTROLLING in back, and the athlete PLAYING GUARD
  (bottom) for open/closed/half guard. Takedown's suffix is undocumented and unreliable.
  Athlete identity itself is relative: swap augmentation means the model predicts relative to
  input order; the mapping to a person is one bit decided per video.
- OWN FOOTAGE IS THE BOTTLENECK (measured 2026-10-03): dataset model gets 32.5 % base accuracy
  on the owner's phone video because an open guard filmed from the end of the mat has the same
  geometry as the dataset's turtle (hip distance 1.27 vs 2.45 torso lengths). Pose stage is fine
  (95 % both found). Mirroring, rotation and affine "viewpoint" augmentation did NOT help;
  fine-tuning on 60 s of own labels did: 14.8 % -> 44.2 % base on the held-out rest.
  Workflow: label (frontend/label.html) -> scripts/eval_own_labels.py -> scripts/finetune_on_own.py.
- The label vocabulary may exceed the dataset's 18 classes (e.g. leg_entanglement1/2). Fine-tuning
  extends the model head and the transition matrix automatically; keep that path working.
- Owner prefers simple explanations anchored in the pipeline picture (see memory).
