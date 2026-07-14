# Stage 3 — Solvers (TTT + hybrid)

* `ttt_solver.py` — **test-time training**. Per unsolved task: sample N programs; execute the
  syntactically-valid failures on this task's inputs — each failure is a perfect label for the
  *hindsight* task (inputs → program(inputs)); fine-tune a copy of the generator on those and
  resample. Arm A (TTT, adapts between rounds) vs Arm B (frozen base, same total budget):
  TTT is real iff A solves tasks B does not.
* `hybrid_solver.py` — run the Stage 2 generator first (fast, unbounded depth); fall back to
  the Stage 1 enumerator (exact, depth-limited). Reports generator-only / enumerator-only /
  both / neither, plus total median execs.

Depends on `stage2.pt` from Stage 2 (git-ignored; train it first). Bundled deps copied here:
`train_stage2.py`, `vocab.py`, `enumerate_v2.py`, `prims.json`, `bench.json`, `stage1_cnn.pt`.
