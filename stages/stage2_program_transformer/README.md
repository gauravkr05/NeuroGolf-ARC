# Stage 2 — Program-generating transformer

A CNN task-encoder (same design as Stage 1) feeds a causal transformer decoder that emits a
program as tokens: `[op] [digit args…] … <eos>`. Trained by teacher forcing on
`synth_train.jsonl`; training is resumable and time-boxed and writes `stage2.pt`.

* `train_stage2.py` — train the generator (produces `stage2.pt`).
* `eval_stage2.py`  — sample up to 256 programs per task, verify in sample order; the metric
  (programs tried until first verified) matches the Stage 1 enumerator so the two are comparable.

`vocab.py`, `prims.json`, `bench.json` are copied from Stage 1 `v3_62ops_final` (imported /
read here). Needs `synth_*.jsonl` (regenerate via Stage 1's `gen_synth4.py`).
