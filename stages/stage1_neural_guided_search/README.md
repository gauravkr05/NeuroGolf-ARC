# Stage 1 — Neural-guided enumerative search

The DSL-golf core. Probe the arc-dsl primitives into a 62-op search vocabulary
(`vocab.py`), generate synthetic `(input, output, program)` data (`gen_synth4.py`),
train a CNN to predict which primitives a task needs (`train_stage1_cnn.py` → `stage1_cnn.pt`),
then run a depth-limited enumerator (`enumerate_v2.py`) restricted to the CNN's **top-k** ops
(the "top 5 / top 8" cutoff). Labels for the CNN come from the executable `bench.json`,
which `bench_build.py` (re)builds by finding a solving op-sequence per task.

Run order:

```bash
python bench_build.py       # build/refresh bench.json (labels)
python gen_synth4.py        # generate synth_train.jsonl / synth_val.jsonl
python train_stage1_cnn.py  # train stage1_cnn.pt
python enumerate_v2.py      # guided enumerative search
```

Requires arc-dsl + ARC tasks — see the top-level `SETUP.md`. This is the version Stage 2
and Stage 3 build on (`vocab.py`, `prims.json`, `bench.json`, `stage1_cnn.pt`).
