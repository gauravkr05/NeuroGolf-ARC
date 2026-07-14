# Setup — external dependencies & hardcoded paths

This project was developed inside a container where two external resources lived under
`/home/claude`. They are **not** part of this repo. Do one of: (a) reproduce those paths,
or (b) edit the paths in the code. Both are quick.

## 1. arc-dsl (the DSL — `dsl.py`)

Referenced in `vocab.py`, `gen_synth2.py`, `gen_synth3.py`, `gen_synth4.py`,
`enumerate_guided.py` via:

```python
sys.path.insert(0, "/home/claude/arc-dsl")
import dsl
```

Fix:

```bash
git clone https://github.com/michaelhodel/arc-dsl /home/claude/arc-dsl
# OR clone anywhere and change the sys.path.insert(...) line to that location,
# OR: pip install -e . inside your arc-dsl checkout and delete the sys.path line.
```

## 2. ARC task data (`compdata/`)

Read as `/home/claude/compdata/task{NUM}.json` in the `gen_synth*`, `train_stage1_cnn`,
`enumerate*`, `eval_stage2`, `hybrid_solver`, and `ttt_solver` scripts. `arcid_to_num.json`
maps ARC task ids to the `NUM` filename index; `bench.json` is keyed by the same numbers.

Fix: place your ARC task JSONs at `/home/claude/compdata/` (or edit the path). Each file is
the standard ARC format: `{"train": [{"input": [...], "output": [...]}, ...], "test": [...]}`.

## 3. Files a fresh clone does NOT contain (generated at runtime)

| File               | Produced by                | Needed by                          |
|--------------------|----------------------------|------------------------------------|
| `synth_train.jsonl`, `synth_val.jsonl` | `gen_synth*.py` | `train_stage1_cnn.py`, `train_stage2.py`, eval scripts |
| `stage2.pt`        | `train_stage2.py`          | `eval_stage2.py`, `ttt_solver.py`, `hybrid_solver.py` |

Run the generator/trainer for a stage before running that stage's eval/solver.

## Quick grep to find every hardcoded path

```bash
grep -rn "/home/claude" .
```
