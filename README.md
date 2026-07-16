# NeuroGolf — ARC Hackathon

A three stage system that learns to solve ARC style grid puzzles by writing small programs. A CNN guesses which operations a task needs, a search engine assembles them into programs, a transformer learns to write whole programs directly, and a test time training experiment checks whether the model can adapt to tasks it fails on.

## The idea

An ARC task shows you a few input/output grid pairs and questions: what's the transformation? Instead of predicting the output grid pixel by pixel, everything here works in **program space**: solutions are short programs over a 62 operation DSL like mirrors, rotations, color swaps, grid concatenations, neighbor painting, and so on. A program is correct if executing it on every demo input reproduces every demo output, so correctness is checked by running code, not by a loss function.

That also solves the "no labels" problem. There's no dataset of (task → program) pairs in the wild, so one gets generated: sample a random valid program, run it on real grids, and the (pairs, program) result is a perfectly labeled training example. ~23k of these, quota-balanced so no operation is rare enough to be unlearnable.

## The three stages

**1. Primitive predictor + guided search** (`stage1_neural_guided_search/`)
A small CNN (~92k params) reads the demo pairs (one-hot encoded, 12×12 canvas) and outputs a probability for each of the 62 operations — "this task smells like `vmirror` and `hconcat`". A DFS enumerator then searches programs using only the top ranked ops first, widening if nothing verifies. DeepCoder style: the network doesn't solve the task, it shrinks the haystack.

* Multi label BCE, Adam `1e-3`, 60 epochs, batch 64, uint8 cached tensors
* Search: tiered depth (deep on top-3 ops, shallow as the beam widens), argument filtering, prefix memoization, 8k program budget

**2. Program transformer** (`stage2_program_transformer/`)
The same CNN encodes the task into a prefix token; a causal transformer decoder (~700k params) then writes the program token by token (`vconcat_hmirror_b <eos>`). Evaluated by sample and verify: draw 256 programs, execute each, first full match wins.

* Cross entropy with teacher forcing, Adam `3e-4`, 150 epochs, batch 48
* Loss lands around 1.6 - part of it is irreducible (argument digits in synthetic programs are partly random)

**3. Solvers** (`stage3_solvers/`)
`hybrid_solver.py` runs the generator first (cheap, unbounded program depth) and falls back to guided search. `ttt_solver.py` is the test-time training experiment: on tasks the generator fails, every failed but valid sample is relabeled as a correct program *for the task it actually computes* (hindsight relabeling), the model is fine-tuned per task on those, and resampled, against a control that gets the same total sample budget with no adaptation.

## Benchmark

39 real ARC tasks, built by **search-labeling**: run the full vocabulary against the ARC training set, keep every task where a verified program exists, and use that program's operations as the labels. Originally 50, pruned to 39 after re verifying each program on hundreds of generated examples — 11 turned out to be lookalike rules that fit the demo pairs but not the task. About half the surviving tasks need 2 operations.

## Results

| | solved | median programs tried |
|---|---|---|
| Guided search | **31/39** | **2** |
| Unguided search (control) | 17/39 | 4,445 |
| Transformer alone | 20/39 | 5 |
| Hybrid | 32/39 | — |
| TTT vs matched-budget control | 4 vs 4 | — |

The headline is the first two rows: with the CNN ranking operations, half the solved tasks fall on the *second* program tried, roughly 2,000× fewer executions than blind search — and the gap grows with vocabulary size, since blind search dies combinatorially while guided search barely notices. The transformer is the opposite personality: low coverage, but when its distribution contains the right program it finds it almost instantly; its failures are usually one wrong argument digit. TTT genuinely rescued 2 tasks the base model couldn't sample (correct ops, wrong combination → adaptation sharpened it), but resampling luck rescued 2 others, so at this scale the adaptation effect is about the size of the noise. Honest verdict: search+guidance is the workhorse, the generator is a promising sketch that needs cross-attention and more compute, and TTT needs augmentation and a stronger base model to shine.

Stage 1 shown here is the final of several iterations (20 → 42 → 62 ops); earlier versions mostly exist as lessons about data bugs, starved classes, label ambiguity, and one self inflicted spurious correlation.

## Running it

Everything is plain Python scripts, happiest on a GPU (Kaggle T4 works).

1. Get the ARC task JSONs and [michaelhodel/arc-dsl](https://github.com/michaelhodel/arc-dsl) (the DSL the ops are built from); point the paths at them.
2. Stage 1: `python gen_synth4.py` → `python train_stage1_cnn.py` → `python enumerate_v2.py`
3. Stage 2: `python train_stage2.py` → `python eval_stage2.py`
4. Stage 3: `python hybrid_solver.py` → `python ttt_solver.py`

Training saves checkpoints (`stage1_cnn.pt`, `stage2.pt`); evaluation scripts print solved counts, medians, and per-task tables.

## Requirements

`torch`, `numpy`. That's it — the DSL and verifier are pure Python.

## Files

```
stages/
├── stage1_neural_guided_search/   62-op DSL probe, synth gen, CNN guide, enumerator
│   ├── vocab.py, gen_synth4.py, bench_build.py, enumerate_v2.py
│   ├── train_stage1_cnn.py, stage1_cnn.pt
│   └── bench.json, prims.json, arcid_to_num.json, task_primitives.json
├── stage2_program_transformer/
│   ├── train_stage2.py      CNN encoder -> causal transformer decoder (teacher forcing)
│   ├── eval_stage2.py       sample-and-verify evaluation
│   └── vocab.py, prims.json, bench.json
└── stage3_solvers/
    ├── ttt_solver.py        test-time training experiment (TTT arm vs frozen-base control)
    ├── hybrid_solver.py     generator-first, enumerator-fallback
    └── train_stage2.py, vocab.py, enumerate_v2.py, prims.json, bench.json, stage1_cnn.pt
```
