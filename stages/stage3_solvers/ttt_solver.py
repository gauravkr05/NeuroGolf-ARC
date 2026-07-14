"""Test-time training (TTT) for the stage-2 generator.

Per unsolved task:
  round 0: sample N programs from base model; verify; if hit -> done
  rounds 1..R: take all syntactically-valid sampled programs, execute them
    on THIS task's inputs -> each failed program P is a perfect label for
    the hindsight task (inputs -> P(inputs)). Fine-tune a copy of the model
    on those pairs, then resample on the real task.

Experiment (the test that TTT works):
  arm A (TTT): R+1 rounds x N samples with adaptation between rounds
  arm B (control): (R+1) x N samples from the frozen base model
  Same total budget. TTT is real iff A solves tasks B does not.
"""
import json, copy, random
import torch, torch.nn as nn
from vocab import VOCAB, apply_step
from train_stage2 import Gen, enc_task, enc_pair, TOKS, TIX, MAXT, V

torch.manual_seed(0); random.seed(0)
SPEC_ARGC = {"g": 0, "gc": 1, "gcc": 2, "gk": 1}
N_PER_ROUND, ROUNDS = 192, 2       # total budget = N * (ROUNDS+1)

ck = torch.load("stage2.pt", map_location="cpu")
base = Gen(); base.load_state_dict(ck["model"]); base.eval()
print(f"stage2 checkpoint epoch {ck['epoch']}")

def decode(row):
    toks = [TOKS[t] for t in row.tolist()][1:]
    prog, i = [], 0
    while i < len(toks) and toks[i] not in ("<eos>", "<pad>"):
        op = toks[i]
        if op not in VOCAB: return None
        k = SPEC_ARGC[VOCAB[op][1]]
        args = toks[i+1:i+1+k]
        if len(args) < k or not all(a.isdigit() for a in args): return None
        prog.append((op, tuple(int(a) for a in args)))
        i += 1 + k
    return prog if prog else None

def run_prog(prog, g):
    for name, args in prog:
        g = apply_step(name, args, g)
    return g

def is_grid12(g):
    return (isinstance(g, tuple) and 1 <= len(g) <= 12 and
            all(isinstance(r, tuple) and len(r) == len(g[0]) and
                all(isinstance(v, int) and 0 <= v <= 9 for v in r)
                for r in g))

def prog_to_seq(prog):
    toks = [TIX["<bos>"]]
    for name, args in prog:
        toks.append(TIX[name])
        toks += [TIX[str(a)] for a in args]
    toks.append(TIX["<eos>"])
    return toks if len(toks) <= MAXT else None

def sample_round(model, pairs, n):
    """Returns (solution|None, list_of_valid_progs)."""
    ins = [tuple(tuple(r) for r in p["input"]) for p in pairs]
    outs = [tuple(tuple(r) for r in p["output"]) for p in pairs]
    x, m = enc_task(pairs)
    rows = model.sample(x.unsqueeze(0), torch.tensor([float(m)]), n, temp=1.0)
    valid, seen = [], set()
    for row in rows:
        prog = decode(row)
        if prog is None or str(prog) in seen: continue
        seen.add(str(prog))
        try:
            res = [run_prog(prog, g) for g in ins]
        except Exception:
            continue
        if res == outs:
            return prog, valid
        valid.append((prog, res))
    return None, valid

def finetune_on_hindsight(model, pairs, valid, steps=60, lr=3e-4):
    ins = [tuple(tuple(r) for r in p["input"]) for p in pairs]
    data = []
    for prog, res in valid:
        if not all(is_grid12(g) for g in res): continue
        if res == list(ins): continue                # identity: useless
        seq = prog_to_seq(prog)
        if seq is None: continue
        hp = [{"input": [list(r) for r in a], "output": [list(r) for r in b]}
              for a, b in zip(ins, res)]
        x, m = enc_task(hp)
        data.append((x, m, seq + [0] * (MAXT - len(seq))))
    if len(data) < 4:
        return model
    ft = copy.deepcopy(model); ft.train()
    opt = torch.optim.AdamW(ft.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=TIX["<pad>"])
    X = torch.stack([d[0] for d in data])
    M = torch.tensor([float(d[1]) for d in data])
    T = torch.tensor([d[2] for d in data], dtype=torch.long)
    for step in range(steps):
        js = torch.randint(0, len(data), (min(32, len(data)),))
        opt.zero_grad()
        logits = ft(X[js], M[js], T[js][:, :-1])
        loss = lossf(logits.reshape(-1, V), T[js][:, 1:].reshape(-1))
        loss.backward(); opt.step()
    ft.eval()
    return ft

def solve_ttt(pairs):
    model = base
    for rnd in range(ROUNDS + 1):
        hit, valid = sample_round(model, pairs, N_PER_ROUND)
        if hit: return hit, rnd
        if rnd < ROUNDS:
            model = finetune_on_hindsight(model, pairs, valid)
    return None, None

def solve_control(pairs):
    for rnd in range(ROUNDS + 1):
        hit, _ = sample_round(base, pairs, N_PER_ROUND)
        if hit: return hit, rnd
    return None, None

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 999
    bench = {k: v for k, v in json.load(open("bench.json")).items() if v}
    tasks = []
    for num in sorted(bench):
        t = json.load(open(f"/home/claude/compdata/task{num}.json"))
        pairs = [p for p in t["train"]
                 if max(len(p["input"]), len(p["input"][0]),
                        len(p["output"]), len(p["output"][0])) <= 12][:3]
        if len(pairs) >= 2:
            tasks.append((num, pairs))
    # focus on tasks the base model fails in round 0 (the interesting set)
    hard = []
    for num, pairs in tasks:
        hit, _ = sample_round(base, pairs, N_PER_ROUND)
        if hit is None:
            hard.append((num, pairs))
        if len(hard) + 0 >= limit and limit < 999:
            break
    print(f"base-unsolved tasks under study: {len(hard)}")
    ttt_w = ctl_w = 0
    for num, pairs in hard[:limit]:
        pt, rt = solve_ttt(pairs)
        pc, rc = solve_control(pairs)
        ttt_w += pt is not None; ctl_w += pc is not None
        tag = ("TTT!" if pt and not pc else
               "ctl " if pc and not pt else
               "both" if pt else "none")
        print(f"  task{num}: {tag}"
              + (f"  (ttt round {rt}: {[s[0] for s in pt]})" if pt else ""),
              flush=True)
    print(f"\nTTT solved {ttt_w}, matched-budget control solved {ctl_w} "
          f"(of {min(limit, len(hard))} base-unsolved tasks)")
    print("TTT is working iff TTT > control here.")
