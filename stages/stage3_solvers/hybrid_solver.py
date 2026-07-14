"""Hybrid solver: generator samples first (fast, unbounded depth),
enumerator as fallback (exact, depth-limited).
Reports: generator-only solves, enumerator-only solves, both solves,
and tasks neither can solve. Also computes total median execs.
"""
import json, random, time
import torch
from vocab import VOCAB, NAMES, apply_step
from train_stage2 import Gen, enc_task, TOKS, TIX
from enumerate_v2 import Solver, model_ranking

# ---- load models ----
import torch.nn as nn
PRIMS = json.load(open("prims.json")); P = len(PRIMS)

class Stage1CNN(nn.Module):
    NP = 3
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(22, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(96, 128), nn.ReLU(),
                                  nn.Linear(128, P))
    def forward(self, x, n):
        B = x.shape[0]
        h = self.cnn(x.view(B * self.NP, 22, 12, 12)).view(B, self.NP, 96)
        return self.head(h.sum(1) / n.clamp(min=1).unsqueeze(1))

ck1 = torch.load("stage1_cnn.pt", map_location="cpu")
m1 = Stage1CNN(); m1.load_state_dict(ck1["model"]); m1.eval()

ck2 = torch.load("stage2.pt", map_location="cpu")
m2 = Gen(); m2.load_state_dict(ck2["model"]); m2.eval()
print(f"stage1 ep{ck1['epoch']}  stage2 ep{ck2['epoch']}")

SPEC_ARGC = {"g": 0, "gc": 1, "gcc": 2, "gk": 1}

def decode(row):
    toks = [TOKS[t] for t in row.tolist()][1:]
    prog, i = [], 0
    while i < len(toks) and toks[i] not in ("<eos>", "<pad>"):
        op = toks[i]
        if op not in VOCAB:
            return None
        argc = SPEC_ARGC[VOCAB[op][1]]
        args = toks[i + 1:i + 1 + argc]
        if len(args) < argc or not all(a.isdigit() for a in args):
            return None
        prog.append((op, tuple(int(a) for a in args)))
        i += 1 + argc
    return prog if prog else None

def solves(prog, ins, outs):
    try:
        for a, b in zip(ins, outs):
            g = a
            for name, args in prog:
                g = apply_step(name, args, g)
            if g != b: return False
        return True
    except Exception:
        return False

def hybrid_solve(pairs, n_gen=128, budget_enum=4000):
    ins = [tuple(tuple(r) for r in p["input"]) for p in pairs]
    outs = [tuple(tuple(r) for r in p["output"]) for p in pairs]
    x, n = enc_task(pairs)
    # Phase 1: generator
    rows = m2.sample(x.unsqueeze(0), torch.tensor([float(n)]),
                     n_gen, temp=1.0)
    gen_execs = 0
    for row in rows:
        prog = decode(row)
        if prog is None: continue
        gen_execs += 1
        if solves(prog, ins, outs):
            return "gen", gen_execs + 0, prog
    # Phase 2: enumerator with stage-1 ranking
    with torch.no_grad():
        lg = m1(x.unsqueeze(0), torch.tensor([float(n)]))
    rank = [PRIMS[i] for i in lg[0].argsort(descending=True).tolist()]
    e, prog = Solver(pairs).search(rank)
    if prog is not None:
        return "enum", gen_execs + e, prog
    return None, gen_execs + e, None

# ---- evaluate ----
bench = {k: v for k, v in json.load(open("bench.json")).items() if v}
tasks = []
for num in sorted(bench):
    t = json.load(open(f"./compdata/task{num}.json"))
    pairs = [p for p in t["train"]
             if max(len(p["input"]), len(p["input"][0]),
                    len(p["output"]), len(p["output"][0])) <= 12][:3]
    if len(pairs) >= 2:
        tasks.append((num, pairs))

t0 = time.time()
gen_only = enum_only = both_fail = total_sol = 0
gen_exs = []; enum_exs = []

# also track what enumerator does alone for comparison
rng = random.Random(0)
enum_solo = 0

for num, pairs in tasks:
    x, n = enc_task(pairs)
    with torch.no_grad():
        lg = m1(x.unsqueeze(0), torch.tensor([float(n)]))
    rank = [PRIMS[i] for i in lg[0].argsort(descending=True).tolist()]
    e_solo, p_solo = Solver(pairs).search(rank)
    if p_solo: enum_solo += 1

    mode, execs, prog = hybrid_solve(pairs)
    if mode == "gen":
        gen_only += 1; total_sol += 1; gen_exs.append(execs)
    elif mode == "enum":
        enum_only += 1; total_sol += 1; enum_exs.append(execs)
    else:
        both_fail += 1

N = len(tasks)
med = lambda v: sorted(v)[len(v)//2] if v else "-"
print(f"\n=== Hybrid results on {N} tasks ===")
print(f"  Generator solved first : {gen_only:3d}  (median {med(gen_exs)} samples)")
print(f"  Enumerator fallback    : {enum_only:3d}  (median {med(enum_exs)} programs)")
print(f"  Neither solved         : {both_fail:3d}")
print(f"  Total hybrid coverage  : {total_sol}/{N} = {100*total_sol/N:.0f}%")
print(f"  Enumerator alone       : {enum_solo}/{N}")
print(f"  Generator alone        : 11/{N}  (from cell 9)")
print(f"  Hybrid gain over best  : +{total_sol - max(enum_solo, 11)} tasks")
print(f"  ({time.time()-t0:.0f}s)")
