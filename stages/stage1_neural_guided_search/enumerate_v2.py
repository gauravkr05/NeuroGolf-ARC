"""Guided enumerator v2.
Improvements over v1:
- arg filtering: gcc src from INPUT colors, dst from OUTPUT colors;
  gc colors from OUTPUT colors (halves the branching)
- score-ordered DFS: steps sorted by model probability, so high-probability
  programs are verified first within each (k, depth) tier
- prefix memoization: shared prefixes executed once (DFS carries grids),
  failed prefixes prune their whole subtree
Metric unchanged: programs checked before a verified solution.
"""
import json, random, time
import torch, torch.nn as nn
from vocab import VOCAB, NAMES, apply_step

KS = [(3, 4), (5, 3), (8, 3), (14, 2), (62, 2)]
BUDGET = 8000
PRIMS = json.load(open("prims.json")); P = len(PRIMS); NP = 3

def enc_pair(pr):
    t = torch.zeros(22, 12, 12)
    for off, mch, grid in ((0, 20, pr["input"]), (10, 21, pr["output"])):
        for r, row in enumerate(grid[:12]):
            for c, v in enumerate(row[:12]):
                t[off + v, r, c] = 1.0; t[mch, r, c] = 1.0
    return t

def enc_task(pairs):
    x = torch.zeros(NP, 22, 12, 12); n = min(len(pairs), NP)
    for i in range(n):
        x[i] = enc_pair(pairs[i])
    return x, n

class Net(nn.Module):
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
        h = self.cnn(x.view(B * NP, 22, 12, 12)).view(B, NP, 96)
        return self.head(h.sum(1) / n.clamp(min=1).unsqueeze(1))

ck = torch.load("stage1_cnn.pt", map_location="cpu")
model = Net(); model.load_state_dict(ck["model"]); model.eval()

def steps_for(prim, pin, pout):
    spec = VOCAB[prim][1]
    if spec == "g":   return [(prim, ())]
    if spec == "gk":  return [(prim, (k,)) for k in (2, 3)]
    if spec == "gc":  return [(prim, (c,)) for c in pout]
    return [(prim, (a, b)) for a in pin for b in pout if a != b]

class Solver:
    def __init__(self, pairs):
        self.ins = [tuple(tuple(r) for r in p["input"]) for p in pairs]
        self.outs = [tuple(tuple(r) for r in p["output"]) for p in pairs]
        self.pin = sorted({v for g in self.ins for r in g for v in r})
        self.pout = sorted({v for g in self.outs for r in g for v in r})
        self.execs = 0

    def dfs(self, grids, steps, depth):
        for step in steps:
            self.execs += 1
            if self.execs > BUDGET:
                return "budget"
            try:
                nxt = [apply_step(step[0], step[1], g) for g in grids]
            except Exception:
                continue
            if nxt == self.outs:
                return [step]
            if depth > 1:
                r = self.dfs(nxt, steps, depth - 1)
                if r == "budget":
                    return r
                if r is not None:
                    return [step] + r
        return None

    def search(self, ranking):
        for k, maxdepth in KS:
            allowed = ranking[:k]
            steps = [s for p in allowed
                     for s in steps_for(p, self.pin, self.pout)]
            for depth in range(1, maxdepth + 1):
                r = self.dfs(list(self.ins), steps, depth)
                if r == "budget":
                    return self.execs, None
                if r is not None:
                    return self.execs, r
        return self.execs, None

def model_ranking(pairs):
    x, n = enc_task(pairs)
    with torch.no_grad():
        lg = model(x.unsqueeze(0), torch.tensor([float(n)]))
    return [PRIMS[i] for i in lg[0].argsort(descending=True).tolist()]

def evaluate(tasks, label):
    rng = random.Random(0)
    res = {}
    for mode in ("guided", "uniform"):
        sol, ex = 0, []
        for pairs in tasks:
            rank = model_ranking(pairs) if mode == "guided" else \
                   rng.sample(NAMES, len(NAMES))
            e, p = Solver(pairs).search(rank)
            if p is not None:
                sol += 1; ex.append(e)
        med = sorted(ex)[len(ex)//2] if ex else "-"
        res[mode] = (sol, med)
    g, u = res["guided"], res["uniform"]
    print(f"{label:>10}: guided {g[0]}/{len(tasks)} (median {g[1]})  |  "
          f"uniform {u[0]}/{len(tasks)} (median {u[1]})", flush=True)

bench = {k: v for k, v in json.load(open("bench.json")).items() if v}
real = []
for num in sorted(bench):
    t = json.load(open(f"/home/claude/compdata/task{num}.json"))
    pairs = [p for p in t["train"]
             if max(len(p["input"]), len(p["input"][0]),
                    len(p["output"]), len(p["output"][0])) <= 12][:3]
    real.append(pairs)
t0 = time.time()
evaluate(real, "BENCH-50")
synth = [json.loads(l)["pairs"] for l in open("synth_val.jsonl")][:150]
evaluate(synth, "SYNTH-150")
print(f"({time.time()-t0:.0f}s)")
