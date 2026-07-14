"""Synthetic dataset v3 over the 42-op vocabulary.
Key improvements institutionalized from v2's bugs:
- quota balancing: force-first-step generation targets the rarest op
- grid-size decorrelation: 20% of ALL programs draw from the small pool
- replace/switch disambiguation retained
"""
import json, glob, random
from collections import Counter
import vocab
from vocab import VOCAB, NAMES, apply_step, is_grid

random.seed(2)

POOL = []
for p in glob.glob("/home/claude/compdata/task*.json"):
    t = json.load(open(p))
    for split in ("train", "test"):
        for ex in t[split]:
            g = tuple(tuple(r) for r in ex["input"])
            if 2 <= len(g) <= 12 and 2 <= len(g[0]) <= 12:
                POOL.append(g)
POOL_SMALL = [g for g in POOL if len(g) <= 4 and len(g[0]) <= 4]

def colors_of(g):
    return {v for r in g for v in r}

def sample_step(name=None):
    if name is None:
        name = random.choice(NAMES)
    spec = VOCAB[name][1]
    if spec == "g":    args = ()
    elif spec == "gc": args = (random.randrange(10),)
    elif spec == "gcc":
        a = random.randrange(10)
        args = (a, random.choice([c for c in range(10) if c != a]))
    else:              args = (random.choice([2, 2, 3]),)
    return (name, args)

def run(steps, g):
    for name, args in steps:
        g = apply_step(name, args, g)
        if not is_grid(g) or len(g) > 12 or len(g[0]) > 12:
            raise ValueError
    return g

def make(first=None):
    n_steps = random.choice([1, 1, 1, 2, 2, 3])
    steps = [sample_step(first)] + [sample_step() for _ in range(n_steps - 1)]
    need = set()
    for n, a in steps:
        if n in ("replace", "switch", "paint_ring", "paint_orth",
                 "paint_diag", "fill_bbox_delta"):
            need.add(a[0])
        elif n in ("keep_only", "subgrid_color"):
            need.add(a[0])
    pool = POOL_SMALL if random.random() < 0.2 else POOL
    cands = [g for g in pool if need <= colors_of(g)] if need else pool
    if len(cands) < 3:
        return None
    pairs, changed = [], 0
    for g in random.sample(cands, 3):
        try:
            o = run(steps, g)
        except Exception:
            return None
        pairs.append((g, o))
        changed += (o != g)
    if changed < 2:
        return None
    return {"pairs": [{"input": [list(r) for r in a],
                       "output": [list(r) for r in b]} for a, b in pairs],
            "primitives": sorted({n for n, _ in steps}),
            "program": " ".join(f"{n} {' '.join(map(str,a))}".strip()
                                for n, a in steps)}

TARGET, QUOTA = 24000, 300
data, counts, tries = [], Counter(), 0
while len(data) < TARGET and tries < 1500000:
    tries += 1
    # half the attempts target the currently-rarest operation
    first = None
    if tries % 2 == 0:
        first = min(NAMES, key=lambda n: counts[n])
    ex = make(first)
    if ex:
        data.append(ex)
        counts.update(ex["primitives"])

random.shuffle(data)
json.dump(NAMES, open("prims.json", "w"))
with open("synth_train.jsonl", "w") as f:
    for ex in data[:23000]:
        f.write(json.dumps(ex) + "\n")
with open("synth_val.jsonl", "w") as f:
    for ex in data[23000:]:
        f.write(json.dumps(ex) + "\n")

print(f"{len(NAMES)} ops | train {min(len(data),23000)} val "
      f"{max(0,len(data)-23000)} (tries {tries})")
low = [(n, counts[n]) for n in NAMES if counts[n] < QUOTA]
print("ops under quota:", low if low else "none")
print("min/median/max coverage:",
      min(counts.values()),
      sorted(counts.values())[len(counts)//2],
      max(counts.values()))
