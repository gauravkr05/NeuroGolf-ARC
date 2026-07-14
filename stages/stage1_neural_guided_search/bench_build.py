"""Search-label real tasks: for every task whose train pairs fit 12x12,
enumerate programs (depth<=2, full 62-op vocab, filtered args) and record
the shortest verified program. Solved tasks become the benchmark; their
ops become the labels. Resumable via bench.json."""
import json, glob, os, time, itertools
from vocab import VOCAB, NAMES, apply_step

BENCH = "bench.json"
found = json.load(open(BENCH)) if os.path.exists(BENCH) else {}

def steps_for(pal_in, pal_out):
    steps = []
    for name in NAMES:
        spec = VOCAB[name][1]
        if spec == "g":    steps += [(name, ())]
        elif spec == "gk": steps += [(name, (k,)) for k in (2, 3)]
        elif spec == "gc": steps += [(name, (c,)) for c in pal_out]
        else:              steps += [(name, (a, b)) for a in pal_in
                                     for b in pal_out if a != b]
    return steps

def solves(prog, ins, outs):
    try:
        for i, o in zip(ins, outs):
            g = i
            for name, args in prog:
                g = apply_step(name, args, g)
            if g != o:
                return False
        return True
    except Exception:
        return False

t0 = time.time()
tasks = sorted(glob.glob("/home/claude/compdata/task*.json"))
done = new = 0
for p in tasks:
    tid = os.path.basename(p)[4:7]
    if tid in found:
        done += 1
        continue
    if time.time() - t0 > 130:
        break
    t = json.load(open(p))
    pairs = [x for x in t["train"]
             if max(len(x["input"]), len(x["input"][0]),
                    len(x["output"]), len(x["output"][0])) <= 12][:3]
    if len(pairs) < 2:
        found[tid] = None; continue
    ins = [tuple(tuple(r) for r in x["input"]) for x in pairs]
    outs = [tuple(tuple(r) for r in x["output"]) for x in pairs]
    pin = sorted({v for g in ins for r in g for v in r})
    pout = sorted({v for g in outs for r in g for v in r})
    steps = steps_for(pin, pout)
    sol = None
    for depth in (1, 2):
        for prog in itertools.product(steps, repeat=depth):
            if solves(prog, ins, outs):
                sol = prog; break
        if sol: break
    found[tid] = ([[n, list(a)] for n, a in sol] if sol else None)
    done += 1; new += (sol is not None)
json.dump(found, open(BENCH, "w"))
solved = {k: v for k, v in found.items() if v}
print(f"scanned {done}/{len(tasks)} tasks, solved so far: {len(solved)}")
