"""Sample-and-verify evaluation of the stage-2 generator.
For each task: sample up to 256 programs, verify in sample order,
execs = programs tried until first verified. Same metric as enumerator.
"""
import json, torch
from vocab import VOCAB, NAMES, apply_step
from train_stage2 import Gen, enc_task, TOKS, TIX

SPEC_ARGC = {"g": 0, "gc": 1, "gcc": 2, "gk": 1}

ck = torch.load("stage2.pt", map_location="cpu")
model = Gen(); model.load_state_dict(ck["model"]); model.eval()
print(f"stage2 checkpoint epoch {ck['epoch']}")

def decode(row):
    toks = [TOKS[t] for t in row.tolist()][1:]          # drop <bos>
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
            if g != b:
                return False
        return True
    except Exception:
        return False

def eval_tasks(tasks, label, n_samples=256):
    sol, ex = 0, []
    for pairs in tasks:
        ins = [tuple(tuple(r) for r in p["input"]) for p in pairs]
        outs = [tuple(tuple(r) for r in p["output"]) for p in pairs]
        x, n = enc_task(pairs)
        rows = model.sample(x.unsqueeze(0), torch.tensor([float(n)]),
                            n_samples, temp=1.0)
        seen, execs, hit = set(), 0, None
        for row in rows:
            prog = decode(row)
            if prog is None:
                continue
            key = str(prog)
            if key in seen:
                continue
            seen.add(key); execs += 1
            if solves(prog, ins, outs):
                hit = execs; break
        if hit:
            sol += 1; ex.append(hit)
    med = sorted(ex)[len(ex)//2] if ex else "-"
    print(f"{label:>10}: generator {sol}/{len(tasks)} solved "
          f"(median execs {med})", flush=True)

bench = {k: v for k, v in json.load(open("bench.json")).items() if v}
real = []
for num in sorted(bench):
    t = json.load(open(f"/home/claude/compdata/task{num}.json"))
    real.append([p for p in t["train"]
                 if max(len(p["input"]), len(p["input"][0]),
                        len(p["output"]), len(p["output"][0])) <= 12][:3])
eval_tasks(real, "BENCH-50")
synth = [json.loads(l)["pairs"] for l in open("synth_val.jsonl")][:100]
eval_tasks(synth, "SYNTH-100")
