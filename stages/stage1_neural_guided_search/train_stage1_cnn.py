"""Stage 1 (CNN encoder): pairs -> [22,10,10] tensors -> shared CNN ->
mean over pairs -> 20 primitive logits. Same labels/eval as before."""
import json, time, random
import torch, torch.nn as nn

torch.manual_seed(0); random.seed(0)
torch.set_num_threads(4)

PRIMS = json.load(open("prims.json"))
P = len(PRIMS); PIDX = {p: i for i, p in enumerate(PRIMS)}
NP = 3   # pairs per task

def enc_pair(pr):
    t = torch.zeros(22, 12, 12)
    for ch_off, mask_ch, grid in ((0, 20, pr["input"]), (10, 21, pr["output"])):
        for r, row in enumerate(grid[:12]):
            for c, v in enumerate(row[:12]):
                t[ch_off + v, r, c] = 1.0
                t[mask_ch, r, c] = 1.0
    return t

def enc_task(pairs):
    x = torch.zeros(NP, 22, 12, 12)
    n = min(len(pairs), NP)
    for i in range(n):
        x[i] = enc_pair(pairs[i])
    return x, n

def load(fn, limit=None):
    import os
    limit = limit or int(os.environ.get("NLIMIT", "10**9"))
    X, N, Y = [], [], []
    for li, line in enumerate(open(fn)):
        if li >= limit:
            break
        ex = json.loads(line)
        x, n = enc_task(ex["pairs"])
        y = torch.zeros(P)
        for p in ex["primitives"]:
            y[PIDX[p]] = 1.0
        X.append(x.to(torch.uint8)); N.append(n); Y.append(y)
    return torch.stack(X), torch.tensor(N, dtype=torch.float), torch.stack(Y)

Xtr, Ntr, Ytr = load("synth_train.jsonl")
Xva, Nva, Yva = load("synth_val.jsonl")

bench = {k: v for k, v in json.load(open("bench.json")).items() if v}
rX, rN, rY, rids = [], [], [], []
for num, prog in bench.items():
    t = json.load(open(f"/home/claude/compdata/task{num}.json"))
    pairs = [p for p in t["train"]
             if max(len(p["input"]), len(p["input"][0]),
                    len(p["output"]), len(p["output"][0])) <= 12][:NP]
    if len(pairs) < 2:
        continue
    x, n = enc_task(pairs)
    y = torch.zeros(P)
    for name, _ in prog:
        y[PIDX[name]] = 1.0
    rX.append(x); rN.append(n); rY.append(y); rids.append(num)
rX, rN, rY = torch.stack(rX), torch.tensor(rN, dtype=torch.float), torch.stack(rY)
print(f"real eval tasks: {len(rids)}", flush=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(22, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(96, 128), nn.ReLU(),
                                  nn.Linear(128, P))
    def forward(self, x, n):                    # x: [B,NP,22,10,10]
        B = x.shape[0]
        h = self.cnn(x.view(B * NP, 22, 12, 12)).view(B, NP, 96)
        h = h.sum(1) / n.clamp(min=1).unsqueeze(1)
        return self.head(h)

def recall_at_k(model, X, N, Y, ks=(3, 5, 8)):
    model.eval(); hits = {k: 0 for k in ks}
    with torch.no_grad():
        for i in range(0, len(X), 256):
            lg = model(X[i:i+256].float(), N[i:i+256])
            for r in range(len(lg)):
                true = set(torch.nonzero(Y[i+r]).flatten().tolist())
                order = lg[r].argsort(descending=True).tolist()
                for k in ks:
                    hits[k] += true <= set(order[:k])
    return {k: hits[k] / len(X) for k in ks}

model = Net()
import os
start = 0
if os.path.exists("stage1_cnn.pt"):
    ck = torch.load("stage1_cnn.pt")
    model.load_state_dict(ck["model"]); start = ck["epoch"]
    print(f"resumed at epoch {start}", flush=True)
print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
lossf = nn.BCEWithLogitsLoss()

t0, ep = time.time(), start
while time.time() - t0 < 100 and ep < start + 30:
    ep += 1
    model.train()
    perm = torch.randperm(len(Xtr)); tot = 0.0
    for i in range(0, len(perm), 64):
        js = perm[i:i+64]
        opt.zero_grad()
        loss = lossf(model(Xtr[js].float(), Ntr[js]), Ytr[js])
        loss.backward(); opt.step()
        tot += loss.item() * len(js)
    rv = recall_at_k(model, Xva, Nva, Yva)
    rr = recall_at_k(model, rX, rN, rY)
    print(f"ep{ep} loss {tot/len(perm):.4f} "
          f"synth r@3/5/8 {rv[3]:.2f}/{rv[5]:.2f}/{rv[8]:.2f} "
          f"REAL r@3/5/8 {rr[3]:.2f}/{rr[5]:.2f}/{rr[8]:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)

torch.save({"model": model.state_dict(), "epoch": ep}, "stage1_cnn.pt")
# show per-task predictions on the real tasks
model.eval()
with torch.no_grad():
    lg = model(rX.float(), rN)
print("\nreal-task top-5 predictions:")
for i, tid in enumerate(rids):
    top = [PRIMS[j] for j in lg[i].argsort(descending=True)[:5].tolist()]
    true = [PRIMS[j] for j in torch.nonzero(rY[i]).flatten().tolist()]
    mark = "OK " if set(true) <= set(top) else "MISS"
    print(f"  {mark} {tid}: true={true} top5={top}")
