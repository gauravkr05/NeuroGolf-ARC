"""Stage 2: program-generating transformer.
CNN task encoder (as stage 1) -> prefix embedding -> causal transformer
decoder emits program tokens: [op] [digit args...] ... <eos>.
Train: teacher forcing on synth_train.jsonl. Resumable, time-boxed.
"""
import json, os, time, random
import torch, torch.nn as nn

torch.manual_seed(0); random.seed(0)
torch.set_num_threads(4)

from vocab import VOCAB, NAMES
TOKS = ["<pad>", "<bos>", "<eos>"] + NAMES + [str(d) for d in range(10)]
TIX = {t: i for i, t in enumerate(TOKS)}
V = len(TOKS); MAXT = 14; NP = 3

def prog_tokens(prog_str):
    toks = prog_str.split()
    return [TIX["<bos>"]] + [TIX[t] for t in toks] + [TIX["<eos>"]]

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

def load(fn, limit=None):
    limit = limit or int(os.environ.get("NLIMIT", "10**9"))
    X, N, T = [], [], []
    for li, line in enumerate(open(fn)):
        if li >= limit:
            break
        ex = json.loads(line)
        seq = prog_tokens(ex["program"])
        if len(seq) > MAXT:
            continue
        x, n = enc_task(ex["pairs"])
        X.append(x.to(torch.uint8)); N.append(n)
        T.append(seq + [0] * (MAXT - len(seq)))
    return (torch.stack(X), torch.tensor(N, dtype=torch.float),
            torch.tensor(T, dtype=torch.long))

class Gen(nn.Module):
    def __init__(self, d=128, layers=3, heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(22, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(96, d)
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(MAXT + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1,
                                           batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, V)

    def encode(self, x, n):
        B = x.shape[0]
        h = self.cnn(x.view(B * NP, 22, 12, 12)).view(B, NP, 96)
        return self.proj(h.sum(1) / n.clamp(min=1).unsqueeze(1))

    def forward(self, x, n, toks):
        B, L = toks.shape
        task = self.encode(x, n).unsqueeze(1)                 # [B,1,d]
        h = torch.cat([task, self.emb(toks)], 1)
        h = h + self.pos.weight[:L + 1]
        mask = torch.triu(torch.full((L + 1, L + 1), float("-inf")), 1
                          ).to(h.device)
        h = self.dec(h, mask=mask)
        return self.out(h[:, :-1])          # predicts toks[t] from prefix

    @torch.no_grad()
    def sample(self, x, n, num, temp=1.0):
        task = self.encode(x, n)                              # [1,d]
        task = task.expand(num, -1).unsqueeze(1)
        toks = torch.full((num, 1), TIX["<bos>"], dtype=torch.long)
        done = torch.zeros(num, dtype=torch.bool)
        for step in range(MAXT - 1):
            h = torch.cat([task, self.emb(toks)], 1)
            h = h + self.pos.weight[:toks.shape[1] + 1]
            L = h.shape[1]
            mask = torch.triu(torch.full((L, L), float("-inf")), 1)
            h = self.dec(h, mask=mask)
            logits = self.out(h[:, -1]) / temp
            logits[:, TIX["<pad>"]] = float("-inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
            nxt[done] = TIX["<pad>"]
            done |= (nxt == TIX["<eos>"])
            toks = torch.cat([toks, nxt.unsqueeze(1)], 1)
            if done.all():
                break
        return toks

if __name__ == "__main__":
    Xtr, Ntr, Ttr = load("synth_train.jsonl")
    print(f"train sequences: {len(Ttr)}  vocab: {V}", flush=True)
    model = Gen()
    start = 0
    if os.path.exists("stage2.pt"):
        ck = torch.load("stage2.pt", map_location="cpu")
        model.load_state_dict(ck["model"]); start = ck["epoch"]
        print(f"resumed at epoch {start}", flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=TIX["<pad>"])
    t0, ep = time.time(), start
    while time.time() - t0 < 100 and ep < start + 30:
        ep += 1
        model.train()
        sub = torch.randperm(len(Ttr))[:4000]
        tot = 0.0
        for i in range(0, len(sub), 48):
            js = sub[i:i + 48]
            opt.zero_grad()
            logits = model(Xtr[js].float(), Ntr[js], Ttr[js][:, :-1])
            loss = lossf(logits.reshape(-1, V), Ttr[js][:, 1:].reshape(-1))
            loss.backward(); opt.step()
            tot += loss.item() * len(js)
        print(f"ep{ep} loss {tot/len(sub):.4f} ({time.time()-t0:.0f}s)",
              flush=True)
    torch.save({"model": model.state_dict(), "epoch": ep}, "stage2.pt")
    print("saved stage2.pt")
