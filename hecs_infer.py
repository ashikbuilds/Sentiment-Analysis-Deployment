"""
HeCS-GNN — real-time inference for VS Code
==========================================
Loads a trained HeCS-GNN and classifies code-mixed (Bangla-English) text you type.
For each sentence it also reports how much each graph RELATION contributed, so you
can see live whether the code-switch relations actually fire on YOUR input.

SETUP
-----
1) pip install torch torch-geometric emoji
2) Put the exported 'deploy/' folder (from the training notebook's [EXPORT] cell)
   in the same directory as this file. It must contain:
       hecs_gnn_weights.pt, stoi.pkl, pmi.pkl, meta.json
3) Run:  python hecs_infer.py            -> interactive prompt
   or:   python hecs_infer.py "your text here"   -> single prediction

If you have no GPU it runs fine on CPU (single sentences are instant).
"""

import os, re, json, math, pickle, sys
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax as geo_softmax

DEPLOY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------------
def _load_artifacts():
    need = ["hecs_gnn_weights.pt", "stoi.pkl", "pmi.pkl", "meta.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(DEPLOY_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {DEPLOY_DIR}. "
            f"Run the [EXPORT] cell in your training notebook and copy the 'deploy' folder here.")
    with open(os.path.join(DEPLOY_DIR, "stoi.pkl"), "rb") as f: stoi = pickle.load(f)
    with open(os.path.join(DEPLOY_DIR, "pmi.pkl"), "rb") as f: pmi = pickle.load(f)
    with open(os.path.join(DEPLOY_DIR, "meta.json")) as f: meta = json.load(f)
    cfg = meta["config"]
    if isinstance(cfg.get("char_ngram"), list): cfg["char_ngram"] = tuple(cfg["char_ngram"])
    return stoi, pmi, meta, cfg

STOI, PMI, META, CFG = _load_artifacts()
VOCAB = META["vocab_size"]; CLASSES = META["classes"]; NUM_CLASSES = META["num_classes"]

# ----------------------------------------------------------------------------
# Preprocessing — MUST match training exactly
# ----------------------------------------------------------------------------
import emoji
BN_LO, BN_HI = "\u0980", "\u09FF"
REL_SEQ, REL_PMI, REL_BN2EN, REL_EN2BN = 0, 1, 2, 3
REL_NAMES = ["sequential", "pmi-semantic", "switch Bn->En", "switch En->Bn"]

def clean_text(t):
    t = str(t); t = emoji.demojize(t, delimiters=(" ", " "))
    t = re.sub(r"http\S+|www\S+", " ", t); t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(rf"[^\w\s{BN_LO}-{BN_HI}]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()

def tokenize(t): return clean_text(t).split()[:CFG["max_len"]]
def word_lang(w): return 0 if any(BN_LO <= c <= BN_HI for c in w) else 1

def _h(s):
    h = 1469598103934665603
    for ch in s: h ^= ord(ch); h = (h*1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
def char_ngram_ids(w, lo, hi, buckets):
    ids = []; w = f"<{w}>"
    for n in range(lo, hi+1):
        for i in range(len(w)-n+1): ids.append(_h(w[i:i+n]) % buckets)
    return ids[:12] if ids else [0]

def sentence_to_graph(text):
    words = tokenize(text)
    if len(words) < 2: words = words + ["<unk>"]*(2-len(words))
    L = len(words)
    wid = torch.tensor([STOI.get(w, 1) for w in words], dtype=torch.long)
    langs = torch.tensor([word_lang(w) for w in words], dtype=torch.long)
    switches = torch.zeros(L, dtype=torch.long)
    for i in range(1, L):
        if langs[i] != langs[i-1]: switches[i] = 1
    cng = torch.zeros(L, 12, dtype=torch.long)
    for i, w in enumerate(words):
        ids = char_ngram_ids(w, *CFG["char_ngram"], CFG["char_buckets"])
        cng[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    src, dst, etype, eweight = [], [], [], []
    for i in range(L):
        for j in range(max(0,i-CFG["seq_window"]), min(L,i+CFG["seq_window"]+1)):
            if i != j: src.append(i); dst.append(j); etype.append(REL_SEQ); eweight.append(1.0)
    wl = [w if w in STOI else "<unk>" for w in words]
    pos = defaultdict(list)
    for i,w in enumerate(wl): pos[w].append(i)
    for i,w in enumerate(wl):
        for (nbr, score) in PMI.get(w, []):
            for j in pos.get(nbr, []):
                if j != i: src.append(i); dst.append(j); etype.append(REL_PMI); eweight.append(float(score))
    for i in range(1, L):
        if switches[i] == 1:
            r = REL_EN2BN if langs[i]==0 else REL_BN2EN
            lo = max(0, i-CFG["switch_window"]); hi = min(L-1, i+CFG["switch_window"])
            for j in range(lo, hi+1):
                if j != i: src.append(i); dst.append(j); etype.append(r); eweight.append(1.0)
    if len(src) == 0:
        for i in range(L-1):
            src += [i,i+1]; dst += [i+1,i]; etype += [REL_SEQ,REL_SEQ]; eweight += [1.0,1.0]
    edge_index = torch.tensor([src,dst], dtype=torch.long)
    edge_type  = torch.tensor(etype, dtype=torch.long)
    edge_weight= torch.tensor(eweight, dtype=torch.float32)
    pm = (edge_type == REL_PMI)
    if pm.any():
        pw = edge_weight[pm]; pw = (pw-pw.min())/(pw.max()-pw.min()+1e-6)
        edge_weight = edge_weight.clone(); edge_weight[pm] = pw.clamp(0.05, 1.0)
    d = Data(edge_index=edge_index, num_nodes=L)
    d.wid=wid; d.cng=cng; d.langs=langs; d.switches=switches
    d.edge_type=edge_type; d.edge_weight=edge_weight
    return d, words, langs

# ----------------------------------------------------------------------------
# Model — MUST match training architecture exactly
# ----------------------------------------------------------------------------
class NodeEmbedder(nn.Module):
    def __init__(self, cfg):
        super().__init__(); D=cfg["embed_dim"]
        self.word=nn.Embedding(VOCAB, D, padding_idx=0)
        self.cng =nn.Embedding(cfg["char_buckets"], D, padding_idx=0)
        self.lang=nn.Embedding(2,16); self.switch=nn.Embedding(2,16)
        self.proj=nn.Linear(D*2+32, cfg["gnn_hidden"]); self.norm=nn.LayerNorm(cfg["gnn_hidden"])
        self.in_drop=nn.Dropout(cfg.get("input_dropout", 0.0))
    def forward(self, wid, cng, langs, switches):
        w=self.in_drop(self.word(wid)); c=self.in_drop(self.cng(cng).mean(1))
        x=torch.cat([w, c, self.lang(langs), self.switch(switches)], -1)
        return self.norm(self.proj(x))

class EdgeGate(nn.Module):
    def __init__(self, H, n_rel, hid):
        super().__init__(); self.rel_emb=nn.Embedding(n_rel,32)
        self.mlp=nn.Sequential(nn.Linear(H*2+32,hid), nn.GELU(), nn.Linear(hid,1))
    def forward(self, x, ei, et):
        s,d=ei; h=torch.cat([x[s],x[d],self.rel_emb(et)],-1)
        return torch.sigmoid(self.mlp(h)).squeeze(-1)

class RelationalGATLayer(MessagePassing):
    def __init__(self, H, n_rel, heads, dropout):
        super().__init__(aggr="add", node_dim=0)
        self.H,self.n_rel,self.heads=H,n_rel,heads; self.dh=H//heads
        self.lin=nn.ModuleList([nn.Linear(H,H) for _ in range(n_rel)])
        self.att=nn.Parameter(torch.empty(n_rel,heads,2*self.dh)); nn.init.xavier_uniform_(self.att)
        self.drop=nn.Dropout(dropout); self.norm=nn.LayerNorm(H); self._store_alpha=None
    def forward(self, x, ei, et, ew, eg, return_alpha=False):
        out=torch.zeros_like(x); alpha_all=torch.zeros(ei.size(1), device=x.device)
        for r in range(self.n_rel):
            m=et==r
            if m.sum()==0: continue
            o,a=self._prop(ei[:,m], self.lin[r](x), r, ew[m], eg[m])
            out=out+o; alpha_all[m]=a
        out=self.norm(F.gelu(out)+x)
        if return_alpha: self._store_alpha=alpha_all.detach()
        return out
    def _prop(self, ei, xr, r, ew, eg):
        s,d=ei
        xs=xr[s].view(-1,self.heads,self.dh); xd=xr[d].view(-1,self.heads,self.dh)
        e=F.leaky_relu((torch.cat([xs,xd],-1)*self.att[r]).sum(-1), 0.2)
        a=geo_softmax(e, d, num_nodes=xr.size(0))
        a=self.drop(a)*(ew*eg).unsqueeze(-1).to(a.dtype)
        msg=(xs*a.unsqueeze(-1)).view(-1,self.H)
        out=torch.zeros_like(xr); out.index_add_(0, d, msg.to(out.dtype))
        return out, a.mean(-1).float()

class AttnPool(nn.Module):
    def __init__(self, H): super().__init__(); self.score=nn.Linear(H,1); self.out=nn.Linear(H*3,H)
    def forward(self, x, batch, ng):
        x=x.float(); a=self.score(x).squeeze(-1)
        gmax=torch.full((ng,), -1e9, device=x.device).index_reduce_(0, batch, a, "amax", include_self=True)
        a=a-gmax[batch]; w=torch.exp(a).clamp_max(1e4)
        den=torch.zeros(ng, device=x.device).index_add_(0, batch, w).clamp_min(1e-9)
        attn=torch.zeros(ng, x.size(-1), device=x.device)
        attn.index_add_(0, batch, x*(w/den[batch]).unsqueeze(-1))
        mean=torch.zeros(ng, x.size(-1), device=x.device)
        cnt=torch.zeros(ng, device=x.device).index_add_(0, batch, torch.ones_like(w)).clamp_min(1.0)
        mean.index_add_(0, batch, x); mean=mean/cnt.unsqueeze(-1)
        mx=torch.full((ng, x.size(-1)), -1e9, device=x.device)
        mx.index_reduce_(0, batch, x, "amax", include_self=True)
        return self.out(torch.cat([attn, mean, mx], -1))

class HeCSGNN(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__(); H=cfg["gnn_hidden"]; self.cfg=cfg
        self.embed=NodeEmbedder(cfg)
        self.edge_gate=EdgeGate(H, cfg["n_relations"], cfg["edge_gate_hidden"])
        self.layers=nn.ModuleList([RelationalGATLayer(H, cfg["n_relations"], cfg["gnn_heads"], cfg["dropout"])
                                   for _ in range(cfg["gnn_layers"])])
        self.pool=AttnPool(H)
        self.proj_head=nn.Sequential(nn.Linear(H,H), nn.GELU(), nn.Linear(H,128))
        self.classifier=nn.Sequential(nn.Linear(H, H//2), nn.GELU(),
                                      nn.Dropout(cfg["dropout"]), nn.Linear(H//2, num_classes))
        self.last_gate=None
    def encode(self, b, return_xai=False):
        x=self.embed(b.wid, b.cng, b.langs, b.switches)
        ei, et, ew = b.edge_index, b.edge_type, b.edge_weight
        gate=self.edge_gate(x, ei, et)
        if return_xai: self.last_gate=(ei.detach(), et.detach(), gate.detach())
        for k, layer in enumerate(self.layers):
            x=layer(x, ei, et, ew, gate, return_alpha=(return_xai and k==len(self.layers)-1))
        return x
    def forward(self, b, return_xai=False):
        x=self.encode(b, return_xai=return_xai)
        g=self.pool(x, b.batch, b.num_graphs)
        return self.classifier(g)

# ----------------------------------------------------------------------------
# Build model + load weights
# ----------------------------------------------------------------------------
MODEL = HeCSGNN(CFG, NUM_CLASSES).to(DEVICE)
state = torch.load(os.path.join(DEPLOY_DIR, "hecs_gnn_weights.pt"), map_location=DEVICE)
MODEL.load_state_dict(state)
MODEL.eval()
print(f"Loaded HeCS-GNN | device={DEVICE} | classes={CLASSES} | vocab={VOCAB}")

# ----------------------------------------------------------------------------
# Prediction with code-mix diagnostics
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(text, show_relation_contrib=True):
    d, words, langs = sentence_to_graph(text)
    b = Batch.from_data_list([d]).to(DEVICE)
    logits = MODEL(b, return_xai=True)
    probs = F.softmax(logits.float(), -1).squeeze(0).cpu().tolist()
    pred_idx = int(max(range(NUM_CLASSES), key=lambda i: probs[i]))
    result = {"text": text, "prediction": CLASSES[pred_idx],
              "confidence": probs[pred_idx],
              "probs": {CLASSES[i]: round(probs[i], 4) for i in range(NUM_CLASSES)}}
    # code-mix stats of THIS sentence
    n_bn = int((langs == 0).sum()); n_tokens = len(words)
    n_switch = int(d.switches.sum())
    result["codemix"] = {"tokens": n_tokens, "bangla_words": n_bn,
                         "bangla_fraction": round(n_bn/max(n_tokens,1), 3),
                         "switch_points": n_switch}
    # per-relation gate mass — does the code-switch relation fire on this input?
    if show_relation_contrib and MODEL.last_gate is not None:
        ei, et, gate = MODEL.last_gate
        et = et.cpu().numpy(); gate = gate.cpu().numpy()
        contrib = {}
        total = float(gate.sum()) or 1.0
        for r in range(4):
            contrib[REL_NAMES[r]] = round(float(gate[et == r].sum())/total, 4)
        result["relation_contribution"] = contrib
    return result

def _print(res):
    print("\n" + "="*64)
    print(f"  TEXT : {res['text']}")
    print(f"  PRED : {res['prediction']}   (confidence {res['confidence']*100:.1f}%)")
    print(f"  PROBS: " + "  ".join(f"{k}={v}" for k,v in res['probs'].items()))
    cm = res["codemix"]
    print(f"  CODE-MIX: {cm['bangla_words']}/{cm['tokens']} Bangla words "
          f"({cm['bangla_fraction']*100:.0f}%), {cm['switch_points']} switch point(s)")
    if "relation_contribution" in res:
        rc = res["relation_contribution"]
        print("  RELATION CONTRIBUTION (share of learned edge-gate mass):")
        for k, v in rc.items():
            bar = "#" * int(v*40)
            print(f"     {k:16s} {v:6.3f} {bar}")
        sw = rc["switch Bn->En"] + rc["switch En->Bn"]
        if cm["switch_points"] == 0:
            print("     -> no code-switch in this sentence (switch relations inactive).")
        elif sw < 0.02:
            print("     -> switch relations fired but contribute ~nothing to the decision.")
        else:
            print(f"     -> switch relations carry {sw*100:.1f}% of gate mass on this input.")
    print("="*64)

def interactive():
    print("\nType a code-mixed (Bangla-English) sentence and press Enter.")
    print("Commands:  :q to quit\n")
    while True:
        try:
            text = input("codemix> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if text in (":q", ":quit", "exit"): break
        if not text: continue
        _print(predict(text))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        _print(predict(" ".join(sys.argv[1:])))
    else:
        interactive()