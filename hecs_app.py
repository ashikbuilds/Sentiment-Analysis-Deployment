"""
HeCS-GNN — Streamlit Web App with Visual XAI
============================================
An interactive browser UI for your trained HeCS-GNN code-mixed sentiment model.

Shows, for any sentence you type:
  • Prediction + class probabilities
  • Code-mix breakdown (Bangla fraction, switch points)
  • Per-relation contribution (does the code-switch relation fire?)
  • Token-level saliency (which words drove the decision)
  • The sentence's heterogeneous graph (4 typed relations)

SETUP
-----
1) Put this file next to hecs_infer.py and the deploy/ folder:

       my_folder/
       ├── hecs_app.py        <-- this file
       ├── hecs_infer.py      <-- your existing inference script (reused)
       └── deploy/            <-- exported from the training notebook
           ├── hecs_gnn_weights.pt
           ├── stoi.pkl
           ├── pmi.pkl
           └── meta.json

2) pip install streamlit torch torch-geometric emoji matplotlib networkx

3) streamlit run hecs_app.py
   (a browser tab opens automatically at http://localhost:8501)
"""

import os, sys
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F

# Reuse the model, preprocessing, and predict() from your existing inference script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hecs_infer as H   # this loads MODEL, STOI, PMI, CFG, predict(), etc.

st.set_page_config(page_title="HeCS-GNN Code-Mix Sentiment", page_icon="🔤", layout="wide")

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
REL_COLORS = {0: "#999999", 1: OKABE[2], 2: OKABE[3], 3: OKABE[4]}

# ----------------------------------------------------------------------------
# Extra XAI: token saliency (gradient norm) — computed with the loaded model
# ----------------------------------------------------------------------------
def token_saliency(text):
    d, words, langs = H.sentence_to_graph(text)
    from torch_geometric.data import Batch
    b = Batch.from_data_list([d]).to(H.DEVICE)
    x = H.MODEL.embed(b.wid, b.cng, b.langs, b.switches).detach().requires_grad_(True)
    ei, et, ew = b.edge_index, b.edge_type, b.edge_weight
    gate = H.MODEL.edge_gate(x, ei, et)
    h = x
    for layer in H.MODEL.layers:
        h = layer(h, ei, et, ew, gate)
    g = H.MODEL.pool(h, b.batch, b.num_graphs)
    logit = H.MODEL.classifier(g)
    tgt = logit.argmax(-1)
    grad = torch.autograd.grad(logit[0, tgt], x)[0]
    sal = grad.norm(dim=-1).detach().cpu().numpy()
    sal = sal[:len(words)]
    if sal.max() > 0:
        sal = sal / sal.max()
    return words, sal, int(tgt.item())

def highlight_tokens(words, sal):
    """Return HTML with each token shaded by its saliency (green=influential)."""
    spans = []
    for w, s in zip(words, sal):
        # blend white -> green by saliency
        r = int(255 - s * 175); g = int(255 - s * 55); bl = int(255 - s * 175)
        spans.append(
            f'<span style="background-color: rgb({r},{g},{bl}); padding:3px 6px; '
            f'margin:2px; border-radius:5px; display:inline-block; font-size:16px;">{w}</span>')
    return " ".join(spans)

def draw_graph(text):
    d, words, langs = H.sentence_to_graph(text)
    L = d.num_nodes
    ei = d.edge_index.numpy(); et = d.edge_type.numpy()
    switches = set((d.switches == 1).nonzero(as_tuple=True)[0].tolist())
    pos = nx.circular_layout(nx.path_graph(L))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for r, ax in enumerate(axes):
        G = nx.DiGraph(); G.add_nodes_from(range(L))
        for a, b_, t in zip(ei[0], ei[1], et):
            if t == r and a < L and b_ < L: G.add_edge(int(a), int(b_))
        node_col = [OKABE[0] if langs[i] == 0 else OKABE[1] for i in range(L)]
        ec = ["black" if i in switches else "none" for i in range(L)]
        lw = [2.2 if i in switches else 0 for i in range(L)]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.55, width=1.4,
                               edge_color=REL_COLORS[r], arrows=(r >= 2), arrowsize=11)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_col,
                               edgecolors=ec, linewidths=lw, node_size=400)
        labels = {i: (words[i][:8] if i < len(words) else str(i)) for i in range(L)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)
        ax.set_title(f"{H.REL_NAMES[r]}\n{G.number_of_edges()} edges", fontsize=10)
        ax.axis("off")
    fig.suptitle("Heterogeneous graph — 4 typed relations", y=1.04, fontsize=12)
    return fig

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🔤 HeCS-GNN — Code-Mixed Sentiment (with Graph XAI)")
st.caption(f"Transformer-free graph model · device: {H.DEVICE} · "
           f"classes: {', '.join(H.CLASSES)} · vocab: {H.VOCAB:,}")

with st.sidebar:
    st.header("About")
    st.markdown(
        "This app runs your trained **HeCS-GNN** on any Bangla-English "
        "code-mixed sentence and explains *why* it made its decision.\n\n"
        "**XAI shown:**\n"
        "- Per-relation contribution (does code-switch fire?)\n"
        "- Token saliency (influential words)\n"
        "- The sentence graph (4 typed relations)")
    st.markdown("---")
    st.markdown("**Try examples:**")
    examples = [
        "this product খুবই bad আমি kinbo na",
        "amazing product ভীষণ ভালো লাগলো",
        "worst experience ever টাকা নষ্ট",
        "delivery was fast and packaging ভালো ছিল",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["text"] = ex
            st.session_state["run_example"] = True
            st.rerun()

text = st.text_area("Enter a code-mixed sentence:",
                    value=st.session_state.get("text", ""),
                    placeholder="e.g.  this product খুবই bad আমি kinbo na",
                    height=90)

analyze = st.button("Analyze", type="primary")

# Run ONLY when the user clicks Analyze (or clicked a sidebar example, which
# sets a flag). Nothing is shown on first load or while just typing.
run_now = analyze or st.session_state.get("run_example", False)
st.session_state["run_example"] = False  # reset the one-shot example flag

if run_now:
    if not text.strip():
        st.warning("Type a sentence first, then click Analyze.")
        st.stop()

    res = H.predict(text)

    # ---- Row 1: prediction + probabilities ----
    c1, c2 = st.columns([1, 1])
    with c1:
        label = res["prediction"]; conf = res["confidence"]
        color = "#009E73" if label.lower().startswith(("pos", "good")) else "#D55E00"
        st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>",
                    unsafe_allow_html=True)
        st.metric("Confidence", f"{conf*100:.1f}%")
        cm = res["codemix"]
        st.markdown(
            f"**Code-mix:** {cm['bangla_words']}/{cm['tokens']} Bangla words "
            f"({cm['bangla_fraction']*100:.0f}%) · {cm['switch_points']} switch point(s)")
    with c2:
        st.markdown("#### Class probabilities")
        probs = res["probs"]
        fig, ax = plt.subplots(figsize=(4, 2.2))
        ax.barh(list(probs.keys()), list(probs.values()),
                color=[OKABE[2] if k == label else "#cccccc" for k in probs])
        ax.set_xlim(0, 1)
        for i, (k, v) in enumerate(probs.items()):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
        ax.set_xlabel("probability"); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # ---- Row 2: relation contribution (the key code-switch diagnostic) ----
    st.markdown("### 🔗 Relation contribution (why the graph decided this)")
    rc = res.get("relation_contribution", {})
    if rc:
        fig, ax = plt.subplots(figsize=(8, 2.6))
        names = list(rc.keys()); vals = list(rc.values())
        bars = ax.barh(names, vals, color=[REL_COLORS[i] for i in range(len(names))])
        ax.set_xlim(0, max(vals + [0.01]) * 1.15); ax.set_xlabel("share of learned edge-gate mass")
        for i, v in enumerate(vals): ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
        ax.invert_yaxis(); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        sw = rc.get("switch Bn->En", 0) + rc.get("switch En->Bn", 0)
        if cm["switch_points"] == 0:
            st.info("No code-switch in this sentence — the switch relations are inactive here.")
        elif sw < 0.02:
            st.warning(f"Code-switch relations fired but carry only {sw*100:.1f}% of the "
                       f"decision — the model relies mostly on sequential/semantic structure.")
        else:
            st.success(f"Code-switch relations carry **{sw*100:.1f}%** of the decision on this input.")

    st.markdown("---")

    # ---- Row 3: token saliency ----
    st.markdown("### 🎯 Token saliency (which words drove the prediction)")
    try:
        words, sal, tgt = token_saliency(text)
        st.markdown(highlight_tokens(words, sal), unsafe_allow_html=True)
        st.caption("Greener = more influential on the prediction (gradient magnitude).")
    except Exception as e:
        st.caption(f"(saliency unavailable: {e})")

    st.markdown("---")

    # ---- Row 4: the graph ----
    st.markdown("### 🕸️ Sentence graph (4 typed relations)")
    st.caption("Blue = Bangla token · Orange = Latin token · black ring = code-switch point")
    fig = draw_graph(text)
    st.pyplot(fig); plt.close(fig)

st.markdown("---")
st.caption("HeCS-GNN · transformer-free heterogeneous code-switch graph network. "
           "Note: if trained on machine-translated data, predictions on natural "
           "code-mix may be unreliable, but the relation-contribution readout still "
           "shows whether the switch pathway activates.")