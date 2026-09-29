"""
Crystallization depth analysis for extractive QA.

BERT (fine-tuned on SQuAD 1.1) is probed layer by layer to measure when the
start and end pointers settle on their final decision. RoBERTa (fine-tuned
on SQuAD 2.0) is used to track the model's answer/abstain choice across
layers on unanswerable questions.

Produces the figures, tables, and summary.json referenced in the paper.
"""

# %% Setup
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets"])
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])

import os, random, re, string, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy import stats
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import spacy

warnings.filterwarnings("ignore")
SEED, MAXLEN, NEG, N_LAYERS = 42, 384, -1e4, 13
SUMMARY = {}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "/kaggle/working/outputs" if os.path.isdir("/kaggle/working") else "outputs"
os.makedirs(output_dir, exist_ok=True)
print("device:", device, "| outputs ->", output_dir)

# %% Data and BERT checkpoint (SQuAD 1.1)
squad_v1_val = load_dataset("squad", split="validation")
squad_v2_val = load_dataset("squad_v2", split="validation")
train_raw = load_dataset("squad", split="train").shuffle(seed=SEED).select(range(3000))
print(f"SQuAD 1.1 dev: {len(squad_v1_val)} | SQuAD 2.0 dev: {len(squad_v2_val)} | probe-training slice: {len(train_raw)}")

BERT_NAME = "csarron/bert-base-uncased-squad-v1"
tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(BERT_NAME, output_hidden_states=True).to(device).eval()
print("loaded:", BERT_NAME)

# %% Helper functions
def normalize_answer(s):  # official SQuAD normalization
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t):
        ex = set(string.punctuation)
        return "".join(ch for ch in t if ch not in ex)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match(pred, golds):
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)

def mask_logits(lg, ctx):  # a pointer may only choose passage tokens
    return lg.masked_fill(ctx == 0, NEG)

def depth_from(pa, final):
    # earliest layer from which the probe agrees with the model's final decision
    # and keeps agreeing through the last layer; N_LAYERS = never settles
    mt = (pa == final[:, None]).astype(int)
    stable = np.flip(np.minimum.accumulate(np.flip(mt, 1), axis=1), 1)
    return np.where(stable.any(1), stable.argmax(1), N_LAYERS)

def best_span(sl, el, n_best=20, max_len=30):
    # best valid (start <= end, <=30 tokens) pair from the top-20 candidates on each side
    si = np.argsort(sl)[::-1][:n_best]
    ei = np.argsort(el)[::-1][:n_best]
    best, bs, be = -np.inf, None, None
    for s in si:
        if sl[s] < NEG / 2: continue
        for e in ei:
            if el[e] < NEG / 2 or e < s or e - s + 1 > max_len: continue
            sc = sl[s] + el[e]
            if sc > best: best, bs, be = sc, int(s), int(e)
    return bs, be

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp

def eta_squared(groups):
    allv = np.concatenate(groups); grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    return ssb / ((allv - grand) ** 2).sum()

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx]); adj[idx] = min(1.0, running)
    return adj

def fmt_p(p): return "< 1e-300" if p < 1e-300 else f"{p:.2e}"

def blend(c1, c2, a=0.5):  # color where two alpha-blended bars overlap
    c1, c2 = np.array(mcolors.to_rgb(c1)), np.array(mcolors.to_rgb(c2))
    return tuple(a * c2 + (1 - a) * (a * c1 + (1 - a) * np.ones(3)))

def overlap_hist(x1, x2, bins, name1, name2, xlabel, title, path, c1="steelblue", c2="darkorange", xticks=None):
    a = 0.5
    plt.figure(figsize=(10, 5))
    plt.hist(x1, bins=bins, alpha=a, density=True, color=c1)
    plt.hist(x2, bins=bins, alpha=a, density=True, color=c2)
    if xticks is not None: plt.xticks(xticks)
    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(title)
    plt.legend(handles=[Patch(facecolor=c1, alpha=a, label=f"{name1} only"),
                        Patch(facecolor=c2, alpha=a, label=f"{name2} only"),
                        Patch(facecolor=blend(c1, c2, a), label="Both overlap")])
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.show()

def bar_fig(tab, title, color, path, rot=0):
    plt.figure(figsize=(8, 5))
    plt.bar(tab.index.astype(str), tab["mean"], yerr=tab["ci95"], color=color, edgecolor="black", capsize=4)
    plt.axhline(0, color="black", lw=1)
    plt.title(title); plt.ylabel("Mean layer gap (start depth − end depth), 95% CI"); plt.xticks(rotation=rot)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.show()

# %% Tokenize the BERT data
def tok_bert(ex):
    enc = tokenizer(ex["question"], ex["context"], truncation="only_second",
                    max_length=MAXLEN, padding="max_length", return_offsets_mapping=True)
    enc["context_mask"] = [[int(s == 1) for s in enc.sequence_ids(i)] for i in range(len(enc["input_ids"]))]
    enc["truncated"] = [int(len(x) > MAXLEN) for x in tokenizer(ex["question"], ex["context"])["input_ids"]]
    return enc

tok_train = train_raw.map(tok_bert, batched=True, remove_columns=train_raw.column_names)
tok_val   = squad_v1_val.map(tok_bert, batched=True, remove_columns=squad_v1_val.column_names)

val_offsets = np.array(tok_val["offset_mapping"])  # (N, 384, 2) character offsets, used for decoding
val_trunc   = np.array(tok_val["truncated"])
keep = val_trunc == 0  # passages that fit in one window
print(f"Passages that do not fit in one {MAXLEN}-token window: {val_trunc.sum()} of {len(val_trunc)} "
      f"({val_trunc.mean():.1%}) -> excluded from every analysis")
SUMMARY["n_dev"] = int(len(keep)); SUMMARY["n_kept"] = int(keep.sum())

COLS = ["input_ids", "attention_mask", "token_type_ids", "context_mask"]
tok_train.set_format("torch", columns=COLS); tok_val.set_format("torch", columns=COLS)
train_loader = DataLoader(tok_train, batch_size=16, shuffle=True)
val_loader   = DataLoader(tok_val, batch_size=32)
print("train batches:", len(train_loader), "| val batches:", len(val_loader))

# %% Probes and control task
class LayerProbes(nn.Module):
    def __init__(self, num_layers=N_LAYERS, hidden=768):
        super().__init__()
        self.start_probes = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_layers)])
        self.end_probes   = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_layers)])
    def forward(self, hs):
        return ([p(h).squeeze(-1) for p, h in zip(self.start_probes, hs)],
                [p(h).squeeze(-1) for p, h in zip(self.end_probes, hs)])

# control task: every token type gets a fixed random score; the target is the passage
# position whose token type has the highest score, so it can only be learned by memorizing types
g = torch.Generator().manual_seed(SEED)
tab_s = torch.rand(tokenizer.vocab_size, generator=g).to(device)
tab_e = torch.rand(tokenizer.vocab_size, generator=g).to(device)
def control_targets(ids, ctx):
    ok = ctx.bool()
    return (tab_s[ids].masked_fill(~ok, -1.0).argmax(-1),
            tab_e[ids].masked_fill(~ok, -1.0).argmax(-1))

def fwd(b):
    return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], token_type_ids=b["token_type_ids"])

EPOCHS, LR = 3, 1e-3
loss_fn = nn.CrossEntropyLoss()

def train_probes(seed):
    # trains the real probes (target = the model's own decision) and the control probes together
    set_seed(seed)
    real, ctrl = LayerProbes().to(device), LayerProbes().to(device)
    opts = [torch.optim.AdamW(m_.parameters(), lr=LR) for m_ in (real, ctrl)]
    for ep in range(EPOCHS):
        for batch in tqdm(train_loader, desc=f"seed {seed} · epoch {ep+1}/{EPOCHS}"):
            b = {k: v.to(device) for k, v in batch.items()}
            ctx = b["context_mask"]
            with torch.no_grad():
                out = fwd(b); hs = out.hidden_states
                real_t = (mask_logits(out.start_logits, ctx).argmax(-1), mask_logits(out.end_logits, ctx).argmax(-1))
                ctrl_t = control_targets(b["input_ids"], ctx)
            for probes, opt, (t_s, t_e) in zip((real, ctrl), opts, (real_t, ctrl_t)):
                p_s, p_e = probes(hs)
                loss = sum(loss_fn(mask_logits(p_s[l], ctx), t_s) + loss_fn(mask_logits(p_e[l], ctx), t_e)
                           for l in range(N_LAYERS))
                opt.zero_grad(); loss.backward(); opt.step()
    return real.eval(), ctrl.eval()

# %% Validation pass
@torch.no_grad()
def val_pass(real, ctrl):
    R = {k: [] for k in ["pa_s", "pa_e", "ca_s", "ca_e", "fs", "fe", "cs", "ce", "sl", "el", "conf"]}
    for batch in tqdm(val_loader, desc="validation pass"):
        b = {k: v.to(device) for k, v in batch.items()}
        ctx = b["context_mask"]
        out = fwd(b); hs = out.hidden_states
        sl, el = mask_logits(out.start_logits, ctx), mask_logits(out.end_logits, ctx)
        r_s, r_e = real(hs); x_s, x_e = ctrl(hs)
        c_s, c_e = control_targets(b["input_ids"], ctx)
        am = lambda L: torch.stack([mask_logits(x, ctx).argmax(-1) for x in L], 1)  # (B, 13)
        vals = dict(pa_s=am(r_s), pa_e=am(r_e), ca_s=am(x_s), ca_e=am(x_e),
                    fs=sl.argmax(-1), fe=el.argmax(-1), cs=c_s, ce=c_e, sl=sl, el=el,
                    conf=sl.softmax(-1).max(-1).values * el.softmax(-1).max(-1).values)
        for k, v in vals.items(): R[k].append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in R.items()}

# %% Train probes, per-layer accuracy vs control (Figure 1)
def layer_table(R):
    t = pd.DataFrame({
        "real_start": (R["pa_s"] == R["fs"][:, None])[keep].mean(0),
        "real_end":   (R["pa_e"] == R["fe"][:, None])[keep].mean(0),
        "ctrl_start": (R["ca_s"] == R["cs"][:, None])[keep].mean(0),
        "ctrl_end":   (R["ca_e"] == R["ce"][:, None])[keep].mean(0)})
    t["selectivity_start"] = t.real_start - t.ctrl_start
    t["selectivity_end"]   = t.real_end - t.ctrl_end
    t.index.name = "layer"
    return t.round(3)

real, ctrl = train_probes(SEED)
R = val_pass(real, ctrl)

lt = layer_table(R)
print("\n=== Per-layer probe accuracy (real vs. control), kept dev examples ===")
print(lt.to_string())
SUMMARY["layer12_real_start"] = float(lt.real_start.iloc[12]); SUMMARY["layer12_real_end"] = float(lt.real_end.iloc[12])
SUMMARY["layer12_ctrl_start"] = float(lt.ctrl_start.iloc[12]); SUMMARY["layer12_ctrl_end"] = float(lt.ctrl_end.iloc[12])
lt.to_csv(os.path.join(output_dir, "per_layer_probe_accuracy.csv"))

x = np.arange(N_LAYERS)
plt.figure(figsize=(9, 5))
plt.plot(x, lt.real_start, "-o", color="steelblue", label="real probe · start")
plt.plot(x, lt.real_end, "-o", color="darkorange", label="real probe · end")
plt.plot(x, lt.ctrl_start, "--s", color="steelblue", label="control probe · start")
plt.plot(x, lt.ctrl_end, "--s", color="darkorange", label="control probe · end")
plt.xticks(x); plt.ylim(0, 1.02); plt.xlabel("Transformer layer"); plt.ylabel("Accuracy on validation examples")
plt.title("Probe accuracy per layer: real task vs. control task"); plt.legend()
plt.savefig(os.path.join(output_dir, "fig7_probe_accuracy_per_layer.png"), dpi=150, bbox_inches="tight"); plt.show()

# %% RQ1: start vs end crystallization depth (Figure 1, seed comparison)
def rq1_stats(R):
    ds = depth_from(R["pa_s"], R["fs"]); de = depth_from(R["pa_e"], R["fe"])
    ok = keep & (ds < N_LAYERS) & (de < N_LAYERS)
    diff = (ds - de)[ok]
    out = dict(n_kept=int(keep.sum()), never_start=(ds[keep] == N_LAYERS).mean(), never_end=(de[keep] == N_LAYERS).mean(),
               n_ok=int(ok.sum()), mean_start=ds[ok].mean(), mean_end=de[ok].mean(), gap=diff.mean(),
               d_z=diff.mean() / diff.std(ddof=1),
               d_av=diff.mean() / np.sqrt((ds[ok].var(ddof=1) + de[ok].var(ddof=1)) / 2),
               p=stats.ttest_rel(ds[ok], de[ok]).pvalue,
               gap_legacy=(np.minimum(ds, 12) - np.minimum(de, 12))[keep].mean())
    return out, ds, de, ok

out42, ds, de, depth_ok = rq1_stats(R)
diff = (ds - de)[depth_ok]
rng = np.random.default_rng(SEED)
boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
ci = np.percentile(boot, [2.5, 97.5])

print("\n=== RQ1: start vs. end crystallization depth (seed 42) ===")
print(f"kept examples: {out42['n_kept']} | share whose layer-12 probe disagrees with the model ('never settles'): "
      f"start {out42['never_start']:.1%}, end {out42['never_end']:.1%}")
print(f"analysed (both probes agree at layer 12): {out42['n_ok']}")
print(f"mean start depth {out42['mean_start']:.3f} | mean end depth {out42['mean_end']:.3f} | gap (start - end) {out42['gap']:.3f}"
      f"  [95% bootstrap CI {ci[0]:.3f}, {ci[1]:.3f}]")
print(f"paired t-test p = {fmt_p(out42['p'])} | Cohen's d_z = {out42['d_z']:.4f} | d_av = {out42['d_av']:.4f}")
print(f"old-style coding (never folded into layer 12): gap = {out42['gap_legacy']:.3f}")
SUMMARY["rq1"] = {k: float(v) for k, v in out42.items()}; SUMMARY["rq1"]["ci_low"], SUMMARY["rq1"]["ci_high"] = float(ci[0]), float(ci[1])

overlap_hist(ds[depth_ok], de[depth_ok], np.arange(-0.5, 13.5, 1), "Start depth", "End depth",
             "Transformer layer", "Crystallization Depth: Start vs. End Pointers",
             os.path.join(output_dir, "fig1_crystallization_histogram.png"), xticks=range(13))

RUN_EXTRA_SEEDS = True  # ~7 minutes per extra seed
seed_rows = {SEED: out42}
if RUN_EXTRA_SEEDS:
    for s in (43, 44):
        r_, c_ = train_probes(s)
        seed_rows[s] = rq1_stats(val_pass(r_, c_))[0]
        del r_, c_
print("\n=== RQ1 across probe seeds ===")
seed_df = pd.DataFrame(seed_rows).T.round(4)
print(seed_df.to_string())
SUMMARY["rq1_seeds"] = seed_df.to_dict()

# %% Standard decoding, exact match, master table
contexts  = list(squad_v1_val["context"])
questions = list(squad_v1_val["question"])
golds     = [a["text"] for a in squad_v1_val["answers"]]
N = len(contexts)

bs_l, be_l, cs_l, ce_l, preds, corr = [], [], [], [], [], []
for i in range(N):
    s, e = best_span(R["sl"][i], R["el"][i])
    if s is None:
        bs_l.append(-1); be_l.append(-1); cs_l.append(0); ce_l.append(0); preds.append(""); corr.append(False); continue
    c0, c1 = int(val_offsets[i][s][0]), int(val_offsets[i][e][1])
    txt = contexts[i][c0:c1]
    bs_l.append(s); be_l.append(e); cs_l.append(c0); ce_l.append(c1); preds.append(txt); corr.append(exact_match(txt, golds[i]))

dv = pd.DataFrame(dict(keep=keep, ds=ds, de=de, depth_ok=depth_ok, bs=bs_l, be=be_l, cstart=cs_l, cend=ce_l,
                       pred=preds, correct=corr, fs=R["fs"], fe=R["fe"], conf=R["conf"]))
dv["consistent"] = (dv.bs == dv.fs) & (dv.be == dv.fe)  # standard best span == each pointer's own top choice
dv["length"] = dv.be - dv.bs + 1

print("\n=== Exact match sanity check ===")
print(f"EM on kept examples: {dv[dv.keep].correct.mean():.2%}   (this checkpoint's documented EM on the full dev set: 80.9%)")
print(f"share of kept examples where the standard span equals the pointers' own top choices: {dv[dv.keep].consistent.mean():.1%}")
SUMMARY["em_kept"] = float(dv[dv.keep].correct.mean()); SUMMARY["consistent_share"] = float(dv[dv.keep].consistent.mean())

# %% RQ2: gap by answer length and syntactic root (Figures 2, 3)
dfA = dv[dv.keep & dv.correct & dv.consistent & dv.depth_ok].copy()
dfA["gap"] = dfA.ds - dfA.de
print("\n=== Analysis set for RQ2 ===")
print(f"kept {int(dv.keep.sum())} -> correct {int((dv.keep & dv.correct).sum())} -> also consistent "
      f"{int((dv.keep & dv.correct & dv.consistent).sum())} -> also both probes settle {len(dfA)}")
print(f"mean gap in this set: {dfA.gap.mean():.3f}")
SUMMARY["n_rq2"] = int(len(dfA))

def group_report(col, order, title):
    grp = {k: dfA.loc[dfA[col] == k, "gap"].to_numpy() for k in order}
    tab = pd.DataFrame({k: dict(mean=v.mean(), sd=v.std(ddof=1), n=len(v), ci95=1.96 * v.std(ddof=1) / np.sqrt(len(v)))
                        for k, v in grp.items()}).T
    one = [stats.ttest_1samp(v, 0) for v in grp.values()]
    tab["p_vs_0_holm"] = holm([o.pvalue for o in one])
    F, p = stats.f_oneway(*grp.values()); eta2 = eta_squared(list(grp.values()))
    print(f"\n--- {title} ---")
    print(tab.round(4).to_string())
    print(f"ANOVA: F = {F:.2f}, p = {fmt_p(p)}, eta^2 = {eta2:.4f}")
    tuk = stats.tukey_hsd(*grp.values()); names = list(grp); rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append((f"{names[i]} vs {names[j]}", grp[names[i]].mean() - grp[names[j]].mean(),
                         tuk.pvalue[i, j], cohens_d(grp[names[i]], grp[names[j]])))
    print("pairwise (Tukey HSD):")
    print(pd.DataFrame(rows, columns=["pair", "mean_diff", "tukey_p", "cohens_d"]).round(4).to_string(index=False))
    return tab, F, p, eta2

dfA["len_cat"] = pd.cut(dfA.length, [0, 2, 5, 10**6], labels=["Short (1-2)", "Medium (3-5)", "Long (6+)"])
len_order = ["Short (1-2)", "Medium (3-5)", "Long (6+)"]
len_tab, F_len, p_len, eta_len = group_report("len_cat", len_order, "Gap by predicted-answer length (tokens)")
bar_fig(len_tab, "End-to-Start Depth Gap by Span Length (Correct Answers Only)", "coral",
        os.path.join(output_dir, "fig2_gap_by_length.png"))

# syntactic root, parsed inside the passage (not the extracted answer in isolation)
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
need = sorted(set(contexts[i] for i in dfA.index))
print(f"\nparsing {len(need)} unique passages with spaCy ...")
docs = dict(zip(need, nlp.pipe(need, batch_size=32)))
def root_pos(i, c0, c1):
    sp = docs[contexts[i]].char_span(int(c0), int(c1), alignment_mode="expand")
    return sp.root.pos_ if sp is not None else "UNKNOWN"
dfA["root_pos"] = [root_pos(i, r.cstart, r.cend) for i, r in dfA.iterrows()]

top = dfA.root_pos.value_counts().nlargest(5).index.tolist()
cover = dfA.root_pos.isin(top).sum()
print(f"top-5 roots {top} cover {cover} of {len(dfA)} examples; others (not analysed): {dfA.root_pos.value_counts().drop(top).to_dict()}")
dfS = dfA[dfA.root_pos.isin(top)]
order_syn = dfS.groupby("root_pos").gap.mean().sort_values(ascending=False).index.tolist()
syn_tab, F_syn, p_syn, eta_syn = group_report("root_pos", order_syn, "Gap by syntactic root of the answer (parsed in context)")
bar_fig(syn_tab, "End-to-Start Depth Gap by Syntactic Category", "skyblue",
        os.path.join(output_dir, "fig3_gap_by_syntax.png"), rot=45)

SUMMARY["length"] = dict(F=float(F_len), p=float(p_len), eta2=float(eta_len), table=len_tab.round(4).to_dict())
SUMMARY["syntax"] = dict(F=float(F_syn), p=float(p_syn), eta2=float(eta_syn), table=syn_tab.round(4).to_dict())

# %% RQ3: does crystallization depth add anything to softmax confidence? (Figure 4)
dk = dv[dv.keep]
y = dk.correct.astype(int).to_numpy()
conf = dk.conf.to_numpy()
dscore = (N_LAYERS - np.maximum(dk.ds, dk.de)).to_numpy()  # higher = settled earlier (13 = never -> 0)

auc_conf, auc_depth = roc_auc_score(y, conf), roc_auc_score(y, dscore)
X = np.column_stack([np.log(conf + 1e-12), dscore])
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
p_both = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
p_conf = cross_val_predict(clf, X[:, :1], y, cv=cv, method="predict_proba")[:, 1]
auc_both, auc_conf_lr = roc_auc_score(y, p_both), roc_auc_score(y, p_conf)

rng = np.random.default_rng(SEED); diffs = []
for _ in range(1000):
    idx = rng.integers(0, len(y), len(y))
    if y[idx].min() == y[idx].max(): continue
    diffs.append(roc_auc_score(y[idx], p_both[idx]) - roc_auc_score(y[idx], p_conf[idx]))
ci_d = np.percentile(diffs, [2.5, 97.5])

print("\n=== RQ3: AUROC for predicting a correct answer (kept examples) ===")
print(f"n = {len(y)} | share correct = {y.mean():.1%}")
print(f"softmax confidence alone (raw score):        {auc_conf:.4f}")
print(f"crystallisation depth alone (13 - max depth): {auc_depth:.4f}")
print(f"logistic regression on confidence only (5-fold CV):        {auc_conf_lr:.4f}")
print(f"logistic regression on confidence + depth (5-fold CV):     {auc_both:.4f}")
print(f"gain from adding depth: {auc_both - auc_conf_lr:+.4f}  [95% bootstrap CI {ci_d[0]:+.4f}, {ci_d[1]:+.4f}]")
SUMMARY["auroc"] = dict(conf=float(auc_conf), depth=float(auc_depth), conf_lr=float(auc_conf_lr), both=float(auc_both),
                        gain=float(auc_both - auc_conf_lr), gain_ci_low=float(ci_d[0]), gain_ci_high=float(ci_d[1]))

plt.figure(figsize=(8, 6))
for score, lab, c in [(conf, f"Softmax confidence (AUC = {auc_conf:.3f})", "blue"),
                      (dscore, f"Crystallization depth (AUC = {auc_depth:.3f})", "orange"),
                      (p_both, f"Confidence + depth, CV (AUC = {auc_both:.3f})", "green")]:
    f_, t_, _ = roc_curve(y, score); plt.plot(f_, t_, label=lab, color=c)
plt.plot([0, 1], [0, 1], "k--", label="Random guessing")
plt.xlabel("False Positive Rate (predicting correct when actually wrong)"); plt.ylabel("True Positive Rate")
plt.title("Error Prediction: Softmax vs. Crystallization Depth"); plt.legend()
plt.savefig(os.path.join(output_dir, "fig4_roc.png"), dpi=150, bbox_inches="tight"); plt.show()

# %% Example cases, drawn at random (not the first match)
rng = np.random.default_rng(SEED)
rows = []
def pick(mask, label):
    cand = dfA.index[mask.to_numpy()]
    if len(cand) == 0:
        print(f"{label}: no qualifying example"); return
    i = int(rng.choice(cand)); r = dfA.loc[i]
    rows.append(dict(Case=label, Question=questions[i], Answer=r.pred, Start=int(r.ds), End=int(r.de), Gap=int(r.gap), Candidates=len(cand)))
pick((dfA.len_cat == "Medium (3-5)") & (dfA.gap >= 4), "End resolves earlier")
pick((dfA.root_pos == "ADJ") & (dfA.gap <= -3), "Start resolves earlier (ADJ)")
pick((dfA.len_cat == "Long (6+)") & (dfA.ds >= 11) & (dfA.de >= 11), "Both resolve late")
case_df = pd.DataFrame(rows)
print("\n=== Example cases (one drawn at random per qualifying set; 'Candidates' = set size) ===")
print(case_df.to_string(index=False))
case_df.to_csv(os.path.join(output_dir, "case_studies.csv"), index=False)

# %% RoBERTa on SQuAD 2.0: layer-wise answer/abstain choice
NAME2 = "deepset/roberta-base-squad2"
tok2 = AutoTokenizer.from_pretrained(NAME2)
model2 = AutoModelForQuestionAnswering.from_pretrained(NAME2, output_hidden_states=True).to(device).eval()

def tok_v2(ex):
    enc = tok2(ex["question"], ex["context"], truncation="only_second", max_length=MAXLEN, padding="max_length")
    enc["allowed"] = [[int(j == 0 or s == 1) for j, s in enumerate(enc.sequence_ids(i))] for i in range(len(enc["input_ids"]))]
    enc["truncated"] = [int(len(x) > MAXLEN) for x in tok2(ex["question"], ex["context"])["input_ids"]]
    return enc

tok_ds = squad_v2_val.map(tok_v2, batched=True, remove_columns=squad_v2_val.column_names)
gold_unans = np.array([len(a["text"]) == 0 for a in squad_v2_val["answers"]])
keep2 = np.asarray(tok_ds["truncated"]) == 0
tok_ds.set_format("torch", columns=["input_ids", "attention_mask", "allowed"])
loader2 = DataLoader(tok_ds, batch_size=32)

states, margins = [], []  # states[n, l-1] = True if layer l's top start choice is <s>
with torch.no_grad():
    for batch in tqdm(loader2, desc="RoBERTa layers"):
        allowed = batch["allowed"].to(device).bool()
        out = model2(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        S, M = [], []
        for l in range(1, 13):
            lg = model2.qa_outputs(out.hidden_states[l])[..., 0].masked_fill(~allowed, NEG)
            S.append(lg.argmax(-1) == 0)
            M.append(lg[:, 0] - lg[:, 1:].max(-1).values)
        states.append(torch.stack(S, 1).cpu()); margins.append(torch.stack(M, 1).cpu())
states = torch.cat(states).numpy()
margins = torch.cat(margins).float().numpy()
print("collected:", states.shape, "| answerable:", int((~gold_unans).sum()), "| unanswerable:", int(gold_unans.sum()),
      "| passages that fit one window:", int(keep2.sum()))

# %% Flip analysis (Figures 5, 6)
from scipy.stats import mannwhitneyu

flips  = (states[:, 1:] != states[:, :-1]).sum(1)
forced = (states[:, 0] != states[:, -1]).astype(int)
excess = flips - forced  # > 0 means genuine back-and-forth beyond the one forced transition
mt = states.astype(int)
stable = np.flip(np.minimum.accumulate(np.flip(mt, 1), axis=1), 1)
onset = np.where(states[:, -1], stable.argmax(1) + 1, 0)

fdf = pd.DataFrame(dict(gold_unans=gold_unans, final_abstain=states[:, -1], flips=flips, excess=excess, onset=onset))[keep2]
print(f"\n=== Flip analysis: {len(fdf)} of {len(keep2)} SQuAD 2.0 dev examples (passage fits one window) ===")
print(f"layer-1 start decision is <s> in {states[keep2, 0].mean():.1%} of examples")

print("\n(1) flips by gold label")
print(pd.crosstab(fdf.gold_unans, fdf.flips).to_string())
g1 = fdf.groupby("gold_unans").agg(n=("flips", "size"), mean_flips=("flips", "mean"), sd_flips=("flips", "std"),
                                   share_ge1_flip=("flips", lambda x: (x >= 1).mean()),
                                   share_back_and_forth=("excess", lambda x: (x > 0).mean()))
print(g1.round(4).to_string())
a, b = fdf.loc[fdf.gold_unans, "flips"].to_numpy(), fdf.loc[~fdf.gold_unans, "flips"].to_numpy()
tt = stats.ttest_ind(a, b)
print(f"mean flips: unanswerable {a.mean():.3f} vs answerable {b.mean():.3f} | t-test p {fmt_p(tt.pvalue)} | Cohen's d = {cohens_d(a, b):.3f}")

print("\n(2) gold label x the model's final decision")
print(pd.crosstab(fdf.gold_unans, fdf.final_abstain, margins=True).to_string())

print("\n(3) by gold label and final decision")
g3 = fdf.groupby(["gold_unans", "final_abstain"]).agg(n=("flips", "size"), mean_flips=("flips", "mean"),
                                                      share_back_and_forth=("excess", lambda x: (x > 0).mean()),
                                                      median_onset_layer=("onset", lambda x: x[x > 0].median()))
print(g3.round(3).to_string())

a = fdf[~fdf.final_abstain & fdf.gold_unans].flips;  b = fdf[~fdf.final_abstain & ~fdf.gold_unans].flips
if len(a) and len(b):
    print(f"\nmodel answered · flips, unanswerable (n={len(a)}) vs answerable (n={len(b)}): "
          f"means {a.mean():.3f} vs {b.mean():.3f}, Mann-Whitney p = {fmt_p(mannwhitneyu(a, b).pvalue)}")
a = fdf[fdf.final_abstain & fdf.gold_unans].onset;   b = fdf[fdf.final_abstain & ~fdf.gold_unans].onset
if len(a) and len(b):
    print(f"model abstained · onset layer, unanswerable (n={len(a)}) vs answerable (n={len(b)}): "
          f"medians {a.median():.1f} vs {b.median():.1f}, Mann-Whitney p = {fmt_p(mannwhitneyu(a, b).pvalue)}")

SUMMARY["flips"] = dict(mean_unans=float(fdf.loc[fdf.gold_unans, 'flips'].mean()), mean_ans=float(fdf.loc[~fdf.gold_unans, 'flips'].mean()),
                        d=float(cohens_d(fdf.loc[fdf.gold_unans, 'flips'].to_numpy(), fdf.loc[~fdf.gold_unans, 'flips'].to_numpy())),
                        share_back_and_forth=float((fdf.excess > 0).mean()),
                        abstain_rate_unans=float(fdf.loc[fdf.gold_unans, 'final_abstain'].mean()),
                        abstain_rate_ans=float(fdf.loc[~fdf.gold_unans, 'final_abstain'].mean()))
fdf.to_csv(os.path.join(output_dir, "flip_table.csv"), index=False)

overlap_hist(fdf.loc[~fdf.gold_unans, "flips"], fdf.loc[fdf.gold_unans, "flips"],
             np.arange(-0.5, max(2, flips.max()) + 1.5, 1), "Answerable", "Unanswerable",
             "Number of flips (layer-to-layer changes between <s> and a passage token)",
             "Flip Count Distribution: Answerable vs. Unanswerable Questions",
             os.path.join(output_dir, "fig5_flip_histogram.png"), xticks=range(0, int(max(2, flips.max())) + 1))

plt.figure(figsize=(9, 5))
for (gg, ff), lab, c, ls in [((False, False), "answerable · model answered", "tab:blue", "-"),
                             ((False, True),  "answerable · model abstained", "tab:blue", "--"),
                             ((True,  False), "unanswerable · model answered", "tab:orange", "-"),
                             ((True,  True),  "unanswerable · model abstained", "tab:orange", "--")]:
    sel = keep2 & (gold_unans == gg) & (states[:, -1] == ff)
    if sel.sum() > 20:
        plt.plot(range(1, 13), margins[sel].mean(0), color=c, ls=ls, label=f"{lab} (n={sel.sum()})")
plt.axhline(0, color="k", lw=1)
plt.xticks(range(1, 13)); plt.xlabel("Layer"); plt.ylabel("<s> logit minus best passage-token logit (mean)")
plt.title("Abstention margin by layer (above 0 = <s> is the top choice)"); plt.legend(fontsize=8)
plt.savefig(os.path.join(output_dir, "fig6_abstention_margin.png"), dpi=150, bbox_inches="tight"); plt.show()

# %% Robustness: passage-level resampling, joint regression, single-token check
import statsmodels.formula.api as smf
pid = pd.factorize(pd.Series(contexts))[0]

# RQ1 gap resampled at the passage level, since questions from the same passage aren't independent
idx = np.where(depth_ok)[0]
d_ok = (ds - de)[idx]
_, inv = np.unique(pid[idx], return_inverse=True)
sums, cnts = np.bincount(inv, weights=d_ok), np.bincount(inv)
rng = np.random.default_rng(SEED)
boot = []
for _ in range(5000):
    pick = rng.integers(0, len(sums), len(sums))
    boot.append(sums[pick].sum() / cnts[pick].sum())
ci_p = np.percentile(boot, [2.5, 97.5])
print("\n=== RQ1 gap with a passage-level bootstrap ===")
print(f"gap = {d_ok.mean():.3f} | 95% CI resampling whole passages [{ci_p[0]:.3f}, {ci_p[1]:.3f}] | passages = {len(sums)}")
SUMMARY["rq1_passage_ci"] = [float(ci_p[0]), float(ci_p[1])]

# length and syntactic root together, standard errors clustered by passage
dfS2 = dfS.assign(passage=pid[dfS.index.to_numpy()], len_cat=dfS.len_cat.astype(str))
m_ols = smf.ols("gap ~ C(root_pos, Treatment('NOUN')) + C(len_cat, Treatment('Short (1-2)'))", data=dfS2) \
           .fit(cov_type="cluster", cov_kwds={"groups": dfS2["passage"]})
print("\n=== RQ2 regression: gap ~ syntactic root + length (reference = NOUN, Short); SEs clustered by passage ===")
print(m_ols.summary().tables[1])

adj = dfS[dfS.root_pos == "ADJ"]
print("\n=== Most common ADJ-rooted answers ===")
print(adj.assign(pred=adj.pred.str.lower()).groupby("pred").agg(n=("gap", "size"), mean_gap=("gap", "mean"))
         .sort_values("n", ascending=False).head(15).round(2).to_string())
print(f"single-token answers: {(adj.length == 1).mean():.1%} of ADJ answers vs {(dfS.length == 1).mean():.1%} of all analysed answers")
print("\n=== Gap by root, single-token answers only ===")
print(dfS[dfS.length == 1].groupby("root_pos").gap.agg(["mean", "size"]).round(3).to_string())

# %% Abstention follow-ups: onset layers, information beyond the final layer
from scipy.stats import mannwhitneyu

print("\n=== Onset layer (first layer from which <s> stays on top), questions the model abstained on ===")
ab = fdf[fdf.final_abstain]
print(pd.crosstab(ab.gold_unans.map({True: "unanswerable (correct abstention)", False: "answerable (wrong abstention)"}), ab.onset).to_string())
a, b = ab[ab.gold_unans].onset.to_numpy(), ab[~ab.gold_unans].onset.to_numpy()
U = mannwhitneyu(a, b).statistic
p_earlier = 1 - U / (len(a) * len(b))
print(f"probability that a correct abstention has an earlier onset than a wrong one (ties split): {p_earlier:.3f}")
print(f"share with onset at layer 12: correct {np.mean(a == 12):.1%} | wrong {np.mean(b == 12):.1%}")
SUMMARY["onset"] = dict(p_earlier=float(p_earlier), share12_correct=float(np.mean(a == 12)), share12_wrong=float(np.mean(b == 12)))

print("\n=== Does anything before the last layer predict unanswerability beyond the final-layer margin? ===")
K = np.where(keep2)[0]
fin = states[K, -1]
feats = np.column_stack([margins[K, -1], margins[K, :-1].max(1), margins[K, :-1].mean(1)])
yy = gold_unans[K].astype(int)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
SUMMARY["route"] = {}
for name, sel in [("model answered", ~fin), ("model abstained", fin)]:
    y_, F_ = yy[sel], feats[sel]
    p_fin = cross_val_predict(clf, F_[:, :1], y_, cv=cv, method="predict_proba")[:, 1]
    p_all = cross_val_predict(clf, F_, y_, cv=cv, method="predict_proba")[:, 1]
    rng = np.random.default_rng(SEED); dif = []
    for _ in range(500):
        ii = rng.integers(0, len(y_), len(y_))
        if y_[ii].min() == y_[ii].max(): continue
        dif.append(roc_auc_score(y_[ii], p_all[ii]) - roc_auc_score(y_[ii], p_fin[ii]))
    lo, hi = np.percentile(dif, [2.5, 97.5])
    print(f"{name} (n={len(y_)}, unanswerable share {y_.mean():.1%}): AUROC final margin only {roc_auc_score(y_, p_fin):.4f} | "
          f"final + earlier layers {roc_auc_score(y_, p_all):.4f} | gain {roc_auc_score(y_, p_all) - roc_auc_score(y_, p_fin):+.4f} [{lo:+.4f}, {hi:+.4f}]")
    SUMMARY["route"][name] = dict(auc_final=float(roc_auc_score(y_, p_fin)), auc_all=float(roc_auc_score(y_, p_all)), gain_ci=[float(lo), float(hi)])

# %% Save summary
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
print(json.dumps(clean(SUMMARY), indent=2))
with open(os.path.join(output_dir, "summary.json"), "w") as f: json.dump(clean(SUMMARY), f, indent=2)
dv.to_csv(os.path.join(output_dir, "dev_master_table.csv"), index=False)
np.savez_compressed(os.path.join(output_dir, "flip_arrays.npz"), states=states, margins=margins, gold_unans=gold_unans, keep2=keep2)
print("\nfiles in", output_dir, ":", sorted(os.listdir(output_dir)))
