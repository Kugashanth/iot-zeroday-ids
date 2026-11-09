"""Interactive dashboard for iotids runs.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

RUNS = Path(__file__).resolve().parents[1] / "runs"
st.set_page_config(page_title="IoT Zero-Day IDS", page_icon="🛡️", layout="wide")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def list_runs(kind: str) -> list[Path]:
    return sorted([p for p in RUNS.glob(f"*-{kind}*") if p.is_dir()], reverse=True) if RUNS.exists() else []


st.title("🛡️ Self-Supervised, Drift-Aware IoT Intrusion Detection")
st.caption("Models are trained on benign traffic only. Test attacks belong to families never seen during training.")

batch_runs, stream_runs = list_runs("batch"), list_runs("stream")
if not batch_runs and not stream_runs:
    st.warning("No runs found. Run `make demo` (or `iotids all --config configs/synthetic.yaml`) first.")
    st.stop()

tab_over, tab_thr, tab_stream, tab_score = st.tabs(
    ["📊 Zero-day benchmark", "🎚️ Threshold explorer", "🌊 Drift & adaptation", "🔍 Score your CSV"]
)

# ------------------------------------------------------------------ benchmark
with tab_over:
    if not batch_runs:
        st.info("No batch run yet.")
    else:
        run = Path(st.selectbox("Run", batch_runs, format_func=lambda p: p.name, key="brun"))
        m = load_csv(str(run / "metrics.csv"))
        zd = m[m["eval"].str.startswith("zero")].set_index("model")
        best = zd["roc_auc"].idxmax()
        c = st.columns(4)
        c[0].metric("Best model (ROC-AUC)", best)
        c[1].metric("Zero-day ROC-AUC", f"{zd.loc[best, 'roc_auc']:.4f}")
        c[2].metric("Recall @ threshold", f"{zd.loc[best, 'recall']:.1%}")
        c[3].metric("False-positive rate", f"{zd.loc[best, 'fpr']:.2%}")

        cols = ["eval", "roc_auc", "pr_auc", "tpr@fpr=1%", "fpr@tpr=95%", "precision", "recall", "f1", "fpr"]
        st.subheader("Metrics")
        for ev, g in m.groupby("eval", sort=False):
            st.markdown(f"**{ev}**")
            st.dataframe(g.set_index("model")[cols[1:]].style.format(precision=4).highlight_max(
                subset=["roc_auc", "pr_auc", "tpr@fpr=1%", "f1"], color="#1b9e7755"), width="stretch")

        sc = load_csv(str(run / "test_scores.csv.gz"))
        y = (sc["family"] != "Benign").astype(int)
        fig = go.Figure()
        for model in [c for c in sc.columns if c not in ("family", "label")]:
            fpr, tpr, _ = roc_curve(y, sc[model])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=model, mode="lines"))
        fig.update_layout(title="Zero-day ROC", xaxis_type="log", xaxis_title="FPR (log)",
                          yaxis_title="TPR", height=420, xaxis_range=[-4, 0])
        left, right = st.columns(2)
        left.plotly_chart(fig, width="stretch")
        fam = pd.read_csv(run / "per_family_detection.csv", index_col=0)
        right.plotly_chart(px.imshow(fam, text_auto=".2f", color_continuous_scale="Viridis", zmin=0, zmax=1,
                                     title="Detection rate per family (Benign = FPR)", height=420),
                           width="stretch")
        auc = pd.read_csv(run / "per_family_auc.csv", index_col=0)
        st.plotly_chart(px.imshow(auc, text_auto=".3f", color_continuous_scale="Viridis", zmin=0.5, zmax=1,
                                  title="Per-family ROC-AUC vs benign"), width="stretch")
        splits = json.load(open(run / "splits.json"))
        with st.expander("Split composition"):
            st.json(splits)

# ------------------------------------------------------------------ threshold explorer
with tab_thr:
    if batch_runs:
        run = Path(st.selectbox("Run", batch_runs, format_func=lambda p: p.name, key="trun"))
        sc = load_csv(str(run / "test_scores.csv.gz"))
        models = [c for c in sc.columns if c not in ("family", "label")]
        model = st.selectbox("Model", models)
        s = sc[model].to_numpy()
        benign = s[sc["family"] == "Benign"]
        q = st.slider("Threshold = benign quantile", 0.90, 0.9999, 0.99, 0.0005, format="%.4f")
        thr = float(np.quantile(benign, q))
        flagged = s > thr
        per = pd.Series(flagged).groupby(sc["family"]).mean().rename("detection rate")
        a, b = st.columns([2, 1])
        log_s = np.log10(s - s.min() + 1e-6)
        fig = px.histogram(pd.DataFrame({"log score": log_s, "family": sc["family"]}), x="log score",
                           color="family", barmode="overlay", histnorm="probability density", nbins=120, opacity=0.5)
        fig.add_vline(x=np.log10(thr - s.min() + 1e-6), line_dash="dash", line_color="red")
        a.plotly_chart(fig, width="stretch")
        y = sc["family"] != "Benign"
        b.metric("FPR", f"{flagged[~y].mean():.3%}")
        b.metric("Recall (unknown attacks)", f"{flagged[y].mean():.2%}")
        b.dataframe(per.to_frame().style.format("{:.3f}"), width="stretch")

# ------------------------------------------------------------------ stream
with tab_stream:
    if not stream_runs:
        st.info("No stream run yet — `iotids stream --config configs/synthetic.yaml`.")
    else:
        run = Path(st.selectbox("Stream run", stream_runs, format_func=lambda p: p.name))
        w = load_csv(str(run / "stream_windows.csv"))
        st.dataframe(load_csv(str(run / "stream_summary.csv")).style.format(precision=4), width="stretch")
        for metric, title in [("fpr", "False-positive rate per window"), ("recall", "Attack recall per window")]:
            fig = px.line(w, x="window", y=metric, color="mode", title=title,
                          color_discrete_map={"static": "#d95f02", "adaptive": "#1b9e77"})
            for x in w.loc[(w["mode"] == "adaptive") & w["adapted"], "window"]:
                fig.add_vline(x=x, line_dash="dot", line_color="#1b9e77")
            post = w[w["after_true_drift"]]
            if len(post):
                fig.add_vline(x=int(post["window"].min()), line_dash="dash", annotation_text="true drift")
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ scoring
with tab_score:
    if batch_runs:
        run = Path(st.selectbox("Trained run", batch_runs, format_func=lambda p: p.name, key="srun"))
        avail = sorted({p.name.split(".")[0] for p in (run / "models").glob("*.threshold.json")})
        model = st.selectbox("Model", avail, key="smodel")
        up = st.file_uploader("CSV of flows (same feature columns as training data)", type="csv")
        if up is not None:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from iotids.cli import predict

            tmp = run / "_upload.csv"
            tmp.write_bytes(up.getvalue())
            out = predict(run, model, tmp)
            tmp.unlink(missing_ok=True)
            st.metric("Flagged as attack", f"{out['is_attack_pred'].mean():.2%} of {len(out):,} flows")
            st.dataframe(out.sort_values("anomaly_score", ascending=False).head(500), width="stretch")
            st.download_button("Download scored CSV", out.to_csv(index=False), "scored.csv", "text/csv")
