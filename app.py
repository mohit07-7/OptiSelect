import warnings
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

import experiments as E
import plots as P
from datasets import DATASETS, load_dataset as _load_dataset

warnings.filterwarnings("ignore")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

st.set_page_config(page_title="OptiSelect AI", page_icon="🧬", layout="wide")
st.markdown("""
<style>
 .hero {padding:1.3rem 1.6rem;border-radius:18px;color:white;margin-bottom:1rem;
        background:linear-gradient(115deg,#14213d,#234e83 65%,#3976a8);}
 .hero h1 {font-size:2.1rem;margin:0;color:white;} .hero p {color:#dce8fa;margin:.35rem 0 0 0;}
 div[data-testid="stMetric"] {background:white;padding:14px;border:1px solid #e6ebf3;border-radius:14px;}
 [data-testid="stSidebar"] {background:#101a32;} [data-testid="stSidebar"] * {color:#eef3ff !important;}
</style>
<div class="hero"><h1>🧬 OptiSelect AI | Feature Selection Dashboard</h1>
<p>Compare classification with ALL features vs automatically selected subsets (SelectKBest, RFE, Manual)</p></div>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=86400)
def load_dataset(name):
    return _load_dataset(name)


# ------------------------------------------------------------ sidebar ----
st.sidebar.markdown("## CONTROLS")
dataset = st.sidebar.selectbox("Dataset", list(DATASETS))
classifier = st.sidebar.selectbox("Classifier", E.CLASSIFIERS)
method = st.sidebar.selectbox("Feature selection method", E.METHODS)
test_size = st.sidebar.slider("Test set size", 0.15, 0.35, 0.20, 0.05)
rs = int(st.sidebar.number_input("Random state", 0, 9999, 42, 1))
st.sidebar.caption("Stratified split. Imputation, scaling and selection are fitted on training data only (Pipeline).")

try:
    with st.spinner(f"Loading {dataset}..."):
        X, y, class_names, raw = load_dataset(dataset)
except Exception as exc:
    st.error(f"Could not load **{dataset}**.")
    st.markdown(
        "This dataset is downloaded from UCI. **Fix:** on your PC run `python download_data.py` once, then "
        "commit the created `data/` folder to GitHub and reboot the Streamlit app (the app then never needs the internet).")
    st.code(str(exc))
    st.stop()

n_features = X.shape[1]

# ----------------------------------------------------------- overview ----
st.markdown("### Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(X):,}")
c2.metric("Input features", n_features)
c3.metric("Classes", y.nunique())
c4.metric("Missing cells", f"{int(X.isna().sum().sum()):,}")
st.caption(DATASETS[dataset]["info"])
with st.expander(f"Dataset preview - {dataset} ({len(X)} rows, {n_features} features)"):
    st.dataframe(X.assign(target=y).head(10))

# ------------------------------------------------------ feature choice ----
left, right = st.columns(2)
with left:
    st.markdown("### Feature selection")
    manual = []
    if method == "Manual":
        manual = st.multiselect("Choose input features", list(X.columns), default=list(X.columns[:min(3, n_features)]),
                                key=f"manual_{dataset}")
        k = len(manual)
        if not manual:
            st.warning("Select at least one feature.")
    else:
        k = st.slider("Number of features", 2, n_features, min(3, n_features), key=f"k_{dataset}")
        st.caption(f"Suggested sizes for this dataset: {', '.join(map(str, DATASETS[dataset]['sizes']))}")
with right:
    st.markdown("### Protocol")
    st.write("- Stratified train/test split\n- Stratified 5-fold CV on the training split\n"
             "- Median imputation + standard scaling + selection inside a Pipeline\n"
             "- Test set used once, for the final evaluation")

run = st.button("🚀 Run Experiment", type="primary", disabled=(method == "Manual" and not manual))
signature = (dataset, classifier, method, k, tuple(manual), test_size, rs)

# ---------------------------------------------------------------- run ----
if run:
    with st.spinner("Running baseline, selection methods and the feature-count sweep..."):
        Xtr, Xte, ytr, yte = E.split(X, y, test_size, rs)
        baseline = E.evaluate(Xtr, Xte, ytr, yte, classifier, E.ALL, rs=rs)
        if method == "Manual":
            opt = E.evaluate(Xtr, Xte, ytr, yte, classifier, "Manual", rs=rs, manual=manual)
            points = [("All features", E.ALL, baseline), ("Manual", "Manual", opt)]
        else:
            opt = E.evaluate(Xtr, Xte, ytr, yte, classifier, method, k, rs)
            other = "RFE" if method == "SelectKBest" else "SelectKBest"
            other_res = E.evaluate(Xtr, Xte, ytr, yte, classifier, other, k, rs)
            points = [("All features", E.ALL, baseline), (method, method, opt), (other, other, other_res)]
        sizes = sorted({s for s in DATASETS[dataset]["sizes"] + [k] if 1 <= s <= n_features})
        sweep = E.cv_sweep(Xtr, ytr, classifier, sizes, rs)
        scores = E.f_scores(Xtr, ytr)

        figs = {
            "1_accuracy_vs_features": P.fig_acc_vs_features(sweep, dataset, classifier),
            "2_baseline_vs_optimized": P.fig_baseline_vs_opt(baseline, opt, dataset, classifier, method),
            "3_feature_scores": P.fig_feature_scores(scores, opt["chosen"], dataset, classifier, method),
            "4_confusion_matrix": P.fig_confusion(yte, opt["prediction"], class_names, dataset, classifier, method, opt["k"]),
            "5_accuracy_vs_runtime": P.fig_acc_vs_runtime(
                [(f"{lab} (k={r['k']})", m, r["train_time"], r["accuracy"]) for lab, m, r in points],
                dataset, classifier),
            "6_class_distribution": P.fig_class_distribution(ytr, class_names, dataset),
            "7_correlation_heatmap": P.fig_correlation(Xtr, dataset),
        }
        pngs = {}
        for name, fig in figs.items():
            buf = BytesIO(); fig.savefig(buf, format="png", dpi=150); pngs[name] = buf.getvalue()
        try:
            RESULTS_DIR.mkdir(exist_ok=True)
            stem = f"{dataset}_{classifier}_{method}".replace(" ", "_").replace("(", "").replace(")", "")
            for name, data in pngs.items():
                (RESULTS_DIR / f"{stem}_{name}.png").write_bytes(data)
        except Exception:
            pass
        st.session_state["res"] = dict(sig=signature, baseline=baseline, opt=opt, sweep=sweep, figs=figs, pngs=pngs,
                                       yte=yte, pred=opt["prediction"], class_names=class_names)

res = st.session_state.get("res")
if res is None:
    st.info("Pick a dataset, classifier, selection method and number of features, then click **Run Experiment**.")
    st.markdown("#### Included datasets")
    cols = st.columns(len(DATASETS))
    for col, (n, d) in zip(cols, DATASETS.items()):
        col.markdown(f"**{n}**"); col.caption(d["info"])
    st.stop()

if res["sig"] != signature:
    st.warning("Settings changed since the last run. Showing the previous results - click **Run Experiment** to update.")

d_name, d_clf, d_method, d_k, *_ = res["sig"]
baseline, opt = res["baseline"], res["opt"]
st.divider()
st.markdown("## Experiment results")
st.caption(f"{d_name} | {d_clf} | {d_method} | {opt['k']} features used | random_state={res['sig'][-1]}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test accuracy", f"{opt['accuracy']:.3f}", f"{opt['accuracy'] - baseline['accuracy']:+.3f} vs all features")
m2.metric("Test macro-F1", f"{opt['macro_f1']:.3f}", f"{opt['macro_f1'] - baseline['macro_f1']:+.3f} vs all features")
m3.metric("CV accuracy (mean ± SD)", f"{opt['cv_mean']:.3f} ± {opt['cv_std']:.3f}")
m4.metric("Training time", f"{opt['train_time'] * 1000:.0f} ms", f"baseline {baseline['train_time'] * 1000:.0f} ms", delta_color="off")

st.markdown("**Selected features:** " + ", ".join(f"`{f}`" for f in opt["chosen"]))
cmp = pd.DataFrame([
    {"Configuration": "All features (baseline)", "k": baseline["k"], "CV accuracy": f"{baseline['cv_mean']:.3f} ± {baseline['cv_std']:.3f}",
     "Test accuracy": baseline["accuracy"], "Macro-F1": baseline["macro_f1"], "Train time (s)": baseline["train_time"]},
    {"Configuration": f"{d_method}", "k": opt["k"], "CV accuracy": f"{opt['cv_mean']:.3f} ± {opt['cv_std']:.3f}",
     "Test accuracy": opt["accuracy"], "Macro-F1": opt["macro_f1"], "Train time (s)": opt["train_time"]},
])
st.dataframe(cmp.style.format({"Test accuracy": "{:.3f}", "Macro-F1": "{:.3f}", "Train time (s)": "{:.3f}"}))

zbuf = BytesIO()
with zipfile.ZipFile(zbuf, "w") as z:
    for n, data in res["pngs"].items():
        z.writestr(f"{n}.png", data)
st.download_button("⬇ Download all figures (PNG zip)", zbuf.getvalue(), file_name="optiselect_figures.zip", mime="application/zip")

tab1, tab2, tab3, tab4 = st.tabs(["Five mandatory figures", "Dataset insights", "Classification report",
                                  "All-datasets comparison table"])
F = res["figs"]
with tab1:
    a, b = st.columns(2)
    a.pyplot(F["1_accuracy_vs_features"]); b.pyplot(F["2_baseline_vs_optimized"])
    a, b = st.columns(2)
    a.pyplot(F["3_feature_scores"]); b.pyplot(F["4_confusion_matrix"])
    st.pyplot(F["5_accuracy_vs_runtime"])
    st.caption("Explain in your report: where accuracy peaks, whether fewer features preserve performance, which inputs matter most, "
               "which classes are confused, and the speed vs accuracy trade-off.")
with tab2:
    a, b = st.columns(2)
    a.pyplot(F["6_class_distribution"]); b.pyplot(F["7_correlation_heatmap"])
    st.markdown("#### Cross-validated accuracy for each subset size")
    st.dataframe(res["sweep"].style.format({"CV accuracy": "{:.3f}", "CV SD": "{:.3f}"}))
with tab3:
    rep = classification_report(res["yte"], res["pred"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3))
with tab4:
    st.write(f"Runs all five datasets with **{classifier}**. For each dataset, the best k for SelectKBest and RFE is picked "
             "by CV on the training split (from the PDF's suggested sizes), then evaluated once on the test split.")
    if st.button("Generate full table (takes about a minute)"):
        with st.spinner("Running all datasets..."):
            try:
                tbl = E.final_table(load_dataset, {n: d["sizes"] for n, d in DATASETS.items()}, classifier, test_size, rs)
                st.session_state["final_table"] = (classifier, tbl)
            except Exception as exc:
                st.error(f"Could not finish: {exc}")
    ft = st.session_state.get("final_table")
    if ft:
        st.caption(f"Classifier: {ft[0]}")
        st.dataframe(ft[1])
        st.download_button("⬇ Download table (CSV)", ft[1].to_csv(index=False).encode(), file_name="optiselect_final_table.csv", mime="text/csv")

plt.close("all")
st.caption("OptiSelect Group 1 | all numbers above are measured live; the PDF mockup values are not used.")
