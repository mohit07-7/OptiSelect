"""OptiSelect - Smart Feature Selection dashboard.  Run: streamlit run app.py"""
import matplotlib.pyplot as plt
import streamlit as st

import experiments as ex
import plots

def show(col, fig):
    col.pyplot(fig)
    plt.close(fig)


st.set_page_config(page_title="OptiSelect", page_icon="🎯", layout="wide")
st.title("🎯 OptiSelect - Smart Feature Selection")
st.caption("Compare a classifier on ALL features vs automatically selected subsets "
           "(SelectKBest filter, RFE wrapper). Selection happens inside a Pipeline, on training data only.")


@st.cache_data(show_spinner="Loading dataset...")
def cached_load(name):
    return ex.load_dataset(name)


@st.cache_data(show_spinner="Running experiment (cross-validation sweep)...")
def cached_run(*args):
    return ex.run_experiment(*args)


with st.sidebar:
    st.header("Controls")
    dataset = st.selectbox("Dataset", ex.DATASETS)
    clf = st.selectbox("Classifier", ex.CLASSIFIERS)
    method = st.selectbox("Selection method", ex.METHODS)
    try:
        n_feat = cached_load(dataset)[0].shape[1]
    except Exception as e:
        st.error(str(e)); st.stop()
    auto = st.checkbox("Pick k automatically (best CV accuracy)")
    k = st.slider("Number of features", 2, n_feat, min(ex.DEFAULT_K[dataset], n_feat), disabled=auto)
    run = st.button("Run Experiment", type="primary", width="stretch")

X, y, classes = cached_load(dataset)
tab_data, tab_res, tab_fig, tab_all = st.tabs(["📊 Dataset", "🧪 Results", "📈 Figures", "📋 All datasets table"])

with tab_data:
    st.subheader(f"{dataset}")
    st.write(ex.DESCRIPTIONS[dataset])
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", X.shape[0]); c2.metric("Features", X.shape[1]); c3.metric("Missing values", int(X.isna().sum().sum()))
    st.dataframe(X.assign(target=y).head(10), width="stretch")
    base = dict(dataset=dataset, X=X, y=y, classes=classes, n=X.shape[1])
    a, b = st.columns([1, 1.4])
    show(a, plots.fig_class_dist(base)); show(b, plots.fig_corr(base))

if run:
    st.session_state["r"] = cached_run(dataset, clf, method, k, auto)
r = st.session_state.get("r")

with tab_res:
    if not r:
        st.info("Choose settings in the sidebar and click **Run Experiment**.")
    else:
        d = r["details"][r["method"]]; b = r["details"]["All features"]
        st.subheader(f"{r['dataset']} · {r['clf']} · {r['method']} · k={r['k']}/{r['n']}")
        c = st.columns(4)
        c[0].metric("Test accuracy", f"{d['test_acc']:.3f}", f"{d['test_acc']-b['test_acc']:+.3f} vs all features")
        c[1].metric("Macro-F1", f"{d['macro_f1']:.3f}", f"{d['macro_f1']-b['macro_f1']:+.3f} vs all features")
        c[2].metric("CV accuracy", f"{d['cv_mean']:.3f} ± {d['cv_std']:.3f}")
        c[3].metric("Training time", f"{d['train_time']:.3f}s")
        st.success("**Selected features:** " + ", ".join(d["features"]))
        st.caption(f"Train {r['n_train']} / test {r['n_test']} (stratified 80/20, random_state=42). "
                   "k was chosen with CV on the training split; the test set was used once.")
        st.dataframe(r["table"], width="stretch", hide_index=True)

with tab_fig:
    if not r:
        st.info("Run an experiment first.")
    else:
        figs = [plots.fig_acc_vs_k, plots.fig_baseline, plots.fig_scores, plots.fig_cm, plots.fig_runtime]
        cols = st.columns(2)
        for i, f in enumerate(figs):
            show(cols[i % 2], f(r))
        st.caption("PNG copies of all five figures are saved in the results/ folder.")

with tab_all:
    st.write("Runs every dataset with the sidebar classifier at each dataset's default k "
             "(Iris 2, Wine 5, Breast Cancer 10, Heart 8, Wine Quality 6) - for the report's final table.")
    if st.button("Generate full comparison table"):
        bar = st.progress(0.0)
        t = ex.full_table(clf, lambda p, m: bar.progress(p, text=m))
        t.to_csv(plots.RESULTS / "results_table.csv", index=False)
        st.session_state["full"] = t
    if "full" in st.session_state:
        st.dataframe(st.session_state["full"], width="stretch", hide_index=True)
        st.download_button("Download CSV", st.session_state["full"].to_csv(index=False), "results_table.csv")
