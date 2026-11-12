import time
import urllib.request
from io import StringIO, BytesIO
import zipfile
import requests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="OptiSelect AI", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .stApp {background: #f6f8fc;}
    [data-testid="stSidebar"] {background: #101a32;}
    [data-testid="stSidebar"] * {color: #eef3ff !important;}
    .hero {padding: 1.4rem 1.7rem; border-radius: 18px; color: white;
           background: linear-gradient(115deg,#14213d,#234e83 65%,#3976a8); margin-bottom: 1rem;}
    .hero h1 {font-size:2.25rem; margin:0;}
    .hero p {color:#dce8fa; margin:.4rem 0 0 0;}
    div[data-testid="stMetric"] {background:white; padding:15px; border:1px solid #e6ebf3; border-radius:14px;}
    .section {font-weight:700; color:#1c3154; font-size:1.15rem; margin-top:.5rem;}
    .small-note {color:#64748b; font-size:.88rem;}
</style>
<div class="hero">
<h1>🧬 OptiSelect AI</h1>
<p>Multi-Dataset Smart Feature Selection · Compare full-feature baselines with optimized models</p>
</div>
""", unsafe_allow_html=True)

DATASET_INFO = {
    "Iris": ("4 numeric features", "3 species", "scikit-learn built-in"),
    "Wine": ("13 numeric features", "3 cultivars", "scikit-learn built-in"),
    "Breast Cancer Wisconsin": ("30 numeric features", "2 classes", "scikit-learn built-in"),
    "Heart Disease (UCI Cleveland)": ("13 numeric features", "2 classes", "UCI Machine Learning Repository"),
    "Wine Quality (Red + White)": ("11 numeric features", "2 classes: quality ≥ 6", "UCI Machine Learning Repository"),
}

def _download_uci_zip(url, timeout=25):
    """Download a UCI dataset archive with a clear network error."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return zipfile.ZipFile(BytesIO(response.content))

def _read_heart_cleveland():
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
            "exang","oldpeak","slope","ca","thal","num"]
    # Current UCI static download endpoint; legacy endpoint retained as fallback.
    sources = [
        ("zip", "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"),
        ("csv", "https://raw.githubusercontent.com/StatQuest/heart-disease/master/processed.cleveland.data"),
        ("csv", "https://raw.githubusercontent.com/StatQuest/logistic_regression_demo/master/processed.cleveland.data"),
    ]
    errors = []
    for kind, url in sources:
        try:
            if kind == "zip":
                archive = _download_uci_zip(url)
                member = next(n for n in archive.namelist()
                              if n.lower().endswith("processed.cleveland.data"))
                raw = archive.read(member)
                df = pd.read_csv(BytesIO(raw), header=None, names=cols, na_values="?")
            else:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text), header=None, names=cols, na_values="?")
            if df.shape[1] != 14 or len(df) < 200:
                raise ValueError("Downloaded file is not the 303-row Cleveland dataset.")
            return df
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Unable to download the UCI Cleveland Heart Disease dataset. "
                       "Check the app's internet access. " + " | ".join(errors))

def _read_wine_quality():
    # Official UCI archive contains both semicolon-separated CSV files.
    sources = [
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    ]
    errors = []
    for url in sources:
        try:
            archive = _download_uci_zip(url)
            names = {n.rsplit("/", 1)[-1].lower(): n for n in archive.namelist()}
            red_name = names["winequality-red.csv"]
            white_name = names["winequality-white.csv"]
            red = pd.read_csv(BytesIO(archive.read(red_name)), sep=";")
            white = pd.read_csv(BytesIO(archive.read(white_name)), sep=";")
            if len(red) != 1599 or len(white) != 4898:
                raise ValueError("Unexpected Wine Quality file dimensions.")
            red["wine_type"] = "red"
            white["wine_type"] = "white"
            return pd.concat([red, white], ignore_index=True)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    # Fallback to the stable legacy UCI CSV endpoints.
    try:
        red_response = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            timeout=25)
        white_response = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
            timeout=25)
        red_response.raise_for_status()
        white_response.raise_for_status()
        red = pd.read_csv(StringIO(red_response.text), sep=";")
        white = pd.read_csv(StringIO(white_response.text), sep=";")
        red["wine_type"] = "red"
        white["wine_type"] = "white"
        return pd.concat([red, white], ignore_index=True)
    except Exception as exc:
        errors.append(f"legacy UCI CSVs: {exc}")
    raise RuntimeError("Unable to download UCI Wine Quality (red + white). "
                       "Check the app's internet access. " + " | ".join(errors))

@st.cache_data(show_spinner=False, ttl=86400)
def load_dataset(name):
    if name == "Iris":
        d = load_iris(as_frame=True)
        X, y = d.data.copy(), pd.Series(d.target, name="target")
        return X, y, list(d.target_names), d.frame.head(8)
    if name == "Wine":
        d = load_wine(as_frame=True)
        X, y = d.data.copy(), pd.Series(d.target, name="target")
        return X, y, list(d.target_names), d.frame.head(8)
    if name == "Breast Cancer Wisconsin":
        d = load_breast_cancer(as_frame=True)
        X, y = d.data.copy(), pd.Series(d.target, name="target")
        return X, y, list(d.target_names), d.frame.head(8)
    if name == "Heart Disease (UCI Cleveland)":
        df = _read_heart_cleveland()
        y = (pd.to_numeric(df["num"], errors="coerce") > 0).astype("int64")
        X = df.drop(columns=["num"]).apply(pd.to_numeric, errors="coerce")
        return X, y, ["No disease", "Disease"], df.head(8)
    if name == "Wine Quality (Red + White)":
        df = _read_wine_quality()
        y = (pd.to_numeric(df["quality"], errors="coerce") >= 6).astype("int64")
        # The 11 physicochemical measurements are the model inputs.
        # wine_type is retained only for preview/description, not as a feature.
        feature_cols = [c for c in df.columns if c not in ("quality", "wine_type")]
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        preview = df[feature_cols + ["quality", "wine_type"]].head(8)
        return X, y, ["Quality < 6", "Quality ≥ 6"], preview
    raise ValueError(f"Unknown dataset: {name}")

def make_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, class_weight="balanced")
    return RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1)

def build_pipeline(model_name, method, k, selected_names, all_features=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "Logistic Regression" or method == "RFE":
        steps.append(("scaler", StandardScaler()))
    if not all_features and method == "SelectKBest":
        steps.append(("selector", SelectKBest(score_func=f_classif, k=k)))
    elif not all_features and method == "RFE":
        steps.append(("selector", RFE(estimator=LogisticRegression(max_iter=3000), n_features_to_select=k)))
    steps.append(("model", make_model(model_name)))
    return Pipeline(steps)

def evaluate(X_train, X_test, y_train, y_test, model_name, method, k, selected_names, all_features=False, cv=None):
    pipe = build_pipeline(model_name, method, k, selected_names, all_features)
    start = time.perf_counter()
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    pred = pipe.predict(X_test)
    if all_features:
        chosen = list(X_train.columns)
    elif method == "Manual":
        chosen = selected_names
    else:
        chosen = list(X_train.columns[pipe.named_steps["selector"].get_support()])
    return {
        "pipeline": pipe, "cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "runtime": float(fit_time), "prediction": pred, "chosen": chosen
    }

# Sidebar controls
st.sidebar.markdown("## EXPERIMENT CONTROLS")
dataset_name = st.sidebar.selectbox("Dataset", list(DATASET_INFO.keys()))
classifier = st.sidebar.selectbox("Classifier", ["Logistic Regression", "Decision Tree", "Random Forest"])
method = st.sidebar.selectbox("Feature selection method", ["SelectKBest", "RFE", "Manual"])
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
test_size = st.sidebar.slider("Test set size", 0.15, 0.35, 0.20, 0.05)
st.sidebar.caption("Stratified split · preprocessing and selection are fitted only on training data.")

with st.spinner("Loading dataset…"):
    try:
        X, y, class_names, preview = load_dataset(dataset_name)
    except Exception as e:
        st.error(f"Could not load {dataset_name}. This dataset needs an internet connection on first load. Details: {e}")
        st.stop()

# Clean up
X = X.replace([np.inf, -np.inf], np.nan)
X = X.apply(pd.to_numeric, errors="coerce")
valid = y.notna()
X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
n_features = X.shape[1]
max_k = n_features
default_k = min(3, n_features)

st.markdown("### Dataset overview")
a,b,c,d = st.columns(4)
a.metric("Rows / samples", f"{len(X):,}")
b.metric("Input features", n_features)
c.metric("Classes", y.nunique())
d.metric("Missing cells", f"{int(X.isna().sum().sum()):,}")
st.caption(f"Source: {DATASET_INFO[dataset_name][2]} · {DATASET_INFO[dataset_name][0]} · {DATASET_INFO[dataset_name][1]}")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(X.assign(target=y).head(12), use_container_width=True)
    st.download_button("Download current dataset CSV", X.assign(target=y).to_csv(index=False).encode("utf-8"),
                       file_name=dataset_name.lower().replace(" ","_")+"_data.csv", mime="text/csv")

left, right = st.columns([1, 1])
with left:
    st.markdown("### Feature selection")
    if method == "Manual":
        chosen_manual = st.multiselect("Choose input features", list(X.columns), default=list(X.columns[:default_k]))
        k = len(chosen_manual)
        if not chosen_manual:
            st.warning("Select at least one feature to run an experiment.")
    else:
        k = st.slider("Number of selected features", min_value=1, max_value=max_k, value=default_k)
        chosen_manual = []
    st.markdown(f"<div class='small-note'>Selection method: {method} · Requested feature count: {k}</div>", unsafe_allow_html=True)
with right:
    st.markdown("### Experiment protocol")
    st.write("• Stratified train/test split")
    st.write("• Stratified cross-validation (up to 5 folds)")
    st.write("• Median imputation inside the pipeline")
    st.write("• Feature selection fitted on training folds only")
    st.write("• Fixed random state for reproducibility")

run = st.button("🚀 Run Experiment", type="primary", use_container_width=True, disabled=(method=="Manual" and not chosen_manual))

if run:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=int(random_state))
    selected_names = chosen_manual if method == "Manual" else []
    if method == "Manual":
        k = len(chosen_manual)
        X_opt_train, X_opt_test = X_train[chosen_manual], X_test[chosen_manual]
    else:
        X_opt_train, X_opt_test = X_train, X_test
    min_class_count = int(y_train.value_counts().min())
    if min_class_count < 2:
        st.error("Not enough samples in the smallest class for stratified cross-validation. Increase the training set size or use a different split.")
        st.stop()
    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    with st.spinner("Training baseline and optimized models…"):
        baseline = evaluate(X_train, X_test, y_train, y_test, classifier, method, n_features, list(X.columns), True, cv)
        optimized = evaluate(X_opt_train, X_opt_test, y_train, y_test, classifier, method, k, selected_names, False, cv)
    st.session_state["last_result"] = {
        "dataset": dataset_name, "classifier": classifier, "method": method, "k": k,
        "X": X, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "baseline": baseline, "optimized": optimized, "class_names": class_names,
        "random_state": int(random_state), "test_size": test_size,
        "manual_features": list(chosen_manual) if method == "Manual" else []
    }

if "last_result" not in st.session_state:
    st.info("Choose your dataset, classifier and selection settings, then click **Run Experiment** to generate measured results.")
    st.markdown("#### Included datasets")
    cols = st.columns(5)
    for col, name in zip(cols, DATASET_INFO):
        col.markdown(f"**{name}**")
        col.caption(DATASET_INFO[name][0])
    st.stop()

r = st.session_state["last_result"]
current_signature = (dataset_name, classifier, method, int(random_state), float(test_size),
                     int(k), tuple(chosen_manual) if method == "Manual" else ())
saved_signature = (r["dataset"], r["classifier"], r["method"], r["random_state"],
                   float(r["test_size"]), int(r["k"]),
                   tuple(r.get("manual_features", ())))
if current_signature != saved_signature:
    st.info("Settings changed. Click **Run Experiment** to calculate results for the selected dataset and settings.")
    st.stop()
baseline, opt = r["baseline"], r["optimized"]
st.divider()
st.markdown("## Experiment results")
st.caption(f"{r['dataset']} · {r['classifier']} · {r['method']} · {r['k']} selected features · random_state={r['random_state']}")
m1,m2,m3,m4 = st.columns(4)
m1.metric("Optimized test accuracy", f"{opt['accuracy']:.3f}", f"{(opt['accuracy']-baseline['accuracy']):+.3f} vs baseline")
m2.metric("Optimized macro-F1", f"{opt['macro_f1']:.3f}")
m3.metric("CV accuracy (mean ± SD)", f"{opt['cv_mean']:.3f} ± {opt['cv_std']:.3f}")
m4.metric("Training + CV time", f"{opt['runtime']:.2f}s")

st.markdown("### Selected features")
st.write(", ".join(opt["chosen"]))
st.markdown("### Baseline vs optimized")
comparison = pd.DataFrame([
    {"Configuration":"All features (baseline)", "Features":len(r["X"].columns), "CV accuracy":baseline["cv_mean"], "Test accuracy":baseline["accuracy"], "Macro-F1":baseline["macro_f1"], "Runtime (s)":baseline["runtime"]},
    {"Configuration":"Selected features", "Features":len(opt["chosen"]), "CV accuracy":opt["cv_mean"], "Test accuracy":opt["accuracy"], "Macro-F1":opt["macro_f1"], "Runtime (s)":opt["runtime"]},
])
st.dataframe(comparison.style.format({"CV accuracy":"{:.3f}","Test accuracy":"{:.3f}","Macro-F1":"{:.3f}","Runtime (s)":"{:.2f}"}), use_container_width=True)
st.download_button("⬇ Download comparison results (CSV)", comparison.to_csv(index=False).encode("utf-8"),
                   file_name="optiselect_comparison.csv", mime="text/csv")

# Feature score computation on training data only
score_X = r["X_train"].copy()
imputer = SimpleImputer(strategy="median")
score_arr = imputer.fit_transform(score_X)
try:
    scores, _ = f_classif(score_arr, r["y_train"])
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
except Exception:
    scores = np.zeros(score_arr.shape[1])
score_df = pd.DataFrame({"Feature":list(r["X_train"].columns), "F-score":scores}).sort_values("F-score", ascending=False)

# Accuracy vs feature count sweep, measured with CV on training split only
st.markdown("### Five mandatory visualizations")
with st.spinner("Calculating feature-count sweep for the accuracy curve…"):
    counts = sorted(set([1, min(2,n_features), min(3,n_features), min(5,n_features), min(8,n_features), max(1,r["k"]), n_features]))
    sweep_rows = []
    for count in counts:
        if count == n_features:
            sweep_rows.append({"Features":count, "Method":"All features", "CV accuracy":baseline["cv_mean"]})
        else:
            sweep_method = r["method"] if r["method"] in ("SelectKBest","RFE") else "SelectKBest"
            sweep_pipe = build_pipeline(r["classifier"], sweep_method, count, [], False)
            vals = cross_val_score(sweep_pipe, r["X_train"], r["y_train"],
                                   cv=StratifiedKFold(n_splits=min(5, int(r["y_train"].value_counts().min())),
                                                      shuffle=True, random_state=r["random_state"]),
                                   scoring="accuracy")
            sweep_rows.append({"Features":count, "Method":sweep_method, "CV accuracy":float(vals.mean())})
    sweep_df = pd.DataFrame(sweep_rows)

tab1, tab2 = st.tabs(["Performance charts", "Dataset insights"])
with tab1:
    fig, ax = plt.subplots(figsize=(8,4))
    for label, grp in sweep_df.groupby("Method"):
        ax.plot(grp["Features"], grp["CV accuracy"], marker="o", label=label)
    ax.set(title=f"{r['dataset']} · {r['classifier']} — Accuracy vs number of features", xlabel="Number of features", ylabel="5-fold CV accuracy")
    ax.set_ylim(0,1.05); ax.grid(alpha=.25); ax.legend()
    st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    metrics = ["accuracy","macro_f1"]
    xloc = np.arange(2); width=.34
    ax.bar(xloc-width/2, [baseline[m] for m in metrics], width, label="All features")
    ax.bar(xloc+width/2, [opt[m] for m in metrics], width, label="Selected features")
    ax.set_xticks(xloc, ["Accuracy","Macro-F1"]); ax.set_ylim(0,1.05)
    ax.set_title(f"{r['dataset']} · {r['classifier']} — Baseline vs optimized")
    ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=.2)
    st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    top = score_df.head(min(15, len(score_df))).sort_values("F-score")
    ax.barh(top["Feature"], top["F-score"])
    ax.set(title=f"{r['dataset']} · Training-set feature scores", xlabel="ANOVA F-score", ylabel="Feature")
    ax.grid(axis="x", alpha=.2); st.pyplot(fig); plt.close(fig)

    labels = np.unique(r["y_test"])
    cm = confusion_matrix(r["y_test"], opt["prediction"], labels=labels)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(x) for x in labels], yticklabels=[str(x) for x in labels], ax=ax)
    ax.set(title=f"{r['dataset']} · {r['classifier']} — Confusion matrix", xlabel="Predicted class", ylabel="Actual class")
    st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.scatter([baseline["runtime"], opt["runtime"]], [baseline["accuracy"], opt["accuracy"]], s=120)
    for label, xval, yval in [("All features",baseline["runtime"],baseline["accuracy"]),("Selected",opt["runtime"],opt["accuracy"])]:
        ax.annotate(label, (xval,yval), xytext=(7,7), textcoords="offset points")
    ax.set(title=f"{r['dataset']} · {r['classifier']} — Accuracy vs runtime", xlabel="CV + fit runtime (seconds)", ylabel="Test accuracy")
    ax.set_ylim(0,1.05); ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)

with tab2:
    st.markdown("#### Class distribution")
    counts_df = r["y_train"].value_counts().sort_index().rename_axis("Class").reset_index(name="Training samples")
    fig, ax = plt.subplots(figsize=(7,3.5))
    ax.bar(counts_df["Class"].astype(str), counts_df["Training samples"])
    ax.set(title=f"{r['dataset']} — Training class distribution", xlabel="Class", ylabel="Samples")
    st.pyplot(fig); plt.close(fig)
    st.markdown("#### Feature correlation heatmap")
    corr = r["X_train"].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"{r['dataset']} — Training feature correlation")
    st.pyplot(fig); plt.close(fig)
    st.markdown("#### Feature scores (training data only)")
    st.dataframe(score_df.sort_values("F-score", ascending=False), use_container_width=True)

st.markdown("### Classification report")
report = classification_report(r["y_test"], opt["prediction"], output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.markdown("---")
st.caption("OptiSelect Group 1 · Reproducible experiments use random_state=42 by default. Results shown above are generated live; reference mockup values are not used.")
