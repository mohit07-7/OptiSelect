"""OptiSelect - dataset loaders and experiment engine (no Streamlit code here)."""
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RS = 42
DATA_DIR = Path(__file__).parent / "data"
DATASETS = ["Iris", "Wine", "Breast Cancer", "Heart Disease", "Wine Quality"]
DEFAULT_K = {"Iris": 2, "Wine": 5, "Breast Cancer": 10, "Heart Disease": 8, "Wine Quality": 6}
CLASSIFIERS = ["Logistic Regression", "Decision Tree", "Random Forest"]
METHODS = ["SelectKBest", "RFE"]

HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
WQ_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-{}.csv"
HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
              "exang", "oldpeak", "slope", "ca", "thal", "target"]

DESCRIPTIONS = {
    "Iris": "150 flowers, 4 measurements, 3 species (scikit-learn).",
    "Wine": "178 wines, 13 chemical measurements, 3 cultivars (scikit-learn).",
    "Breast Cancer": "569 tumours, 30 measurements, malignant vs benign (scikit-learn).",
    "Heart Disease": "UCI Cleveland, 303 patients, 13 features. Target: disease present (num > 0). "
                     "Missing values ('?') are median-imputed inside the pipeline.",
    "Wine Quality": "UCI red + white vinho verde, ~6,500 wines, 11 features. "
                    "Target: quality >= 6 is 'good' (1), otherwise 0.",
}


def _fetch(url, fname):
    """Return a local path for a UCI file, downloading it once into data/."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / fname
    if not path.exists():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                path.write_bytes(r.read())
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Could not download {url} ({e}). Download it manually and save it as "
                f"{path} (see README.md).") from e
    return path


def load_dataset(name):
    """Return X (DataFrame), y (Series), class_names (list[str])."""
    if name in ("Iris", "Wine", "Breast Cancer"):
        d = {"Iris": load_iris, "Wine": load_wine, "Breast Cancer": load_breast_cancer}[name]()
        X = pd.DataFrame(d.data, columns=[c.replace(" (cm)", "") for c in d.feature_names])
        return X, pd.Series(d.target, name="target"), [str(c) for c in d.target_names]
    if name == "Heart Disease":
        p = _fetch(HEART_URL, "processed.cleveland.data")
        df = pd.read_csv(p, header=None, names=HEART_COLS, na_values="?")
        y = (df.pop("target") > 0).astype(int)
        return df, y.rename("target"), ["no disease", "disease"]
    if name == "Wine Quality":
        frames = [pd.read_csv(_fetch(WQ_URL.format(c), f"winequality-{c}.csv"), sep=";")
                  for c in ("red", "white")]
        df = pd.concat(frames, ignore_index=True)
        y = (df.pop("quality") >= 6).astype(int)
        return df, y.rename("target"), ["quality < 6", "quality >= 6"]
    raise ValueError(name)


def make_classifier(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RS)
    return RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=-1)


def make_pipeline(clf, method, k):
    """method None/'All features' => baseline with every feature."""
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if method == "SelectKBest":
        steps.append(("selector", SelectKBest(f_classif, k=k)))
    elif method == "RFE":
        steps.append(("selector", RFE(LogisticRegression(max_iter=2000), n_features_to_select=k)))
    steps.append(("model", make_classifier(clf)))
    return Pipeline(steps)


def sweep_ks(n, k):
    base = np.unique(np.linspace(2, n, min(n - 1, 6)).round().astype(int))
    return sorted(set(base.tolist()) | {k, n})


def _final_eval(pipe, Xtr, ytr, Xte, yte, cols):
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    pred = pipe.predict(Xte)
    out = dict(test_acc=accuracy_score(yte, pred),
               macro_f1=f1_score(yte, pred, average="macro"),
               train_time=fit_time, cm=confusion_matrix(yte, pred))
    if "selector" in pipe.named_steps:
        sel = pipe.named_steps["selector"]
        mask = sel.get_support()
        score = np.nan_to_num(sel.scores_) if hasattr(sel, "scores_") else 1.0 / sel.ranking_
        out["features"] = list(cols[mask])
        out["scores"] = pd.DataFrame({"feature": cols, "score": score, "selected": mask})
    else:
        out["features"] = list(cols)
        out["scores"] = None
    return out


def run_experiment(dataset, clf, method, k, auto_k=False):
    X, y, classes = load_dataset(dataset)
    n = X.shape[1]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    ks = sweep_ks(n, k)
    rows = []
    for m in METHODS:
        for kk in ks:
            s = cross_val_score(make_pipeline(clf, m, kk), Xtr, ytr, cv=cv, scoring="accuracy")
            rows.append(dict(method=m, k=kk, cv_mean=s.mean(), cv_std=s.std()))
    sweep = pd.DataFrame(rows)
    if auto_k:  # best mean CV accuracy, ties -> fewer features (CV only, never the test set)
        sub = sweep[sweep.method == method].sort_values(["cv_mean", "k"], ascending=[False, True])
        k = int(sub.iloc[0].k)

    def cv_of(m, kk):
        r = sweep[(sweep.method == m) & (sweep.k == kk)].iloc[0]
        return r.cv_mean, r.cv_std

    details = {}
    for label, m, kk in [("All features", None, n), ("SelectKBest", "SelectKBest", k), ("RFE", "RFE", k)]:
        if kk not in ks:  # only possible if auto_k changed nothing; kept for safety
            s = cross_val_score(make_pipeline(clf, m, kk), Xtr, ytr, cv=cv, scoring="accuracy")
            cvm, cvs = s.mean(), s.std()
        else:
            cvm, cvs = cv_of(m or "SelectKBest", kk)  # k=n selection == all features
        d = _final_eval(make_pipeline(clf, m, kk), Xtr, ytr, Xte, yte, X.columns)
        d.update(method=label, k=kk, n=n, cv_mean=cvm, cv_std=cvs)
        details[label] = d
    table = pd.DataFrame([dict(Dataset=dataset, Classifier=clf, Method=v["method"],
                               **{"k / total": f"{v['k']}/{n}",
                                  "CV accuracy (mean ± SD)": f"{v['cv_mean']:.3f} ± {v['cv_std']:.3f}",
                                  "Test accuracy": round(v["test_acc"], 4),
                                  "Macro-F1": round(v["macro_f1"], 4),
                                  "Train time (s)": round(v["train_time"], 4)})
                          for v in details.values()])
    return dict(dataset=dataset, clf=clf, method=method, k=k, n=n, classes=classes,
                X=X, y=y, sweep=sweep, details=details, table=table,
                n_train=len(Xtr), n_test=len(Xte))


def full_table(clf, progress=None):
    parts = []
    for i, ds in enumerate(DATASETS):
        if progress:
            progress(i / len(DATASETS), f"Running {ds}...")
        parts.append(run_experiment(ds, clf, "SelectKBest", DEFAULT_K[ds])["table"])
    if progress:
        progress(1.0, "Done")
    return pd.concat(parts, ignore_index=True)
