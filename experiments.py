"""Model evaluation for OptiSelect (no Streamlit code here).

Leakage prevention: imputation, scaling and feature selection live INSIDE one
Pipeline, so they are fitted on training folds only. The test set is used once
for the final evaluation of a chosen configuration.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = ["Logistic Regression", "Decision Tree", "Random Forest"]
METHODS = ["SelectKBest", "RFE", "Manual"]
ALL = "All features"


def make_model(name, rs=42):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=rs)
    return RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1)


def build_pipeline(model_name, method, k, rs=42):
    steps = [("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())]
    if method == "SelectKBest":
        steps.append(("selector", SelectKBest(score_func=f_classif, k=k)))
    elif method == "RFE":
        steps.append(("selector", RFE(estimator=LogisticRegression(max_iter=3000), n_features_to_select=k)))
    steps.append(("model", make_model(model_name, rs)))
    return Pipeline(steps)


def split(X, y, test_size=0.2, rs=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)


def make_cv(y_train, rs=42):
    n = max(2, min(5, int(y_train.value_counts().min())))
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=rs)


def evaluate(X_train, X_test, y_train, y_test, model_name, method, k=None, rs=42, manual=None):
    """method: 'All features' | 'SelectKBest' | 'RFE' | 'Manual'. Returns a dict of measured values."""
    if method == "Manual":
        Xtr, Xte, pipe_method = X_train[manual], X_test[manual], ALL
    else:
        Xtr, Xte, pipe_method = X_train, X_test, method
    pipe = build_pipeline(model_name, pipe_method, k, rs)
    cv = make_cv(y_train, rs)
    cv_scores = cross_val_score(pipe, Xtr, y_train, cv=cv, scoring="accuracy")
    t0 = time.perf_counter()
    pipe.fit(Xtr, y_train)
    train_time = time.perf_counter() - t0          # fit time of the final pipeline (selection + model)
    pred = pipe.predict(Xte)
    if method == "Manual":
        chosen = list(manual)
    elif method == ALL:
        chosen = list(X_train.columns)
    else:
        chosen = list(X_train.columns[pipe.named_steps["selector"].get_support()])
    return {
        "method": method, "k": len(chosen), "chosen": chosen,
        "cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "train_time": float(train_time), "prediction": pred,
    }


def cv_sweep(X_train, y_train, model_name, sizes, rs=42):
    """Mean CV accuracy for each subset size and each method (training split only)."""
    cv = make_cv(y_train, rs)
    n = X_train.shape[1]
    rows = []
    for method in ("SelectKBest", "RFE"):
        for k in sizes:
            if k >= n:
                pipe = build_pipeline(model_name, ALL, None, rs)
            else:
                pipe = build_pipeline(model_name, method, k, rs)
            s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
            rows.append({"Method": method, "Features": k, "CV accuracy": s.mean(), "CV SD": s.std()})
    return pd.DataFrame(rows)


def f_scores(X_train, y_train):
    arr = SimpleImputer(strategy="median").fit_transform(X_train)
    sc, _ = f_classif(arr, y_train)
    sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({"Feature": X_train.columns, "F-score": sc}).sort_values("F-score", ascending=False)


def final_table(datasets_loader, dataset_sizes, model_name, test_size=0.2, rs=42):
    """Section-9 table: for every dataset -> All features, best SelectKBest k, best RFE k (k picked by CV on train,
    then evaluated once on the held-out test split)."""
    rows = []
    for name, sizes in dataset_sizes.items():
        X, y, _, _ = datasets_loader(name)
        X_tr, X_te, y_tr, y_te = split(X, y, test_size, rs)
        n = X.shape[1]
        base = evaluate(X_tr, X_te, y_tr, y_te, model_name, ALL, rs=rs)
        rows.append({"Dataset": name, "Method": ALL, "k / total": f"{n}/{n}",
                     "CV accuracy": f"{base['cv_mean']:.3f} ± {base['cv_std']:.3f}",
                     "Test accuracy": round(base["accuracy"], 3), "Macro-F1": round(base["macro_f1"], 3),
                     "Train time (s)": round(base["train_time"], 3)})
        sweep = cv_sweep(X_tr, y_tr, model_name, [s for s in sizes if s < n], rs)
        for method in ("SelectKBest", "RFE"):
            sub = sweep[sweep["Method"] == method]
            if sub.empty:
                continue
            best_k = int(sub.loc[sub["CV accuracy"].idxmax(), "Features"])
            r = evaluate(X_tr, X_te, y_tr, y_te, model_name, method, best_k, rs)
            rows.append({"Dataset": name, "Method": method, "k / total": f"{best_k}/{n}",
                         "CV accuracy": f"{r['cv_mean']:.3f} ± {r['cv_std']:.3f}",
                         "Test accuracy": round(r["accuracy"], 3), "Macro-F1": round(r["macro_f1"], 3),
                         "Train time (s)": round(r["train_time"], 3)})
    return pd.DataFrame(rows)
