"""Dataset loaders for OptiSelect.

Iris, Wine and Breast Cancer ship with scikit-learn (always work offline).
Heart Disease (UCI Cleveland) and Wine Quality (UCI) are loaded in this order:
  1. local files in data/            (most reliable - commit them to GitHub)
  2. download from UCI / mirrors     (saved into data/ on success)
  3. OpenML via scikit-learn         (last-resort fallback)
Run `python download_data.py` once on your PC to create the local files.
"""
from io import BytesIO, StringIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

DATA_DIR = Path(__file__).resolve().parent / "data"
HEART_FILE = DATA_DIR / "heart_cleveland.csv"
WINE_RED_FILE = DATA_DIR / "winequality-red.csv"
WINE_WHITE_FILE = DATA_DIR / "winequality-white.csv"

HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Subset sizes suggested in the project PDF
DATASETS = {
    "Iris":                     {"sizes": [2, 3, 4],        "info": "4 features · 3 species · scikit-learn"},
    "Wine":                     {"sizes": [5, 7, 10, 13],   "info": "13 features · 3 cultivars · scikit-learn"},
    "Breast Cancer Wisconsin":  {"sizes": [5, 10, 15, 30],  "info": "30 features · 2 classes · scikit-learn"},
    "Heart Disease (UCI)":      {"sizes": [5, 8, 13],       "info": "13 features · disease yes/no · UCI Cleveland (missing values imputed)"},
    "Wine Quality":             {"sizes": [4, 6, 8, 11],    "info": "11 features · target: quality >= 6 · UCI (red + white)"},
}

UA = {"User-Agent": "Mozilla/5.0 (OptiSelect student project)"}


def _get(url, timeout=30):
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    return r


def _try_save(df, path, **kw):
    try:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(path, index=False, **kw)
    except Exception:
        pass  # read-only file systems are fine


# ---------------------------------------------------------------- Heart ----
def _read_heart_file(path):
    """Accepts our CSV (with header) or raw processed.cleveland.data (no header)."""
    first = Path(path).read_text(errors="ignore").splitlines()[0]
    if "age" in first.lower():
        df = pd.read_csv(path, na_values="?")
    else:
        df = pd.read_csv(path, header=None, names=HEART_COLS, na_values="?")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _heart_local():
    for p in [HEART_FILE, DATA_DIR / "processed.cleveland.data", DATA_DIR / "heart.csv"]:
        if p.exists():
            df = _read_heart_file(p)
            if "num" in df.columns or "target" in df.columns:
                return df.rename(columns={"target": "num"})
    return None


def _heart_download():
    errors = []
    try:
        z = zipfile.ZipFile(BytesIO(_get("https://archive.ics.uci.edu/static/public/45/heart+disease.zip").content))
        member = next(n for n in z.namelist() if n.lower().endswith("processed.cleveland.data"))
        df = pd.read_csv(BytesIO(z.read(member)), header=None, names=HEART_COLS, na_values="?")
        return df
    except Exception as e:
        errors.append(f"UCI zip: {e}")
    for url in ["https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"]:
        try:
            return pd.read_csv(StringIO(_get(url).text), header=None, names=HEART_COLS, na_values="?")
        except Exception as e:
            errors.append(f"{url}: {e}")
    try:  # OpenML fallback (Cleveland = 'heart-c'); categorical -> ordinal codes
        from sklearn.datasets import fetch_openml
        d = fetch_openml("heart-c", version=1, as_frame=True, parser="auto")
        X = d.data.copy()
        for c in X.columns:
            if str(X[c].dtype) in ("category", "object"):
                X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
        X["num"] = (d.target.astype(str).str.strip() != "<50").astype(int).values
        X.columns = [c.lower() for c in X.columns]
        return X
    except Exception as e:
        errors.append(f"OpenML: {e}")
    raise RuntimeError("; ".join(errors))


def load_heart():
    df = _heart_local()
    if df is None:
        df = _heart_download()
        _try_save(df, HEART_FILE)
    feats = [c for c in df.columns if c != "num"]
    X = df[feats].apply(pd.to_numeric, errors="coerce")        # '?' -> NaN, imputed later in Pipeline
    y = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)   # 0 = healthy, 1-4 = disease
    return X, y, ["No disease", "Disease"], df


# ----------------------------------------------------------- Wine quality ----
def _wine_local():
    if WINE_RED_FILE.exists() and WINE_WHITE_FILE.exists():
        return (pd.read_csv(WINE_RED_FILE, sep=";"), pd.read_csv(WINE_WHITE_FILE, sep=";"))
    return None


def _wine_download():
    errors = []
    try:
        z = zipfile.ZipFile(BytesIO(_get("https://archive.ics.uci.edu/static/public/186/wine+quality.zip").content))
        names = {n.rsplit("/", 1)[-1].lower(): n for n in z.namelist()}
        return (pd.read_csv(BytesIO(z.read(names["winequality-red.csv"])), sep=";"),
                pd.read_csv(BytesIO(z.read(names["winequality-white.csv"])), sep=";"))
    except Exception as e:
        errors.append(f"UCI zip: {e}")
    try:
        base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
        return (pd.read_csv(StringIO(_get(base + "winequality-red.csv").text), sep=";"),
                pd.read_csv(StringIO(_get(base + "winequality-white.csv").text), sep=";"))
    except Exception as e:
        errors.append(f"UCI legacy: {e}")
    try:
        from sklearn.datasets import fetch_openml
        out = []
        for did in (40691, 40498):   # wine-quality-red, wine-quality-white
            d = fetch_openml(data_id=did, as_frame=True, parser="auto")
            df = d.data.copy()
            df["quality"] = pd.to_numeric(d.target.astype(str), errors="coerce")
            df.columns = [c.replace("_", " ") for c in df.columns]
            out.append(df)
        return tuple(out)
    except Exception as e:
        errors.append(f"OpenML: {e}")
    raise RuntimeError("; ".join(errors))


def load_wine_quality():
    pair = _wine_local()
    if pair is None:
        pair = _wine_download()
        try:
            DATA_DIR.mkdir(exist_ok=True)
            pair[0].to_csv(WINE_RED_FILE, sep=";", index=False)
            pair[1].to_csv(WINE_WHITE_FILE, sep=";", index=False)
        except Exception:
            pass
    red, white = pair
    df = pd.concat([red, white], ignore_index=True)
    y = (pd.to_numeric(df["quality"], errors="coerce") >= 6).astype(int)   # documented target
    X = df.drop(columns=["quality"]).apply(pd.to_numeric, errors="coerce")
    return X, y, ["Quality < 6", "Quality >= 6"], df


# ------------------------------------------------------------------ main ----
def _sk(loader):
    d = loader(as_frame=True)
    return d.data.copy(), pd.Series(d.target, name="target"), [str(c) for c in d.target_names], d.frame


def load_dataset(name):
    """Returns X (DataFrame), y (Series), class_names (list), raw preview frame."""
    if name == "Iris":
        out = _sk(load_iris)
    elif name == "Wine":
        out = _sk(load_wine)
    elif name == "Breast Cancer Wisconsin":
        out = _sk(load_breast_cancer)
    elif name == "Heart Disease (UCI)":
        out = load_heart()
    elif name == "Wine Quality":
        out = load_wine_quality()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    X, y, names, raw = out
    X = X.replace([np.inf, -np.inf], np.nan)
    ok = y.notna()
    return X.loc[ok].reset_index(drop=True), y.loc[ok].reset_index(drop=True), names, raw
