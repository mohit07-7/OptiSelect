"""Run once on your own PC (needs internet):  python download_data.py
Creates data/heart_cleveland.csv, data/winequality-red.csv, data/winequality-white.csv
Commit the data/ folder to GitHub so the Streamlit Cloud app never needs to download anything."""
from datasets import load_heart, load_wine_quality, HEART_FILE, WINE_RED_FILE

X, y, _, _ = load_heart()
print("Heart Disease:", X.shape, "missing cells:", int(X.isna().sum().sum()), "->", HEART_FILE)
X, y, _, _ = load_wine_quality()
print("Wine Quality:", X.shape, "positive class share:", round(float(y.mean()), 3), "->", WINE_RED_FILE.parent)
print("Done. Now add the data/ folder to your GitHub repo.")
