# OptiSelect AI - Multi-Dataset Smart Feature Selection

Files: `app.py` (dashboard) · `experiments.py` (pipelines, CV, evaluation) · `plots.py` (figures) ·
`datasets.py` (loaders) · `download_data.py` · `data/` · `results/` (PNG figures saved on every run).

## Run locally
```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
python download_data.py         # ONE time, needs internet: saves Heart Disease + Wine Quality into data/
streamlit run app.py
```

## Deploy on Streamlit Cloud (important)
1. Run `python download_data.py` on your PC so `data/` contains `heart_cleveland.csv`,
   `winequality-red.csv`, `winequality-white.csv`.
2. Upload EVERYTHING (including `data/`) to the GitHub repo. `app.py` must be at the repo root
   (or set the correct main-file path in Streamlit Cloud).
3. In Streamlit Cloud choose "Reboot app" (or "Clear cache").

Iris, Wine and Breast Cancer come from scikit-learn and always work offline.
Heart Disease: '?' values become NaN and are median-imputed inside the Pipeline (train data only).
Wine Quality: red + white merged, 11 physicochemical features, target = quality >= 6.
