# OptiSelect - Smart Feature Selection (Group 1, Optimization)

Streamlit dashboard comparing classifiers on ALL features vs SelectKBest / RFE subsets
for Iris, Wine, Breast Cancer, Heart Disease (UCI Cleveland) and Wine Quality (UCI, quality >= 6 = good).

## Run
```
python -m venv .venv && .venv\Scripts\activate      # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Heart Disease and Wine Quality download automatically from UCI on first use and are cached in `data/`.
If you are offline, save these files in `data/` manually:
- `processed.cleveland.data` (UCI heart-disease folder)
- `winequality-red.csv`, `winequality-white.csv` (UCI wine-quality folder)

## Files
- `experiments.py` - loaders, Pipelines (imputer -> scaler -> selector -> model), CV + test evaluation
- `plots.py` - the 5 mandatory figures + class distribution / correlation heatmap (PNGs saved to `results/`)
- `app.py` - dashboard; `results/results_table.csv` is produced by the "All datasets table" tab

## Method notes
random_state=42; stratified 80/20 split; imputation, scaling and selection are fitted on training data only;
5-fold stratified CV on the training split picks/compares k; the test set is evaluated once per configuration.
