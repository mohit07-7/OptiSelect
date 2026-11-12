# OptiSelect AI — Group 1

A Streamlit web application for comparing all-feature classification baselines with selected feature subsets across five datasets.

## Included datasets
1. Iris — scikit-learn
2. Wine — scikit-learn
3. Breast Cancer Wisconsin — scikit-learn
4. Heart Disease (UCI Cleveland) — loaded from UCI-compatible public data source
5. Wine Quality (red + white) — loaded from the UCI Machine Learning Repository; classification target is `quality >= 6`

The first three datasets are bundled with scikit-learn. Heart Disease and Wine Quality require an internet connection when first loaded.

## Run locally
Install Python 3.10 or newer, then in this folder:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually http://localhost:8501). Open it in a browser.

## Features
- Three classifiers: Logistic Regression, Decision Tree, Random Forest
- SelectKBest, RFE, and manual feature selection
- Stratified train/test split and 5-fold stratified cross-validation
- Imputation and feature selection are fitted inside training pipelines to reduce leakage
- Baseline vs optimized accuracy, macro-F1, CV accuracy and runtime
- Five required visualizations, class distribution, correlation heatmap, classification report
- Dataset CSV and experiment comparison CSV downloads

## Deployment (Streamlit Community Cloud)
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Open https://share.streamlit.io/ and sign in.
3. Choose the repository, branch, and `app.py` as the main file.
4. Deploy. Keep the repository public if using the free deployment workflow.

## Methodology notes
- `random_state=42` is the default.
- Heart Disease target: `num > 0` means disease; missing numeric values are median-imputed inside the model pipeline.
- Wine Quality target: `quality >= 6` is the positive class; red and white wines are combined. The 11 physicochemical measurements are model inputs; `wine_type` is shown only as preview metadata.
- The accuracy-vs-feature-count curve is based on cross-validation on the training split only.
- Runtime includes cross-validation and final fitting, so it is an experiment runtime, not pure prediction latency.
- The feature-score chart uses ANOVA F-scores computed on the training split only.
- No mockup/example accuracies are hard-coded as experiment results.
