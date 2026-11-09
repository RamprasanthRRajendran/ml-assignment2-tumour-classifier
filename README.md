# Tumour Diagnosis Classifier Lab – ML Assignment 2

**Live app:** https://ramprasanth-tumour-classifier.streamlit.app/  **Repository:** https://github.com/RamprasanthRRajendran/ml-assignment2-tumour-classifier

## a. Problem statement
Predict whether a breast tumour is **malignant (0)** or **benign (1)** from 30 numeric features computed from a digitised fine-needle-aspirate (FNA) image, and compare five classification algorithms — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine — using Accuracy, AUC, Precision, Recall, F1 and MCC. The best models are served through an interactive Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository – *Breast Cancer Wisconsin (Diagnostic)* (also on Kaggle).
- **Instances:** 569 (357 benign, 212 malignant) — binary classification.
- **Features:** 30 real-valued features: mean, standard error and "worst" values of 10 nucleus characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- **Target:** `diagnosis` (0 = malignant, 1 = benign). No missing values.
- **Split:** 80/20 stratified train/test (random_state = 42). The 114-row test split is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` inside each sklearn `Pipeline`, so scaling is saved with the model.

## c. GitHub repository link
https://github.com/RamprasanthRRajendran/ml-assignment2-tumour-classifier

```
project-folder/
├── app.py               # Streamlit app
├── train_models.py      # training + evaluation script
├── requirements.txt
├── README.md
├── test_data.csv        # held-out test data (features + target)
├── .github/workflows/train.yml  # CI job that trains the models and commits them
└── model/               # 5 saved pipelines (.joblib), metrics.json, feature_info.json
```

## d. Models used and comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9358 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9967 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

*Positive class = benign (1). Hyper-parameters: LR (C=1, max_iter=5000), DT (max_depth=5, min_samples_leaf=4), KNN (k=7, distance-weighted), GaussianNB, SVM (RBF, C=2).*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Joint best accuracy (98.25%) and MCC (0.962). The classes are close to linearly separable after scaling, so a simple linear boundary works very well and is easy to interpret. |
| Decision Tree | Weakest model (90.35% accuracy, MCC 0.797). Even with depth limited to 5, axis-aligned splits are unstable and overfit noise; its lower AUC reflects coarse, step-like probabilities. |
| KNN | Strong (97.37%) with perfect recall for benign cases, but slightly lower precision — a few malignant tumours were labelled benign, which matters clinically. Depends heavily on feature scaling. |
| Naive Bayes | Reasonable (92.98%) with high AUC (0.987), but the independence assumption is violated (radius, perimeter and area are strongly correlated), which hurts hard-label accuracy. |
| Support Vector Machine | Ties LR on accuracy/F1/MCC and has the highest AUC (0.9967). The RBF kernel with a maximum-margin boundary gives the best class separation and ranking. |

**Overall winner:** **Support Vector Machine (RBF)** — tied best on Accuracy, F1 and MCC with Logistic Regression and highest AUC. Logistic Regression is an equally good, more interpretable alternative.

## Streamlit app features
1. CSV upload of test data  2. Model selection dropdown  3. Six evaluation metrics  4. Confusion matrix + classification report  5. Side-by-side comparison of all models. Models are loaded from `model/` and never retrained.

## Run locally
```bash
pip install -r requirements.txt
python train_models.py   # optional: re-creates models and test_data.csv
streamlit run app.py
```
