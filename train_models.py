"""Train 5 classifiers on the UCI Breast Cancer Wisconsin (Diagnostic) dataset."""
import json, joblib, pandas as pd, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

import os
os.makedirs("model", exist_ok=True)
RANDOM_STATE = 42
raw = load_breast_cancer(as_frame=True)
df = raw.frame.rename(columns={"target": "diagnosis"})  # 1 = benign, 0 = malignant
X, y = df.drop(columns="diagnosis"), df["diagnosis"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000, C=1.0),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=2.0, probability=True, random_state=RANDOM_STATE),
}
results = {}
for name, clf in classifiers.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)]).fit(X_tr, y_tr)
    pred, prob = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    results[name] = {k: round(float(v), 4) for k, v in {
        "Accuracy": accuracy_score(y_te, pred), "AUC": roc_auc_score(y_te, prob),
        "Precision": precision_score(y_te, pred), "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred), "MCC": matthews_corrcoef(y_te, pred)}.items()}
    joblib.dump(pipe, f"model/{name.lower().replace(' ', '_')}.joblib")
json.dump({"features": list(X.columns), "target": "diagnosis"}, open("model/feature_info.json", "w"), indent=2)
json.dump(results, open("model/metrics.json", "w"), indent=2)
print(pd.DataFrame(results).T.to_string())
