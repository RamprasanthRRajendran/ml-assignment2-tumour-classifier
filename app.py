"""Streamlit app: Breast Cancer Diagnosis – Classifier Comparison (ML Assignment 2)."""
import json, joblib, pandas as pd, streamlit as st, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

MODEL_DIR = Path(__file__).parent / "model"
MODEL_FILES = {"Logistic Regression": "logistic_regression.joblib", "Decision Tree": "decision_tree.joblib",
               "KNN": "knn.joblib", "Naive Bayes": "naive_bayes.joblib",
               "Support Vector Machine": "support_vector_machine.joblib"}

st.set_page_config(page_title="Tumour Diagnosis Classifier Lab", page_icon="🩺", layout="wide")
st.title("🩺 Tumour Diagnosis Classifier Lab")
st.caption("Compare five pre-trained classifiers on the UCI Breast Cancer Wisconsin (Diagnostic) data. "
           "Models are loaded from disk – no retraining happens on uploaded data.")

@st.cache_resource
def load_model(fname): return joblib.load(MODEL_DIR / fname)
info = json.loads((MODEL_DIR / "feature_info.json").read_text())

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload test CSV (features + 'diagnosis')", type="csv")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))
    show_all = st.checkbox("Also compare all models on this data", value=True)
    st.markdown("Label: **1 = benign, 0 = malignant**")

if uploaded is None:
    st.info("⬅️ Upload `test_data.csv` from the GitHub repo to begin.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in info["features"] + [info["target"]] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {missing}"); st.stop()
X, y = df[info["features"]], df[info["target"]]
st.subheader("📄 Uploaded data preview")
st.write(f"{len(df)} rows × {len(info['features'])} features")
st.dataframe(df.head(), use_container_width=True)

def evaluate(model):
    pred, prob = model.predict(X), model.predict_proba(X)[:, 1]
    return pred, {"Accuracy": accuracy_score(y, pred), "AUC": roc_auc_score(y, prob),
                  "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred),
                  "F1": f1_score(y, pred), "MCC": matthews_corrcoef(y, pred)}

pred, m = evaluate(load_model(MODEL_FILES[model_name]))
st.subheader(f"📊 Evaluation metrics – {model_name}")
for col, (k, v) in zip(st.columns(6), m.items()): col.metric(k, f"{v:.4f}")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4, 3.2))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="PuBu",
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig)
with c2:
    st.markdown("**Classification report**")
    rep = classification_report(y, pred, target_names=["Malignant", "Benign"], output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

if show_all:
    st.subheader("🏁 All models on the uploaded data")
    table = pd.DataFrame({n: evaluate(load_model(f))[1] for n, f in MODEL_FILES.items()}).T.round(4)
    st.dataframe(table.style.highlight_max(axis=0, color="#c6efce"), use_container_width=True)
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
