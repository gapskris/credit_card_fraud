# %%
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import io

# Load model and scaler
model = joblib.load("SMOTEENN + LightGBM.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Demo")

# Sidebar: threshold tuning
st.sidebar.header("🎚 Threshold Tuning")
default_threshold = -0.033567
threshold = st.sidebar.slider("Set Threshold", min_value=-1.0, max_value=1.0, value=default_threshold, step=0.001)

# Sidebar: business cost simulation
st.sidebar.header("💰 Business Impact")
fp_cost = st.sidebar.number_input("Cost of False Positive (₹)", min_value=0, value=100)
fn_cost = st.sidebar.number_input("Cost of False Negative (₹)", min_value=0, value=1000)

# Input form
st.subheader("🧾 Enter Transaction Details")
amount = st.number_input("Amount", min_value=0.0, value=100.0)
time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0)
pca_inputs = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]

# Predict single transaction
input_data = np.array([[time] + pca_inputs + [amount]])
input_scaled = scaler.transform(input_data)
score = model.predict_proba(input_scaled)[:, 1]  # Probability of class 1 (fraud)
prediction = int(score[0] >= threshold)

st.subheader("🔍 Prediction Result")
st.write(f"Anomaly Score: `{score[0]:.4f}`")
st.write(f"Prediction: {'🚨 Fraudulent' if prediction else '✅ Legitimate'}")

# CSV upload for batch scoring
st.subheader("📤 Upload CSV for Batch Scoring")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    batch_scaled = scaler.transform(batch_df)
    batch_scores = model.decision_function(batch_scaled)
    batch_preds = (batch_scores >= threshold).astype(int)
    batch_df["Score"] = batch_scores
    batch_df["Prediction"] = batch_preds
    st.write("🔍 Batch Predictions")
    st.dataframe(batch_df)

    # Export predictions
    csv_buffer = io.StringIO()
    batch_df.to_csv(csv_buffer, index=False)
    st.download_button("Download Predictions as CSV", csv_buffer.getvalue(), "batch_predictions.csv", "text/csv")

# Simulate business cost (using dummy test set)
# Replace with actual test labels and scores if available
y_test = np.random.choice([0, 1], size=1000, p=[0.98, 0.02])
test_scores = np.random.normal(loc=0.0, scale=0.5, size=1000)
y_pred = (test_scores >= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
total_cost = fp * fp_cost + fn * fn_cost

st.subheader("💰 Business Impact Simulation")
st.markdown(f"**False Positives:** {fp}, **False Negatives:** {fn}")
st.markdown(f"**Estimated Total Cost:** ₹{total_cost:,}")

# Score distribution
st.subheader("📊 Score Distribution")
fig, ax = plt.subplots()
sns.histplot(test_scores[y_test == 0], color='blue', label='Legit', kde=True, ax=ax)
sns.histplot(test_scores[y_test == 1], color='red', label='Fraud', kde=True, ax=ax)
ax.axvline(threshold, color='black', linestyle='--', label='Threshold')
ax.legend()
st.pyplot(fig)

# PR Curve
prec, rec, _ = precision_recall_curve(y_test, test_scores)
ap = average_precision_score(y_test, test_scores)

st.subheader("📈 Precision-Recall Curve")
fig2, ax2 = plt.subplots()
ax2.plot(rec, prec, label=f"PR-AUC: {ap:.4f}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend()
st.pyplot(fig2)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
st.subheader("📋 Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

# Export report
csv_buffer = io.StringIO()
pd.DataFrame(report).transpose().to_csv(csv_buffer)
st.download_button("Download Report as CSV", csv_buffer.getvalue(), "classification_report.csv", "text/csv")


