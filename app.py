import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
st.title("🎓 AI Student Placement Predictor")

# Load data
df = pd.read_csv("dataset.csv")

X = df.drop("Placed", axis=1)
y = df["Placed"]

@st.cache_resource
def train_models(X, y):
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    lr.fit(X, y)
    rf.fit(X, y)
    
    return lr, rf


lr_model, rf_model = train_models(X, y)


# 🔹 Feature importance MUST be defined BEFORE using it
importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

st.subheader("Enter Student Details")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
internships = st.number_input("Number of Internships", 0, 5, 1)
projects = st.number_input("Number of Projects", 0, 10, 2)
backlogs = st.number_input("Backlogs", 0, 10, 0)
aptitude = st.slider("Aptitude Score", 0, 100, 70)
communication = st.slider("Communication Score", 0, 100, 75)

if st.button("Predict Placement"):
    student = [[cgpa, internships, projects, backlogs, aptitude, communication]]
    
    prob = rf_model.predict_proba(student)[0][1] * 100

    st.write(f"📊 Placement Probability: **{prob:.2f}%**")

    if prob >= 50:
        st.success("✅ Student is LIKELY to be PLACED")
    else:
        st.error("❌ Student is NOT likely to be placed")

st.subheader("📌 Feature Importance")
st.bar_chart(importance)
lr_acc = lr_model.score(X, y)
rf_acc = rf_model.score(X, y)

st.subheader("📈 Model Comparison")
st.write(f"Logistic Regression Accuracy: {lr_acc:.2f}")
st.write(f"Random Forest Accuracy: {rf_acc:.2f}")
