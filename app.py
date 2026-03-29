# ==========================================
# FINAL PROFESSIONAL FAKE NEWS APP
# ==========================================

import streamlit as st
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# -------------------------------
# LOAD MODELS
# -------------------------------
lr = pickle.load(open("lr_model.pkl", "rb"))
rf = pickle.load(open("rf_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
y_test, y_pred_lr, y_pred_rf = pickle.load(open("results.pkl", "rb"))

# -------------------------------
# LOAD LOTTIE
# -------------------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

# -------------------------------
# CLEAN FUNCTION
# -------------------------------
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fake News Detection", layout="wide")

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

h1 {
    text-align: center;
    font-size: 3rem;
}

.stButton>button {
    background: linear-gradient(90deg, #00D4FF, #007BFF);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

textarea {
    border-radius: 10px !important;
}

.stMetric {
    background-color: #1C1F26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER WITH ANIMATION
# -------------------------------
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("""
    <h1>Fake News Detection System</h1>
    <p style='color:gray; font-size:18px;'>
    AI-powered classification using Machine Learning & Ensemble Learning
    </p>
    """, unsafe_allow_html=True)

with col2:
    st_lottie(lottie_ai, height=200)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest (Ensemble)"]
)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Comparison", "Visualizations"])

# ===============================
# TAB 1: PREDICTION
# ===============================
with tab1:

    st.subheader("Enter News Content")
    user_input = st.text_area("", height=200)

    if st.button("Analyze"):

        with st.spinner("Analyzing content..."):

            text = clean(user_input)
            vector = vectorizer.transform([text]).toarray()

            if "Logistic" in model_choice:
                pred = lr.predict(vector)[0]
                prob = lr.predict_proba(vector)[0]
            else:
                pred = rf.predict(vector)[0]
                prob = rf.predict_proba(vector)[0]

            confidence = max(prob)

        # METRICS
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Word Count", len(user_input.split()))

        with col2:
            st.metric("Model", model_choice)

        with col3:
            st.metric("Confidence", f"{confidence:.2f}")

        st.progress(confidence)

        # RESULT
        if pred == 1:
            st.markdown("<h3 style='color:lightgreen'>REAL NEWS</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red'>FAKE NEWS</h3>", unsafe_allow_html=True)

        # REASONING
        st.subheader("Why this prediction?")

        reasons = []

        if "!" in user_input:
            reasons.append("Excessive punctuation (clickbait style)")

        if len(user_input.split()) < 20:
            reasons.append("Very short content")

        trigger_words = ["shocking", "breaking", "secret", "truth"]

        for word in trigger_words:
            if word in user_input.lower():
                reasons.append(f"Contains word '{word}'")

        if not reasons:
            reasons.append("Neutral and structured language")

        for r in reasons:
            st.write("- " + r)

# ===============================
# TAB 2: COMPARISON
# ===============================
with tab2:

    st.subheader("Model Accuracy Comparison")

    lr_acc = (y_pred_lr == y_test).mean()
    rf_acc = (y_pred_rf == y_test).mean()

    df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [lr_acc, rf_acc]
    })

    st.dataframe(df)

    plt.figure()
    plt.bar(df["Model"], df["Accuracy"])
    plt.title("Accuracy Comparison")
    st.pyplot(plt)

# ===============================
# TAB 3: VISUALIZATION
# ===============================
with tab3:

    st.subheader("Confusion Matrix")

    model_type = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest"]
    )

    if model_type == "Logistic Regression":
        cm = confusion_matrix(y_test, y_pred_lr)
    else:
        cm = confusion_matrix(y_test, y_pred_rf)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(model_type)
    st.pyplot(plt)