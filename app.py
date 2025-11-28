import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords

# download stopwords quietly on first run
nltk.download('stopwords', quiet=True)
stop = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join([w for w in text.split() if w not in stop])

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

@st.cache_data
def load_sample():
    try:
        return pd.read_csv("sample_small.csv")
    except Exception:
        return pd.read_csv("sample_data.csv")

st.set_page_config(page_title="Sentiment Classifier", layout="wide")
st.title("Amazon Review Sentiment — Demo")

model, tfidf = load_model()
sample_df = load_sample()

st.sidebar.header("About")
st.sidebar.write("Simple demo: TF-IDF + Logistic Regression")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Enter a review to predict sentiment")
    user_input = st.text_area("Type or paste a review here", height=160)

    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_input)
            vect = tfidf.transform([cleaned])
            pred = model.predict(vect)[0]
            proba = model.predict_proba(vect).max()
            label = "Positive" if pred == 1 else "Negative"
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** {proba:.2f}")

    st.markdown("---")
    st.subheader("Or choose a sample review")
    if not sample_df.empty:
        idx = st.number_input("Sample row index", min_value=0, max_value=len(sample_df)-1, value=0)
        row = sample_df.iloc[idx]
        st.write("**Review (label = {})**".format(row.get("label", "")))
        st.write(row.get("review", ""))
        if st.button("Predict sample"):
            cleaned = clean_text(row.get("review",""))
            vect = tfidf.transform([cleaned])
            pred = model.predict(vect)[0]
            proba = model.predict_proba(vect).max()
            label = "Positive" if pred == 1 else "Negative"
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** {proba:.2f}")
    else:
        st.info("No sample file found. Upload sample_small.csv to the app folder.")

with col2:
    st.subheader("Quick EDA")
    if not sample_df.empty:
        counts = sample_df['label'].value_counts()
        st.write("Class distribution (sample):")
        st.bar_chart(counts)
        st.write("Sample rows:")
        st.dataframe(sample_df.head(10))
    else:
        st.write("No sample data available.")

st.markdown("---")
st.caption("Model: TF-IDF + Logistic Regression — saved as model.pkl.")
