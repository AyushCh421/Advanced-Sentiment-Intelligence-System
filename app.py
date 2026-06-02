import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
import nltk
import os
import gdown

# -----------------------------
# Download stopwords only once
# -----------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = "bert_sentiment_model.pth"

# -----------------------------
# Download model if not present
# -----------------------------
if not os.path.exists(MODEL_PATH):
    file_id = "1SiTUjX-eePKFlJIqplgKeCYFAQoF0BLO"
    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner("Downloading model... Please wait ⏳"):
        gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# Load tokenizer and model
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# -----------------------------
# Stopwords
# -----------------------------
stop_words = set(stopwords.words("english"))

# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return words

# -----------------------------
# Predict sentiment
# -----------------------------
def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    prediction = torch.argmax(logits, dim=1).item()

    return prediction

# -----------------------------
# Extract phrases safely
# -----------------------------
def extract_phrases(text):

    words = clean_text(text)

    # Need at least 2 words for bigrams
    if len(words) < 2:
        return []

    cleaned_text = " ".join(words)

    try:
        vectorizer = CountVectorizer(
            ngram_range=(2, 2)
        )

        X = vectorizer.fit_transform([cleaned_text])

        phrases = vectorizer.get_feature_names_out()

        return list(phrases)

    except ValueError:
        return []

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Advanced Sentiment Intelligence System",
    page_icon="📊"
)

st.title("📊 Advanced Sentiment Intelligence System")

st.write(
    "Analyze customer reviews and detect sentiment with possible reasons."
)

review = st.text_area(
    "Enter Customer Review",
    height=150
)

if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    try:

        sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        sentiment_id = predict_sentiment(review)

        sentiment = sentiment_map.get(
            sentiment_id,
            "Unknown"
        )

        keywords = clean_text(review)

        phrases = extract_phrases(review)

        st.subheader("Sentiment")
        st.success(sentiment)

        st.subheader("Keyword Reasons")

        if keywords:
            st.write(keywords)
        else:
            st.write("No keywords found.")

        st.subheader("Reason Phrases")

        if phrases:
            st.write(phrases)
        else:
            st.write("No meaningful phrases found.")

    except Exception as e:
        st.error(f"Error: {str(e)}")