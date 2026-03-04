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
# Download stopwords
# -----------------------------
nltk.download("stopwords")

# -----------------------------
# Model download from Google Drive
# -----------------------------
MODEL_PATH = "bert_sentiment_model.pth"

if not os.path.exists(MODEL_PATH):

    file_id = "1SiTUjX-eePKFlJIqplgKeCYFAQoF0BLO"
    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner("Downloading model... This may take a moment ⏳"):
        gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# -----------------------------
# Load model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

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
# Sentiment prediction
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
# Extract phrases
# -----------------------------
def extract_phrases(text):

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(2, 2)
    )

    X = vectorizer.fit_transform([text])

    phrases = vectorizer.get_feature_names_out()

    return phrases

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Advanced Sentiment Intelligence System")

st.write("Analyze customer reviews and detect sentiment with possible reasons.")

review = st.text_area("Enter Customer Review")

if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        sentiment_id = predict_sentiment(review)
        sentiment = sentiment_map[sentiment_id]

        keywords = clean_text(review)
        phrases = extract_phrases(review)

        st.subheader("Sentiment")
        st.write(sentiment)

        st.subheader("Keyword Reasons")
        st.write(keywords)

        st.subheader("Reason Phrases")
        st.write(phrases)