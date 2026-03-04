import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
import nltk

# download stopwords (only runs first time)
nltk.download("stopwords")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location="cpu"))
model.eval()

# Stopwords
stop_words = set(stopwords.words("english"))


# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return words


# -----------------------------
# Sentiment Prediction
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
# Phrase Extraction
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