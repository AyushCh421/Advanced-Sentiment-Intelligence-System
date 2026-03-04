# Advanced Sentiment Intelligence System (ASIS)

## Overview

The Advanced Sentiment Intelligence System (ASIS) is a Natural Language Processing project that performs sentiment analysis on customer reviews and automatically identifies the reasons behind customer satisfaction or dissatisfaction.

The system uses a fine-tuned BERT transformer model to classify reviews into Positive, Neutral, or Negative sentiments. After predicting sentiment, the system extracts meaningful keywords and phrase-level insights to explain why a review was classified in a particular way.

This project demonstrates a complete NLP pipeline including model training, evaluation, inference, and an interactive web interface.

---

## Features

- Transformer-based Sentiment Analysis using BERT
- Real-time sentiment prediction
- Automatic keyword extraction
- Phrase-level reason detection using bigrams
- Dataset-level insight extraction
- Interactive web application using Streamlit

---

## Model Details

Model Used:
- BERT (bert-base-uncased)

Task:
- Multi-class sentiment classification

Sentiment Classes:
- Negative
- Neutral
- Positive

Training Framework:
- PyTorch
- HuggingFace Transformers

---

## System Pipeline

Customer Review  
↓  
BERT Sentiment Classification  
↓  
Keyword Extraction  
↓  
Phrase Detection (Bigrams)  
↓  
Reason Explanation  

Example:

Input Review:The delivery was very late and the food was cold


Output:


Sentiment: Negative
Reason Keywords: delivery, late, food, cold
Reason Phrases: delivery late, food cold


---

## Project Structure


Advanced-Sentiment-Intelligence-System/

│
├── notebooks/
│ ├── Sentiment_Analysis(01).ipynb│
|  |--Reason_Insight_Analysis(02).ipynb
│
├── app.py
├── requirements.txt
│
├── README.md
└── .gitignore


---

## Model File

The trained BERT model file (`bert_sentiment_model.pth`) is not included in this repository because it exceeds GitHub's file size limit.

To run the project locally:

1. Train the model using the training notebook:
notebooks/Sentiment_Analysis(01).ipynb
2. Save the trained model as:

bert_sentiment_model.pth
3. Place the file in the root project directory.

Once the model file is generated, the Streamlit application can load it and perform sentiment prediction.

## Installation

Clone the repository:git clone https://github.com/AyushCh421/Advanced-Sentiment-Intelligence-System


Move into the project folder:


cd Advanced-Sentiment-Intelligence-System


Install dependencies:


pip install -r requirements.txt


---

## Running the Streamlit Application

Start the Streamlit app:


streamlit run app.pyAfter running, open the following in your browser:


http://localhost:8501


---

## Example Usage

Input:


The food was delicious but the delivery was late


Output:


Sentiment: Negative
Keywords: food, delicious, delivery, late
Reason Phrases: food delicious, delivery late


---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Streamlit
- Scikit-learn
- NLTK
- Pandas

---

## Future Improvements

- Improved sentiment calibration for mixed reviews
- Phrase-level explanation using dependency parsing
- Advanced topic modeling for deeper insights
- Deployment using cloud platforms

---

## Author

Ayush Chauhan  
B.Tech Student  
Aspiring Machine Learning Engineer
