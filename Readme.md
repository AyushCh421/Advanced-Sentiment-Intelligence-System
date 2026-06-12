# Advanced Sentiment Intelligence System (ASIS)

> Fine-tuned BERT transformer for 3-class sentiment classification 
> on customer reviews — achieving **85%+ accuracy**, outperforming 
> TF-IDF baseline by 12 points.

## Performance
| Metric | Score |
|--------|-------|
| Accuracy | 85%+ |
| Model | BERT (bert-base-uncased) |
| Classes | Positive / Neutral / Negative |
| Dataset | 10,000+ customer reviews |
| Baseline (TF-IDF) | ~73% |

## Overview
The Advanced Sentiment Intelligence System (ASIS) is an NLP project 
that performs sentiment analysis on customer reviews and automatically 
identifies the reasons behind customer satisfaction or dissatisfaction.

## Features

- Transformer-based Sentiment Analysis using BERT
- Real-time sentiment prediction
- Automatic keyword extraction
- Phrase-level reason detection using bigrams
- Dataset-level insight extraction
- Interactive web application using Streamlit

---

Note: The trained model file is excluded from the repository due to GitHub file size limitations.

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
