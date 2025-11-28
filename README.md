🚀 Project Overview

Customer reviews play a major role in e-commerce platforms.
Manual review analysis is slow, subjective, and not scalable.

This project solves that by creating a real-time sentiment classifier trained on the Amazon Polarity Dataset.

Users can:

Enter any review text

Get a sentiment prediction instantly

View EDA from sample data

Explore sample reviews

📊 Dataset

We use the Amazon Polarity Dataset (3.6M reviews).
Due to size constraints, a random subset (sample_small.csv) is used for EDA.

Dataset fields:

label – 0 (Negative), 1 (Positive)

title – Review title

text – Full review

🧠 Model Used

The model pipeline includes:

Text Cleaning

TF-IDF Vectorizer

Logistic Regression Classifier

Saved using pickle as:

tfidf.pkl

model.pkl

Why Logistic Regression?

Works well for sparse text data

Fast

High accuracy for binary classification

🧪 How to Run Locally
1. Clone the Repository
git clone https://github.com/Shivam816936/Hackathon_Sentiment_Project.git
cd Hackathon_Sentiment_Project

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run Streamlit App
streamlit run app.py

🌐 Deployment Link

Your live application is hosted on Streamlit Cloud.

🔗 https://hackathonsentimentproject-9zegjocvmjvmyfbphaskmg.streamlit.app/

📦 Files in Repository
File	Description
app.py	Streamlit front-end
model.pkl	Trained logistic regression model
tfidf.pkl	TF-IDF vectorizer
requirements.txt	Python dependencies
sample_small.csv	Subset of dataset for EDA
📈 Results

Achieved strong accuracy on test data

Real-time predictions using Streamlit

Clean UI for review testing

🏁 Conclusion

This project demonstrates a full ML pipeline:
Dataset → Processing → Model Training → Deployment → UI.

It can be extended into:

Multi-class sentiment analysis

Aspect-based sentiment

Real-time customer feedback engines
