# Sentiment Analysis of Tech Brands at SXSW
**Phase 4 Data Science Project**

## 1. Project Overview
This project applies Natural Language Processing (NLP) to analyze public sentiment regarding Apple and Google products during the 2011 South by Southwest (SXSW) conference. Using a dataset of over 9,000 tweets, we built a machine learning model to classify sentiment as Positive, Negative, or Neutral.

### The Business Problem
In a high-stakes environment like a major tech conference, brands need to understand public perception in real-time. This project aims to provide a tool that can automatically categorize feedback, allowing marketing teams to respond to negative trends quickly and amplify positive engagement.

## 2. Data Description
* **Source:** Brands and Product Emotions dataset (Crowdflower/Appen).
* **Dataset Size:** ~9,000 hand-labeled tweets.
* **Target Classes:** 
    * Positive Emotion
    * Negative Emotion
    * No emotion toward brand or product (Neutral)

## 3. Methodology

### Text Preprocessing
Raw tweets are inherently noisy. Our pipeline included:
* **Cleaning:** Removing URLs, @mentions, hashtags, and special characters using Regular Expressions (Regex).
* **Tokenization:** Breaking text into individual words.
* **Lemmatization:** Reducing words to their root forms (e.g., "running" → "run") using NLTK.
* **Stopword Removal:** Filtering out common words (e.g., "the", "and") that do not carry sentiment.

### Modeling
1. **Baseline Model:** Multinomial Naive Bayes using TF-IDF vectorization.
2. **Iterative Model:** Random Forest Classifier with `class_weight='balanced'` to address the significant class imbalance in the dataset.

## 4. Key Findings & Visualizations
* **Class Imbalance:** Over 60% of tweets were neutral. Negative tweets represented the smallest fraction (<10%), making high recall for negative sentiment a primary challenge.
* **WordCloud Analysis:** Major topics included "iPhone," "iPad," and "Google Party," showing that experiential marketing was a primary driver of social conversation.
* **Model Performance:** The Random Forest model achieved a significant improvement in identifying negative sentiment compared to the baseline.

## 5. Conclusions & Recommendations
* **Proactive Engagement:** Brands should use this model to monitor "Negative" sentiment spikes during live events to prevent PR issues.
* **Event Strategy:** Positive sentiment was highly correlated with social events and "pop-up" shops, suggesting that experiential marketing provides better ROI than standard product demos.

## 6. Project Structure
```text
├── Data/                          # CSV data file
├── index.ipynb                    # Primary analysis notebook
├── final_sentiment_model.pkl      # Saved Random Forest model
├── label_encoder.pkl              # Saved target encoder
└── README.md                      # Project summary