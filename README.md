# 🎯 Customer Feedback Analysis System - AI-Powered

An AI-powered customer feedback analysis system with sentiment classification, summarization, trend prediction, and web deployment.

## Key Features

    Python 3.13.9, Flask 3.1.2 backend

    Sentiment analysis with 88.39% F1-score using Logistic Regression

    TF-IDF based text summarization

    Time-series sentiment trend forecasting

    5,675 real feedback records processed from multiple sources

    Flask web app with Dashboard, Reports, and Settings pages

    REST API with 9 endpoints for prediction, analytics, and insights

    Dockerized production-ready deployment

## Quick Start

```bash
    # Setup environment and dependencies
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    # Download NLTK data
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

    # Run app
    python src/app.py
```

## 🌐 Live Demo

**Deployed Application:** [https://customer-feedback-analysis-system.onrender.com/](https://customer-feedback-analysis-system.onrender.com/)