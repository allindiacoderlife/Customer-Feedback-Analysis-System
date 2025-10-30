from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Global variables for models and data
sentiment_model = None
vectorizer = None
data = None
summaries = None
insights = None

def load_models_and_data():
    """Load all models and data on startup"""
    global sentiment_model, vectorizer, data, summaries, insights
    
    print("Loading models and data...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load sentiment model
    model_path = os.path.join(base_dir, '..', 'models', 'sentiment_model.pkl')
    with open(model_path, 'rb') as f:
        sentiment_model = pickle.load(f)
    print("✓ Sentiment model loaded")
    
    # Load vectorizer
    vectorizer_path = os.path.join(base_dir, '..', 'models', 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✓ Vectorizer loaded")
    
    # Load data
    data_path = os.path.join(base_dir, '..', 'dataset', 'cleaned_customer_feedback.csv')
    data = pd.read_csv(data_path)
    print(f"✓ Data loaded: {len(data):,} records")
    
    # Load summaries
    try:
        summaries_path = os.path.join(base_dir, '..', 'models', 'text_summaries.pkl')
        with open(summaries_path, 'rb') as f:
            summaries = pickle.load(f)
        print("✓ Summaries loaded")
    except:
        summaries = None
        print("⚠ Summaries not found")
    
    # Load insights
    try:
        insights_path = os.path.join(base_dir, '..', 'models', 'predictive_insights.pkl')
        with open(insights_path, 'rb') as f:
            insights = pickle.load(f)
        print("✓ Insights loaded")
    except:
        insights = None
        print("⚠ Insights not found")


@app.route('/')
def home():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/predictor')
def predictor():
    """Sentiment predictor page"""
    return render_template('predictor.html')


@app.route('/api/stats')
def get_stats():
    """Get overall statistics"""
    if data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    stats = {
        'total_feedback': len(data),
        'sentiment_distribution': data['sentiment_label'].value_counts().to_dict(),
        'average_rating': float(data['rating'].mean()),
        'rating_distribution': data['rating'].value_counts().to_dict(),
        'total_positive': int((data['sentiment_label'] == 'positive').sum()),
        'total_negative': int((data['sentiment_label'] == 'negative').sum()),
        'total_neutral': int((data['sentiment_label'] == 'neutral').sum()),
    }
    
    return jsonify(stats)


@app.route('/api/predict', methods=['POST'])
def predict_sentiment():
    """Predict sentiment for new text"""
    try:
        data_input = request.get_json()
        text = data_input.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Vectorize text
        text_vectorized = vectorizer.transform([text])
        
        # Predict
        prediction = sentiment_model.predict(text_vectorized)[0]
        probabilities = sentiment_model.predict_proba(text_vectorized)[0]
        
        # Get class labels
        classes = sentiment_model.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        result = {
            'prediction': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trends')
def get_trends():
    """Get sentiment trends over time"""
    if data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Parse dates
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Monthly trends
        df['year_month'] = df['date'].dt.to_period('M').astype(str)
        monthly_sentiment = df.groupby(['year_month', 'sentiment_label']).size().unstack(fill_value=0)
        
        trends = {
            'labels': monthly_sentiment.index.tolist(),
            'positive': monthly_sentiment.get('positive', pd.Series([0])).tolist(),
            'negative': monthly_sentiment.get('negative', pd.Series([0])).tolist(),
            'neutral': monthly_sentiment.get('neutral', pd.Series([0])).tolist(),
        }
        
        return jsonify(trends)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/issues')
def get_issues():
    """Get common issues from negative feedback"""
    if data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        negative_df = data[data['sentiment_label'] == 'negative']
        
        issue_categories = {
            'Quality': ['quality', 'poor', 'bad', 'terrible', 'awful', 'horrible', 'disappointing'],
            'Service': ['service', 'staff', 'waiter', 'waitress', 'employee', 'rude', 'slow', 'unfriendly'],
            'Price': ['expensive', 'overpriced', 'costly', 'price', 'money', 'waste'],
            'Wait Time': ['wait', 'waiting', 'long', 'slow', 'delay', 'late'],
            'Food': ['food', 'meal', 'dish', 'taste', 'bland', 'cold', 'overcooked'],
            'Cleanliness': ['dirty', 'clean', 'messy', 'filthy', 'unhygienic'],
            'Atmosphere': ['noise', 'noisy', 'loud', 'crowded', 'uncomfortable', 'atmosphere']
        }
        
        issue_counts = {}
        for category, keywords in issue_categories.items():
            count = 0
            for text in negative_df['feedback_text'].astype(str):
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in keywords):
                    count += 1
            issue_counts[category] = count
        
        # Sort by count
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'categories': [item[0] for item in sorted_issues],
            'counts': [item[1] for item in sorted_issues]
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/summaries')
def get_summaries():
    """Get sentiment summaries"""
    if summaries is None:
        return jsonify({'error': 'Summaries not loaded'}), 404
    
    try:
        sentiment_summaries = summaries.get('sentiment_summaries', {})
        
        result = {}
        for sentiment, data in sentiment_summaries.items():
            result[sentiment] = {
                'count': data.get('count', 0),
                'short_summary': data.get('short_summary', ''),
                'detailed_summary': data.get('detailed_summary', '')
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/insights')
def get_insights():
    """Get actionable insights"""
    if insights is None:
        return jsonify({'error': 'Insights not loaded'}), 404
    
    try:
        insights_data = insights.get('insights', [])
        return jsonify(insights_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recent-feedback')
def get_recent_feedback():
    """Get recent feedback samples"""
    if data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Get last 10 records
        recent = data.tail(10)[['feedback_text', 'sentiment_label', 'rating']].to_dict('records')
        return jsonify(recent)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset')
def get_dataset():
    """Get full dataset for dataset explorer page"""
    if data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Prepare dataset with necessary fields
        dataset_records = []
        
        for idx, row in data.iterrows():
            record = {
                'id': int(idx) + 1,
                'date': row.get('date', datetime.now().strftime('%Y-%m-%d')),
                'feedback': row.get('feedback_text', ''),
                'sentiment': row.get('sentiment_label', 'neutral').capitalize(),
                'rating': int(row.get('rating', 3)),
                'confidence': float(row.get('sentiment_score', 0.5))
            }
            dataset_records.append(record)
        
        return jsonify({
            'data': dataset_records,
            'total': len(dataset_records),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': sentiment_model is not None,
        'data_loaded': data is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Load models and data on startup
    load_models_and_data()
    
    print("\n" + "="*80)
    print("🚀 FLASK APPLICATION STARTING")
    print("="*80)
    print("\n✅ Server ready!")
    print("📍 URL: http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000/")
    print("🔌 API: http://localhost:5000/api/stats")
    print("\n" + "="*80 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
