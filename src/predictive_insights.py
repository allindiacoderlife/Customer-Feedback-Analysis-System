import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime, timedelta
from collections import Counter
import re
warnings.filterwarnings('ignore')

# ML and visualization libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_data_and_models():
    """Load preprocessed data and trained models"""
    print("="*80)
    print("LOADING DATA AND MODELS")
    print("="*80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load dataset
    data_path = os.path.join(script_dir, '..', 'dataset', 'cleaned_customer_feedback.csv')
    df = pd.read_csv(data_path)
    print(f"\n✓ Dataset loaded: {len(df):,} records")
    
    # Load sentiment model
    model_path = os.path.join(script_dir, '..', 'models', 'sentiment_model.pkl')
    with open(model_path, 'rb') as f:
        sentiment_model = pickle.load(f)
    print(f"✓ Sentiment model loaded")
    
    # Load vectorizer
    vectorizer_path = os.path.join(script_dir, '..', 'models', 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"✓ TF-IDF vectorizer loaded")
    
    return df, sentiment_model, vectorizer


def parse_dates(df):
    """Parse and process date information"""
    print("\n" + "="*80)
    print("PROCESSING TEMPORAL DATA")
    print("="*80)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Remove rows with invalid dates
    valid_dates = df['date'].notna()
    print(f"\n✓ Valid dates: {valid_dates.sum():,} / {len(df):,}")
    
    df = df[valid_dates].copy()
    
    # Extract temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Create period labels
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    df['year_week'] = df['date'].dt.to_period('W').astype(str)
    
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"✓ Time span: {(df['date'].max() - df['date'].min()).days} days")
    
    return df


def analyze_sentiment_trends(df):
    """Analyze sentiment trends over time"""
    print("\n" + "="*80)
    print("SENTIMENT TREND ANALYSIS")
    print("="*80)
    
    # Monthly sentiment distribution
    monthly_sentiment = df.groupby(['year_month', 'sentiment_label']).size().unstack(fill_value=0)
    
    # Calculate sentiment percentages
    monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
    
    print("\n📈 Monthly Sentiment Distribution (%):")
    print(monthly_sentiment_pct.round(2).head(10))
    
    # Overall trend
    monthly_positive = monthly_sentiment_pct.get('positive', pd.Series([0]))
    trend_direction = "📈 Improving" if monthly_positive.iloc[-1] > monthly_positive.iloc[0] else "📉 Declining"
    
    print(f"\n✓ Sentiment Trend: {trend_direction}")
    print(f"  First month positive: {monthly_positive.iloc[0]:.1f}%")
    print(f"  Last month positive: {monthly_positive.iloc[-1]:.1f}%")
    
    return monthly_sentiment, monthly_sentiment_pct


def predict_rating_from_sentiment(df):
    """Build model to predict rating from sentiment and text features"""
    print("\n" + "="*80)
    print("RATING PREDICTION MODEL")
    print("="*80)
    
    # Prepare features
    df_clean = df.dropna(subset=['rating', 'sentiment_label', 'word_count']).copy()
    
    # Encode sentiment as numeric
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df_clean['sentiment_numeric'] = df_clean['sentiment_label'].map(sentiment_map)
    
    # Features for prediction
    features = ['sentiment_numeric', 'word_count', 'feedback_length']
    X = df_clean[features].values
    y = df_clean['rating'].values
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train models
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Evaluation
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"\n📊 Linear Regression Performance:")
    print(f"   MAE: {lr_mae:.3f}")
    print(f"   MSE: {lr_mse:.3f}")
    print(f"   R²:  {lr_r2:.3f}")
    
    print(f"\n📊 Random Forest Performance:")
    print(f"   MAE: {rf_mae:.3f}")
    print(f"   MSE: {rf_mse:.3f}")
    print(f"   R²:  {rf_r2:.3f}")
    
    # Select best model
    best_model = rf_model if rf_mae < lr_mae else lr_model
    best_name = "Random Forest" if rf_mae < lr_mae else "Linear Regression"
    
    print(f"\n🏆 Best Model: {best_name}")
    
    return best_model, best_name, {'lr': (lr_mae, lr_r2), 'rf': (rf_mae, rf_r2)}


def detect_common_issues(df):
    """Detect and categorize common issues from negative feedback"""
    print("\n" + "="*80)
    print("ISSUE DETECTION & CATEGORIZATION")
    print("="*80)
    
    # Focus on negative feedback
    negative_df = df[df['sentiment_label'] == 'negative'].copy()
    print(f"\n✓ Analyzing {len(negative_df):,} negative reviews")
    
    # Define issue keywords
    issue_categories = {
        'Service': ['service', 'staff', 'waiter', 'waitress', 'employee', 'rude', 'slow', 'unfriendly'],
        'Quality': ['quality', 'poor', 'bad', 'terrible', 'awful', 'horrible', 'disappointing'],
        'Price': ['expensive', 'overpriced', 'costly', 'price', 'money', 'waste'],
        'Cleanliness': ['dirty', 'clean', 'messy', 'filthy', 'unhygienic'],
        'Wait Time': ['wait', 'waiting', 'long', 'slow', 'delay', 'late'],
        'Food': ['food', 'meal', 'dish', 'taste', 'bland', 'cold', 'overcooked'],
        'Atmosphere': ['noise', 'noisy', 'loud', 'crowded', 'uncomfortable', 'atmosphere']
    }
    
    # Count issues
    issue_counts = {}
    for category, keywords in issue_categories.items():
        count = 0
        for text in negative_df['feedback_text'].astype(str):
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in keywords):
                count += 1
        issue_counts[category] = count
    
    # Sort by frequency
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🔍 Top Issues Detected:")
    for category, count in sorted_issues:
        percentage = (count / len(negative_df)) * 100
        print(f"   {category:15s}: {count:4d} mentions ({percentage:5.1f}%)")
    
    return issue_counts, sorted_issues


def generate_actionable_insights(df, issue_counts, sentiment_trends):
    """Generate actionable business recommendations"""
    print("\n" + "="*80)
    print("ACTIONABLE INSIGHTS & RECOMMENDATIONS")
    print("="*80)
    
    insights = []
    
    # 1. Overall sentiment insight
    sentiment_dist = df['sentiment_label'].value_counts(normalize=True) * 100
    positive_pct = sentiment_dist.get('positive', 0)
    negative_pct = sentiment_dist.get('negative', 0)
    
    if positive_pct > 60:
        insights.append({
            'category': 'Overall Sentiment',
            'status': 'Positive',
            'insight': f'{positive_pct:.1f}% of feedback is positive',
            'recommendation': 'Maintain current quality standards and consider expanding successful practices'
        })
    elif negative_pct > 30:
        insights.append({
            'category': 'Overall Sentiment',
            'status': 'Concerning',
            'insight': f'{negative_pct:.1f}% of feedback is negative',
            'recommendation': 'Immediate action required to address customer concerns'
        })
    
    # 2. Top issue insight
    if issue_counts:
        top_issue, top_count = max(issue_counts.items(), key=lambda x: x[1])
        insights.append({
            'category': 'Primary Issue',
            'status': 'Action Required',
            'insight': f'{top_issue} is the #1 complaint ({top_count} mentions)',
            'recommendation': f'Prioritize improvements in {top_issue.lower()} to reduce negative feedback'
        })
    
    # 3. Rating insight
    avg_rating = df['rating'].mean()
    if avg_rating < 3.0:
        insights.append({
            'category': 'Rating',
            'status': 'Critical',
            'insight': f'Average rating is {avg_rating:.2f}/5.0',
            'recommendation': 'Implement immediate quality improvement initiatives'
        })
    elif avg_rating > 4.0:
        insights.append({
            'category': 'Rating',
            'status': 'Excellent',
            'insight': f'Average rating is {avg_rating:.2f}/5.0',
            'recommendation': 'Leverage positive reviews in marketing materials'
        })
    
    # 4. Trend insight
    monthly_sentiment_pct = sentiment_trends
    if len(monthly_sentiment_pct) >= 2:
        recent_positive = monthly_sentiment_pct.iloc[-1].get('positive', 0)
        prev_positive = monthly_sentiment_pct.iloc[0].get('positive', 0)
        
        if recent_positive > prev_positive + 5:
            insights.append({
                'category': 'Trend',
                'status': 'Improving',
                'insight': f'Positive sentiment increased from {prev_positive:.1f}% to {recent_positive:.1f}%',
                'recommendation': 'Continue current improvement strategies'
            })
        elif recent_positive < prev_positive - 5:
            insights.append({
                'category': 'Trend',
                'status': 'Declining',
                'insight': f'Positive sentiment decreased from {prev_positive:.1f}% to {recent_positive:.1f}%',
                'recommendation': 'Investigate recent changes and address emerging issues'
            })
    
    # Print insights
    print("\n💡 Key Insights:\n")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight['category']} - {insight['status']}")
        print(f"   📊 {insight['insight']}")
        print(f"   ✅ {insight['recommendation']}\n")
    
    return insights


def create_visualizations(df, monthly_sentiment_pct, issue_counts):
    """Create visualization charts"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sentiment Trend Over Time
    plt.figure(figsize=(12, 6))
    monthly_sentiment_pct.plot(kind='line', marker='o')
    plt.title('Sentiment Trend Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_trend.png'), dpi=300)
    print(f"✓ Saved: sentiment_trend.png")
    plt.close()
    
    # 2. Issue Distribution
    if issue_counts:
        plt.figure(figsize=(10, 6))
        issues = list(issue_counts.keys())
        counts = list(issue_counts.values())
        
        plt.barh(issues, counts, color='coral')
        plt.title('Common Issues in Negative Feedback', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Mentions', fontsize=12)
        plt.ylabel('Issue Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'issue_distribution.png'), dpi=300)
        print(f"✓ Saved: issue_distribution.png")
        plt.close()
    
    # 3. Rating Distribution
    plt.figure(figsize=(10, 6))
    df['rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Rating Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), dpi=300)
    print(f"✓ Saved: rating_distribution.png")
    plt.close()
    
    # 4. Sentiment Distribution Pie Chart
    plt.figure(figsize=(8, 8))
    sentiment_counts = df['sentiment_label'].value_counts()
    colors = ['#90EE90', '#FFB6C1', '#87CEEB']
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Overall Sentiment Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300)
    print(f"✓ Saved: sentiment_distribution.png")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


def save_insights(insights, predictions, output_path):
    """Save insights and predictions"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'models', output_path)
    
    results = {
        'insights': insights,
        'predictions': predictions,
        'generated_at': datetime.now().isoformat()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Insights saved to: {output_path}")


def main():
    """Main execution function"""
    print("\n" + "#"*100)
    print("#" + " "*98 + "#")
    print("#" + " "*25 + "PART 4 - PREDICTIVE INSIGHT GENERATION" + " "*36 + "#")
    print("#" + " "*98 + "#")
    print("#"*100 + "\n")
    
    # Load data and models
    df, sentiment_model, vectorizer = load_data_and_models()
    
    # Parse dates
    df = parse_dates(df)
    
    # Analyze sentiment trends
    monthly_sentiment, monthly_sentiment_pct = analyze_sentiment_trends(df)
    
    # Predict ratings
    rating_model, model_name, prediction_metrics = predict_rating_from_sentiment(df)
    
    # Detect issues
    issue_counts, sorted_issues = detect_common_issues(df)
    
    # Generate insights
    insights = generate_actionable_insights(df, issue_counts, monthly_sentiment_pct)
    
    # Create visualizations
    create_visualizations(df, monthly_sentiment_pct, issue_counts)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    predictions = {
        'rating_model': model_name,
        'metrics': prediction_metrics,
        'issue_counts': issue_counts
    }
    
    save_insights(insights, predictions, 'predictive_insights.pkl')
    
    # Final summary
    print("\n" + "#"*100)
    print("#" + " "*98 + "#")
    print("#" + " "*30 + "PART 4 COMPLETED SUCCESSFULLY" + " "*38 + "#")
    print("#" + " "*98 + "#")
    print("#"*100)
    
    print("\n✅ DELIVERABLES:")
    print("   ✓ Sentiment trend analysis completed")
    print("   ✓ Rating prediction model trained")
    print("   ✓ Issue detection & categorization done")
    print("   ✓ Actionable insights generated")
    print("   ✓ 4 visualizations created")
    print("   ✓ Predictive insights saved")
    
    print(f"\n📊 KEY FINDINGS:")
    print(f"   Total Insights: {len(insights)}")
    print(f"   Issues Detected: {len(issue_counts)}")
    print(f"   Best Prediction Model: {model_name}")
    print(f"   Visualizations: 4 charts")
    
    print("\n✅ READY FOR PART 5: Deployment\n")


if __name__ == "__main__":
    main()
