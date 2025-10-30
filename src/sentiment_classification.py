import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

def load_data(filepath):
    """Load preprocessed dataset"""
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    df = pd.read_csv(filepath)
    print(f"\n✓ Dataset loaded: {len(df):,} records")
    print(f"✓ Columns: {list(df.columns)}")
    
    print(f"\nSentiment Distribution:")
    print(df['sentiment_label'].value_counts())
    
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare features and split data"""
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)
    
    # Remove rows with NaN values
    df_clean = df.dropna(subset=['processed_text', 'sentiment_label'])
    print(f"\n✓ Removed {len(df) - len(df_clean)} rows with missing values")
    
    X = df_clean['processed_text'].values
    y = df_clean['sentiment_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n✓ Training set: {len(X_train):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test


def create_tfidf_features(X_train, X_test, max_features=5000):
    """Create TF-IDF features"""
    print("\n" + "="*80)
    print("CREATING TF-IDF FEATURES")
    print("="*80)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"\n✓ Training features: {X_train_tfidf.shape}")
    print(f"✓ Test features: {X_test_tfidf.shape}")
    
    return vectorizer, X_train_tfidf, X_test_tfidf


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*80)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("\n✓ Logistic Regression trained successfully")
    
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print("\n✓ Random Forest trained successfully")
    
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print("\n" + "="*80)
    print(f"{model_name.upper()} - EVALUATION")
    print("="*80)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    print(f"\n📈 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }


def cross_validate_model(model, X_train, y_train, model_name):
    """Perform cross-validation"""
    print(f"\nPerforming 5-Fold Cross-Validation for {model_name}...")
    
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, scoring='f1_weighted', n_jobs=-1
    )
    
    print(f"✓ CV F1 Scores: {cv_scores}")
    print(f"✓ Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores


def save_models(vectorizer, lr_model, rf_model, best_model, metrics, output_dir='../models'):
    """Save all trained models"""
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    # Use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vectorizer
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ TF-IDF Vectorizer: {vectorizer_path}")
    
    # Save Logistic Regression
    lr_path = os.path.join(output_dir, 'logistic_regression_model.pkl')
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"✓ Logistic Regression: {lr_path}")
    
    # Save Random Forest
    rf_path = os.path.join(output_dir, 'random_forest_model.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✓ Random Forest: {rf_path}")
    
    # Save best model as sentiment_model.pkl
    best_model_path = os.path.join(output_dir, 'sentiment_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✓ Best Model (sentiment_model.pkl): {best_model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'model_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Model Metrics: {metrics_path}")
    
    print("\n✓ All models saved successfully!")


def test_sample_predictions(model, vectorizer):
    """Test model on sample texts"""
    print("\n" + "="*80)
    print("TESTING SAMPLE PREDICTIONS")
    print("="*80)
    
    sample_texts = [
        "absolutely amazing experience loved every minute highly recommend",
        "terrible service worst experience ever never coming back",
        "okay nothing special average experience",
        "fantastic product exceeded expectations wonderful",
        "disappointing poor quality waste money"
    ]
    
    sample_tfidf = vectorizer.transform(sample_texts)
    predictions = model.predict(sample_tfidf)
    probabilities = model.predict_proba(sample_tfidf)
    
    for i, (text, pred, probs) in enumerate(zip(sample_texts, predictions, probabilities), 1):
        print(f"\n{i}. Text: '{text}'")
        print(f"   Predicted: {pred}")
        print(f"   Confidence: {max(probs):.2%}")


def main():
    """Main execution function"""
    print("\n" + "#"*100)
    print("#" + " "*98 + "#")
    print("#" + " "*25 + "PART 2 - SENTIMENT CLASSIFICATION MODEL" + " "*33 + "#")
    print("#" + " "*98 + "#")
    print("#"*100 + "\n")
    
    # Load data
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'dataset', 'cleaned_customer_feedback_minimal.csv')
    df = load_data(data_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Create TF-IDF features
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf_features(X_train, X_test)
    
    # Train models
    lr_model = train_logistic_regression(X_train_tfidf, y_train)
    rf_model = train_random_forest(X_train_tfidf, y_train)
    
    # Evaluate models
    lr_metrics = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, X_test_tfidf, y_test, "Random Forest")
    
    # Cross-validation
    print("\n" + "="*80)
    print("CROSS-VALIDATION")
    print("="*80)
    cross_validate_model(lr_model, X_train_tfidf, y_train, "Logistic Regression")
    cross_validate_model(rf_model, X_train_tfidf, y_train, "Random Forest")
    
    # Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [lr_metrics['accuracy'], rf_metrics['accuracy']],
        'Precision': [lr_metrics['precision'], rf_metrics['precision']],
        'Recall': [lr_metrics['recall'], rf_metrics['recall']],
        'F1 Score': [lr_metrics['f1_score'], rf_metrics['f1_score']]
    })
    
    print(f"\n{comparison.to_string(index=False)}")
    
    # Determine best model
    best_model_idx = comparison['F1 Score'].idxmax()
    best_model_name = comparison.loc[best_model_idx, 'Model']
    best_model = rf_model if best_model_name == 'Random Forest' else lr_model
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   F1 Score: {comparison.loc[best_model_idx, 'F1 Score']:.4f}")
    
    # Save models
    metrics_dict = {
        'logistic_regression': lr_metrics,
        'random_forest': rf_metrics,
        'best_model': best_model_name
    }
    save_models(vectorizer, lr_model, rf_model, best_model, metrics_dict)
    
    # Test sample predictions
    test_sample_predictions(best_model, vectorizer)
    
    # Final summary
    print("\n" + "#"*100)
    print("#" + " "*98 + "#")
    print("#" + " "*30 + "PART 2 COMPLETED SUCCESSFULLY" + " "*38 + "#")
    print("#" + " "*98 + "#")
    print("#"*100)
    
    print("\n✅ DELIVERABLES:")
    print("   ✓ Text classification models trained (LR, RF)")
    print("   ✓ Models evaluated with accuracy, precision, recall, F1")
    print("   ✓ Best model saved as sentiment_model.pkl")
    print("   ✓ Cross-validation performed")
    print("   ✓ Sample predictions tested")
    
    print(f"\n📊 BEST MODEL PERFORMANCE:")
    print(f"   Accuracy:  {comparison.loc[best_model_idx, 'Accuracy']:.4f}")
    print(f"   Precision: {comparison.loc[best_model_idx, 'Precision']:.4f}")
    print(f"   Recall:    {comparison.loc[best_model_idx, 'Recall']:.4f}")
    print(f"   F1 Score:  {comparison.loc[best_model_idx, 'F1 Score']:.4f}")
    
    print("\n✅ READY FOR PART 3: Text Summarization\n")


if __name__ == "__main__":
    main()
