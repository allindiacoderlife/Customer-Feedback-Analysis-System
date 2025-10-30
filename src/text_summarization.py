import pandas as pd
import numpy as np
import pickle
import os
import warnings
from collections import Counter
import re
warnings.filterwarnings('ignore')

# NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)


def load_feedback_data(filepath):
    """Load customer feedback dataset"""
    print("="*80)
    print("LOADING CUSTOMER FEEDBACK DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    print(f"\n✓ Dataset loaded: {len(df):,} records")
    print(f"✓ Columns: {list(df.columns)}")
    
    return df


def preprocess_for_summarization(text):
    """Preprocess text for summarization"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Ensure proper sentence endings
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    text = text.strip()
    return text


def extract_sentences(text):
    """Extract sentences from text"""
    # Use NLTK sentence tokenizer
    sentences = sent_tokenize(text)
    # Clean sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def calculate_sentence_scores_tfidf(sentences):
    """Calculate sentence importance using TF-IDF"""
    if len(sentences) == 0:
        return {}
    
    if len(sentences) == 1:
        return {0: 1.0}
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        lowercase=True
    )
    
    try:
        # Get TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores as sum of TF-IDF values
        sentence_scores = {}
        for idx in range(len(sentences)):
            sentence_scores[idx] = tfidf_matrix[idx].sum()
        
        # Normalize scores
        max_score = max(sentence_scores.values()) if sentence_scores else 1
        if max_score > 0:
            sentence_scores = {k: v/max_score for k, v in sentence_scores.items()}
        
        return sentence_scores
    
    except:
        # Fallback: equal scores
        return {i: 1.0/len(sentences) for i in range(len(sentences))}


def calculate_sentence_scores_frequency(sentences):
    """Calculate sentence importance using word frequency"""
    # Tokenize all sentences
    stop_words = set(stopwords.words('english'))
    
    # Calculate word frequencies
    word_freq = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [w for w in words if w.isalnum() and w not in stop_words]
        word_freq.update(words)
    
    # Normalize frequencies
    max_freq = max(word_freq.values()) if word_freq else 1
    word_freq = {word: freq/max_freq for word, freq in word_freq.items()}
    
    # Calculate sentence scores
    sentence_scores = {}
    for idx, sentence in enumerate(sentences):
        words = word_tokenize(sentence.lower())
        words = [w for w in words if w.isalnum() and w not in stop_words]
        
        if len(words) > 0:
            score = sum(word_freq.get(word, 0) for word in words) / len(words)
            sentence_scores[idx] = score
        else:
            sentence_scores[idx] = 0
    
    return sentence_scores


def generate_extractive_summary(text, num_sentences=2, method='tfidf'):
    """
    Generate extractive summary by selecting top sentences
    
    Args:
        text: Input text to summarize
        num_sentences: Number of sentences in summary
        method: 'tfidf' or 'frequency'
    
    Returns:
        Summary string
    """
    # Preprocess text
    text = preprocess_for_summarization(text)
    
    # Extract sentences
    sentences = extract_sentences(text)
    
    # Handle edge cases
    if len(sentences) == 0:
        return text[:200] + "..." if len(text) > 200 else text
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Calculate sentence scores
    if method == 'tfidf':
        sentence_scores = calculate_sentence_scores_tfidf(sentences)
    else:
        sentence_scores = calculate_sentence_scores_frequency(sentences)
    
    # Select top sentences
    top_indices = sorted(sentence_scores.keys(), 
                        key=lambda i: sentence_scores[i], 
                        reverse=True)[:num_sentences]
    
    # Sort selected sentences by original order
    top_indices = sorted(top_indices)
    
    # Create summary
    summary = ' '.join([sentences[i] for i in top_indices])
    
    return summary


def generate_short_summary(text, method='tfidf'):
    """Generate short summary (1-2 sentences)"""
    return generate_extractive_summary(text, num_sentences=2, method=method)


def generate_detailed_summary(text, method='tfidf'):
    """Generate detailed summary (3-5 sentences)"""
    # Determine optimal number of sentences
    sentences = extract_sentences(text)
    num_sentences = min(5, max(3, len(sentences) // 2))
    
    return generate_extractive_summary(text, num_sentences=num_sentences, method=method)


def summarize_feedback_by_sentiment(df, sentiment, method='tfidf'):
    """Generate summaries for all feedback of a specific sentiment"""
    print(f"\n{'='*80}")
    print(f"SUMMARIZING {sentiment.upper()} FEEDBACK")
    print("="*80)
    
    # Filter by sentiment
    sentiment_df = df[df['sentiment_label'] == sentiment].copy()
    print(f"\n✓ Processing {len(sentiment_df):,} {sentiment} reviews")
    
    # Combine all text for overall summary
    combined_text = ' '.join(sentiment_df['feedback_text'].astype(str).tolist())
    
    # Generate summaries
    short_summary = generate_short_summary(combined_text, method=method)
    detailed_summary = generate_detailed_summary(combined_text, method=method)
    
    print(f"\n📝 Short Summary ({sentiment}):")
    print(f"   {short_summary}")
    
    print(f"\n📄 Detailed Summary ({sentiment}):")
    print(f"   {detailed_summary}")
    
    return {
        'sentiment': sentiment,
        'count': len(sentiment_df),
        'short_summary': short_summary,
        'detailed_summary': detailed_summary
    }


def evaluate_summary_quality(original_text, summary):
    """Evaluate summary quality with basic metrics"""
    original_sentences = extract_sentences(original_text)
    summary_sentences = extract_sentences(summary)
    
    # Calculate metrics
    compression_ratio = len(summary) / len(original_text) if len(original_text) > 0 else 0
    sentence_count = len(summary_sentences)
    avg_sentence_length = len(summary) / sentence_count if sentence_count > 0 else 0
    
    # Calculate coverage (what % of important words are included)
    stop_words = set(stopwords.words('english'))
    
    original_words = set(word_tokenize(original_text.lower()))
    original_words = {w for w in original_words if w.isalnum() and w not in stop_words}
    
    summary_words = set(word_tokenize(summary.lower()))
    summary_words = {w for w in summary_words if w.isalnum() and w not in stop_words}
    
    coverage = len(summary_words.intersection(original_words)) / len(original_words) if len(original_words) > 0 else 0
    
    return {
        'compression_ratio': compression_ratio,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'coverage': coverage
    }


def generate_individual_summaries(df, num_samples=10):
    """Generate summaries for individual feedback samples"""
    print(f"\n{'='*80}")
    print(f"INDIVIDUAL FEEDBACK SUMMARIES (Sample: {num_samples})")
    print("="*80)
    
    # Sample feedback from different sentiments
    samples = []
    for sentiment in ['positive', 'negative', 'neutral']:
        sentiment_df = df[df['sentiment_label'] == sentiment]
        if len(sentiment_df) > 0:
            sample = sentiment_df.sample(min(num_samples // 3, len(sentiment_df)))
            samples.append(sample)
    
    sample_df = pd.concat(samples).reset_index(drop=True)
    
    summaries = []
    for idx, row in sample_df.head(num_samples).iterrows():
        text = str(row['feedback_text'])
        sentiment = row['sentiment_label']
        
        # Generate short summary
        short_summary = generate_short_summary(text)
        
        # Evaluate
        quality = evaluate_summary_quality(text, short_summary)
        
        summaries.append({
            'id': row.get('id', idx),
            'sentiment': sentiment,
            'original_length': len(text),
            'summary_length': len(short_summary),
            'compression_ratio': quality['compression_ratio'],
            'summary': short_summary
        })
        
        if idx < 5:  # Print first 5
            print(f"\n📌 Sample {idx + 1} ({sentiment}):")
            print(f"   Original ({len(text)} chars): {text[:100]}...")
            print(f"   Summary ({len(short_summary)} chars): {short_summary}")
            print(f"   Compression: {quality['compression_ratio']:.1%}")
    
    return pd.DataFrame(summaries)


def save_summaries(summaries_dict, output_path):
    """Save summaries to file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'models', output_path)
    
    with open(output_path, 'wb') as f:
        pickle.dump(summaries_dict, f)
    
    print(f"\n✓ Summaries saved to: {output_path}")


def main():
    """Main execution function"""
    print("\n" + "#"*100)
    print("#" + " "*98 + "#")
    print("#" + " "*30 + "PART 3 - TEXT SUMMARIZATION" + " "*41 + "#")
    print("#" + " "*98 + "#")
    print("#"*100 + "\n")
    
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'dataset', 'cleaned_customer_feedback.csv')
    df = load_feedback_data(data_path)
    
    # Check if feedback_text column exists
    if 'feedback_text' not in df.columns:
        print("\n⚠ 'feedback_text' column not found. Using 'processed_text' instead.")
        df['feedback_text'] = df['processed_text']
    
    # Remove NaN values
    df = df.dropna(subset=['feedback_text', 'sentiment_label'])
    print(f"\n✓ Working with {len(df):,} valid records")
    
    # Generate summaries by sentiment
    print("\n" + "="*80)
    print("GENERATING SENTIMENT-BASED SUMMARIES")
    print("="*80)
    
    sentiment_summaries = {}
    for sentiment in ['positive', 'negative', 'neutral']:
        summary_data = summarize_feedback_by_sentiment(df, sentiment, method='tfidf')
        sentiment_summaries[sentiment] = summary_data
    
    # Generate individual summaries
    individual_summaries = generate_individual_summaries(df, num_samples=10)
    
    # Calculate overall statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    stats = individual_summaries.describe()
    print(f"\n{stats[['original_length', 'summary_length', 'compression_ratio']].to_string()}")
    
    print(f"\n📊 Average Metrics:")
    print(f"   Original Length: {individual_summaries['original_length'].mean():.0f} chars")
    print(f"   Summary Length: {individual_summaries['summary_length'].mean():.0f} chars")
    print(f"   Compression Ratio: {individual_summaries['compression_ratio'].mean():.1%}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING SUMMARIES")
    print("="*80)
    
    all_summaries = {
        'sentiment_summaries': sentiment_summaries,
        'individual_summaries': individual_summaries.to_dict('records'),
        'statistics': {
            'avg_compression': individual_summaries['compression_ratio'].mean(),
            'avg_original_length': individual_summaries['original_length'].mean(),
            'avg_summary_length': individual_summaries['summary_length'].mean()
        }
    }
    
    save_summaries(all_summaries, 'text_summaries.pkl')
    
    # Export to CSV
    csv_path = os.path.join(script_dir, '..', 'dataset', 'individual_summaries.csv')
    individual_summaries.to_csv(csv_path, index=False)
    print(f"✓ Individual summaries saved to: {csv_path}")
    
    # Final summary
    print("\n" + "#"*100)
    print("#" + " "*98 + "#")
    print("#" + " "*30 + "PART 3 COMPLETED SUCCESSFULLY" + " "*38 + "#")
    print("#" + " "*98 + "#")
    print("#"*100)
    
    print("\n✅ DELIVERABLES:")
    print("   ✓ Extractive summarization implemented (TF-IDF)")
    print("   ✓ Short summaries generated (1-2 sentences)")
    print("   ✓ Detailed summaries generated (3-5 sentences)")
    print("   ✓ Sentiment-based summaries created")
    print("   ✓ Individual feedback summaries generated")
    print("   ✓ Quality metrics calculated")
    
    print(f"\n📊 SUMMARIZATION PERFORMANCE:")
    print(f"   Average Compression: {individual_summaries['compression_ratio'].mean():.1%}")
    print(f"   Summaries Generated: {len(individual_summaries)}")
    print(f"   Sentiment Groups: {len(sentiment_summaries)}")
    
    print("\n✅ READY FOR PART 4: Predictive Insight Generation\n")


if __name__ == "__main__":
    main()
