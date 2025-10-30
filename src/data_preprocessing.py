import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')


def download_nltk_resources():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"✓ Downloaded: {resource}")
        except Exception as e:
            print(f"✗ Error downloading {resource}: {e}")


def clean_text(text):
    """
    Clean text by removing URLs, emails, special characters, etc.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def tokenize_text(text):
    """Tokenize text into words"""
    try:
        return word_tokenize(text)
    except:
        return text.split()


def remove_stopwords(tokens, stop_words):
    """Remove stopwords from tokenized text"""
    return [word for word in tokens if word not in stop_words and len(word) > 2]


def lemmatize_tokens(tokens, lemmatizer):
    """Lemmatize tokens"""
    return [lemmatizer.lemmatize(word) for word in tokens]


def preprocess_dataset(input_path, output_path):
    """
    Main preprocessing pipeline
    """
    print("="*80)
    print("PART 1 - DATA PREPROCESSING")
    print("="*80)
    
    # Initialize tools
    print("\nInitializing NLP tools...")
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Load dataset
    print(f"\nLoading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✓ Dataset loaded: {len(df):,} records")
    
    # Initial statistics
    print(f"\nInitial dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Data Cleaning
    print("\n" + "-"*80)
    print("STEP 1: DATA CLEANING")
    print("-"*80)
    
    # Handle missing values
    print("\nHandling missing values...")
    before = len(df)
    df = df.dropna(subset=['feedback_text'])
    df['feedback_text'] = df['feedback_text'].astype(str)
    after = len(df)
    print(f"✓ Rows dropped: {before - after}")
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=['feedback_text'], keep='first')
    after = len(df)
    print(f"✓ Duplicates removed: {before - after}")
    
    # Remove short feedback
    print("\nRemoving short feedback (< 3 words)...")
    before = len(df)
    df['temp_word_count'] = df['feedback_text'].apply(lambda x: len(str(x).split()))
    df = df[df['temp_word_count'] >= 3]
    df = df.drop('temp_word_count', axis=1)
    after = len(df)
    print(f"✓ Short feedback removed: {before - after}")
    
    print(f"\n✓ Cleaned dataset size: {len(df):,} records")
    
    # Text Preprocessing
    print("\n" + "-"*80)
    print("STEP 2: TEXT PREPROCESSING")
    print("-"*80)
    
    print("\nApplying text preprocessing pipeline...")
    print("This may take a few minutes...\n")
    
    # Clean text
    print("1/4 Cleaning text...")
    df['cleaned_text'] = df['feedback_text'].apply(clean_text)
    print("✓ Text cleaning completed")
    
    # Tokenize
    print("2/4 Tokenizing...")
    df['tokens'] = df['cleaned_text'].apply(tokenize_text)
    print("✓ Tokenization completed")
    
    # Remove stopwords
    print("3/4 Removing stopwords...")
    df['tokens_no_stopwords'] = df['tokens'].apply(lambda x: remove_stopwords(x, stop_words))
    print("✓ Stopwords removed")
    
    # Lemmatize
    print("4/4 Lemmatizing...")
    df['lemmatized_tokens'] = df['tokens_no_stopwords'].apply(lambda x: lemmatize_tokens(x, lemmatizer))
    print("✓ Lemmatization completed")
    
    # Create processed text
    df['processed_text'] = df['lemmatized_tokens'].apply(lambda x: ' '.join(x))
    
    # Add statistics
    df['token_count'] = df['lemmatized_tokens'].apply(len)
    df['word_count'] = df['feedback_text'].apply(lambda x: len(str(x).split()))
    df['feedback_length'] = df['feedback_text'].apply(len)
    
    # Final dataset
    print("\n" + "-"*80)
    print("STEP 3: CREATING FINAL DATASET")
    print("-"*80)
    
    final_columns = [
        'id', 'source', 'date', 'feedback_text', 'cleaned_text', 
        'processed_text', 'lemmatized_tokens', 'rating', 
        'sentiment_label', 'token_count', 'word_count', 'feedback_length'
    ]
    
    # Select columns that exist
    available_columns = [col for col in final_columns if col in df.columns]
    df_final = df[available_columns].copy()
    df_final = df_final.reset_index(drop=True)
    
    # Save dataset
    print(f"\nSaving cleaned dataset to: {output_path}")
    df_to_save = df_final.copy()
    df_to_save['lemmatized_tokens'] = df_to_save['lemmatized_tokens'].apply(lambda x: ' '.join(x))
    df_to_save.to_csv(output_path, index=False)
    print(f"✓ Dataset saved: {len(df_final):,} records")
    
    # Save minimal version
    minimal_columns = ['id', 'processed_text', 'sentiment_label', 'rating']
    minimal_cols = [col for col in minimal_columns if col in df_final.columns]
    if minimal_cols:
        df_minimal = df_final[minimal_cols].copy()
        minimal_path = output_path.replace('.csv', '_minimal.csv')
        df_minimal.to_csv(minimal_path, index=False)
        print(f"✓ Minimal dataset saved: {minimal_path}")
    
    # Final report
    print("\n" + "="*80)
    print("FINAL DATA QUALITY REPORT")
    print("="*80)
    print(f"\n✓ Total Records: {len(df_final):,}")
    print(f"✓ Features: {len(df_final.columns)}")
    print(f"✓ Missing Values: {df_final.isnull().sum().sum()}")
    
    # Check duplicates on non-list columns only
    non_list_cols = [col for col in df_final.columns if col != 'lemmatized_tokens']
    dup_count = df_final[non_list_cols].duplicated().sum() if non_list_cols else 0
    print(f"✓ Duplicate Records: {dup_count}")
    
    if 'sentiment_label' in df_final.columns:
        print(f"\n✓ Sentiment Distribution:")
        print(df_final['sentiment_label'].value_counts())
    
    print(f"\n✓ Average Feedback Length: {df_final['feedback_length'].mean():.2f} characters")
    print(f"✓ Average Word Count: {df_final['word_count'].mean():.2f} words")
    print(f"✓ Average Token Count: {df_final['token_count'].mean():.2f} tokens")
    
    print("\n" + "="*80)
    print("✓ PART 1 COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return df_final


if __name__ == "__main__":
    # File paths
    INPUT_FILE = "../dataset/dataset.csv"
    OUTPUT_FILE = "../dataset/cleaned_customer_feedback.csv"
    
    # Run preprocessing
    df_final = preprocess_dataset(INPUT_FILE, OUTPUT_FILE)
    
    print("\n✅ Ready for Part 2: Sentiment Classification Model\n")
