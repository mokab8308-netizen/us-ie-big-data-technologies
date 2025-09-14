# bdt-sentiment.py

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from google.cloud import storage
import io
import torch

def download_blob(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    # Initialize a client without authentication for public buckets
    storage_client = storage.Client.create_anonymous_client()
    
    # Get the bucket and blob
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    # Download the blob as a string
    data = blob.download_as_string()
    
    return data

def calculate_sentiment_accuracy():
    """
    Downloads tweets from Google Cloud Storage, runs sentiment analysis using a high-accuracy Hugging Face pipeline,
    and calculates the accuracy of the pre-trained model against the ground truth.
    """
    # GCS bucket and file details
    bucket_name = "bdt-sentiment"
    blob_name = "tweets-sentiment-synth.csv"

    print("--- Step 1: Loading Data from Google Cloud Storage ---")
    try:
        # Download the CSV file from GCS
        data = download_blob(bucket_name, blob_name)
        
        # Load the CSV data into a pandas DataFrame
        df = pd.read_csv(io.StringIO(data.decode('utf-8')))
        print(f"Successfully loaded {len(df)} tweets from GCS.")
        
        # Debug: Print column names to see what's available
        print(f"Available columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
    except Exception as e:
        print(f"Error: Failed to load data from Google Cloud Storage. {e}")
        return

    # Check if the required columns exist
    if 'Text' not in df.columns:
        print("Error: 'Text' column not found in the dataset.")
        # Try to find similar columns
        text_cols = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower()]
        if text_cols:
            print(f"Found potential text columns: {text_cols}")
            # Use the first potential text column
            text_column = text_cols[0]
        else:
            print("No text-like columns found. Please check the dataset structure.")
            return
    else:
        text_column = 'Text'
    
    if 'sentiment' not in df.columns:
        print("Error: 'sentiment' column not found in the dataset.")
        # Try to find similar columns
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]
        if sentiment_cols:
            print(f"Found potential sentiment columns: {sentiment_cols}")
            # Use the first potential sentiment column
            sentiment_column = sentiment_cols[0]
        else:
            print("No sentiment-like columns found. Please check the dataset structure.")
            return
    else:
        sentiment_column = 'sentiment'

    # Prepare the data for the pipeline and for accuracy calculation
    tweets = df[text_column].tolist()
    ground_truth_labels = df[sentiment_column].tolist()

    print("\n--- Step 2: Initializing High-Accuracy Sentiment Analysis Pipeline ---")
    try:
        # Use a high-accuracy RoBERTa-based model fine-tuned for sentiment analysis
        # This model typically achieves >80% accuracy on sentiment analysis tasks
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Initialize tokenizer and model separately for better control
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create a pipeline with the model and tokenizer
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
            truncation=True,
            padding=True
        )
        
        print(f"Pipeline initialized successfully with high-accuracy model: {model_name}")
        print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
    except Exception as e:
        print(f"Error: Failed to initialize the Hugging Face pipeline. {e}")
        # Fall back to a reliable model if the preferred one fails
        try:
            sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
            print("Fell back to alternative model: siebert/sentiment-roberta-large-english")
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return

    print("\n--- Step 3: Analyzing Tweet Sentiments ---")
    # Run the pipeline on all the tweets. This may take a few moments.
    try:
        # Process tweets in batches to avoid memory issues
        batch_size = 50  # Smaller batch size for the larger model
        predicted_labels = []
        predicted_scores = []
        
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i+batch_size]
            results = sentiment_pipeline(batch)
            
            # Extract labels and confidence scores
            batch_labels = [result['label'] for result in results]
            batch_scores = [result['score'] for result in results]
            
            predicted_labels.extend(batch_labels)
            predicted_scores.extend(batch_scores)
            
            print(f"Processed {min(i+batch_size, len(tweets))}/{len(tweets)} tweets")
            
        print("Sentiment analysis complete.")
        
        # Print average confidence score
        avg_confidence = sum(predicted_scores) / len(predicted_scores)
        print(f"Average confidence score: {avg_confidence:.4f}")
            
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return

    print("\n--- Step 4: Calculating Model Accuracy ---")
    # Normalize the ground truth labels to match the pipeline's output format
    # The twitter-roberta model uses: 'negative', 'neutral', 'positive'
    # Let's check what format our ground truth uses and normalize accordingly
    
    # First, let's see what values we have in ground truth
    unique_truth = set(ground_truth_labels)
    print(f"Unique ground truth values: {unique_truth}")
    
    # Normalize based on what we find
    if all(isinstance(x, (int, float)) for x in ground_truth_labels):
        # If ground truth is numeric, map to sentiment labels
        normalized_ground_truth = []
        for label in ground_truth_labels:
            if label == 0:
                normalized_ground_truth.append('negative')
            elif label == 1:
                normalized_ground_truth.append('positive')
            else:
                normalized_ground_truth.append('neutral')
    else:
        # If ground truth is text, normalize to lowercase
        normalized_ground_truth = [str(label).lower() for label in ground_truth_labels]
    
    # Also normalize predicted labels to ensure consistent formatting
    normalized_predicted = [label.lower() for label in predicted_labels]

    # Calculate the accuracy using scikit-learn's accuracy_score function.
    accuracy = accuracy_score(normalized_ground_truth, normalized_predicted)

    # Print the final result, formatted as a percentage.
    print("\n--- Final Result ---")
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
    
    # Show a sample of predictions vs actual
    print("\n--- Sample Predictions ---")
    for i in range(min(10, len(tweets))):
        print(f"Tweet: {tweets[i][:70]}...")
        print(f"Predicted: {predicted_labels[i]} (score: {predicted_scores[i]:.4f}), Actual: {ground_truth_labels[i]}")
        print()

def main():
    """Main function to invoke the script."""
    calculate_sentiment_accuracy()

if __name__ == "__main__":
    main()
