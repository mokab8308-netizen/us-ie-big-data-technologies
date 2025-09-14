# bdt-sentiment.py

import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score

def calculate_sentiment_accuracy():
    """
    Downloads tweets, runs sentiment analysis using a Hugging Face pipeline,
    and calculates the accuracy of the pre-trained model against the ground truth.
    """
    # URL of the dataset
    url = "https://storage.googleapis.com/bdt-sentiment/tweets-sentiment-synth.csv"

    print("--- Step 1: Loading Data ---")
    try:
        # Load the CSV file directly from the URL into a pandas DataFrame
        df = pd.read_csv(url)
        print(f"Successfully loaded {len(df)} tweets.")
        
        # Debug: Print column names to see what's available
        print(f"Available columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
    except Exception as e:
        print(f"Error: Failed to load data from the URL. {e}")
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

    print("\n--- Step 2: Initializing Hugging Face Pipeline ---")
    try:
        # Initialize the sentiment analysis pipeline with a specific model
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        print(f"Pipeline initialized successfully with model: {model_name}")
    except Exception as e:
        print(f"Error: Failed to initialize the Hugging Face pipeline. {e}")
        return

    print("\n--- Step 3: Analyzing Tweet Sentiments ---")
    # Run the pipeline on all the tweets. This may take a few moments.
    try:
        # Process tweets in batches to avoid memory issues
  
      batch_size = 100
        predicted_labels = []
        
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i+batch_size]
            results = sentiment_pipeline(batch)
            predicted_labels.extend([result['label'] for result in results])
            print(f"Processed {min(i+batch_size, len(tweets))}/{len(tweets)} tweets")
            
        print("Sentiment analysis complete.")
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return

    print("\n--- Step 4: Calculating Model Accuracy ---")
    # Normalize the ground truth labels to uppercase to match the pipeline's output.
    normalized_ground_truth = [str(label).upper() for label in ground_truth_labels]
    
    # Also normalize predicted labels to ensure consistent formatting
    normalized_predicted = [label.upper() for label in predicted_labels]

    # Calculate the accuracy using scikit-learn's accuracy_score function.
    accuracy = accuracy_score(normalized_ground_truth, normalized_predicted)

    # Print the final result, formatted as a percentage.
    print("\n--- Final Result ---")
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
    
    # Show a sample of predictions vs actual
    print("\n--- Sample Predictions ---")
    for i in range(min(5, len(tweets))):
        print(f"Tweet: {tweets[i][:50]}...")
        print(f"Predicted: {predicted_labels[i]}, Actual: {ground_truth_labels[i]}")
        print()

def main():
    """Main function to invoke the script."""
    calculate_sentiment_accuracy()

if __name__ == "__main__":
    main()
