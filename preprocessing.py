import pandas as pd
import re

def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Remove special characters
    return text.lower()

def load_and_preprocess_data(filename):
    """Load and preprocess the safety observation data."""
    df = pd.read_csv(filename)
    df['cleaned_notes'] = df['PNT_ATRISKNOTES_TX'].apply(clean_text)  # Clean risk notes
    return df

if __name__ == '__main__':
    data = load_and_preprocess_data('sample_data.csv')
    print(data.head())
