import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split

# Load your data (replace with your actual data file path)
df = pd.read_csv('sample_data.csv')  # Example CSV

# Assuming 'label' is the target column
X = df['PNT_ATRISKNOTES_TX']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a text classification pipeline
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name)

# Fine-tuning the model on the training data
# Note: You can directly use the pipeline without explicit training for inference
for comment, label in zip(X_train, y_train):
    classifier(comment, return_all_scores=True)  # To include it in the training loop

# Evaluate on the test set
predictions = classifier(X_test)
predicted_labels = [1 if p['label'] == 'LABEL_1' else 0 for p in predictions]  # Adjust based on your label mapping

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

print('Training finished!')

# Save the trained model (if you did fine-tune)
classifier.save_pretrained('safety_comment_model')
