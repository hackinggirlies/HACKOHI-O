from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import preprocessing
import torch

def fine_tune_bert():
    """Fine-tune a BERT model on the safety observation data."""
    data = preprocessing.load_and_preprocess_data('sample_data.csv')

    # Tokenize text for BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(data['PNT_ATRISKNOTES_TX']), truncation=True, padding=True, max_length=512)

    # Prepare dataset for training
    class SafetyObservationDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data['PNT_ATRISKNOTES_TX'], data['QUALIFIER_TXT'], test_size=0.2, random_state=42)
    train_dataset = SafetyObservationDataset(train_encodings, y_train.tolist())

    # Fine-tune BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # Assuming 4 unique labels
    training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=16, evaluation_strategy='epoch')
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    
    # Train and evaluate
    trainer.train()

if __name__ == '__main__':
    fine_tune_bert()