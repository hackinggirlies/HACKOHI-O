from sklearn.metrics import classification_report
import model_training

def evaluate_model():
    """Evaluate the trained model."""
    model, X_test, y_test = model_training.train_model()  # This should return your trained model and test data

    # Predict and evaluate
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == '__main__':
    evaluate_model()
