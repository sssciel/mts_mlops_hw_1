from catboost import CatBoostClassifier
import logging

class ClassifierModel:
    def __init__(self, model_path: str, output_path: str, threshold: float = 0.95):
        logging.info(f"Loading model from '{model_path}'...")
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.threshold = threshold
        self.output_path = output_path
        logging.info("Model loaded successfully.")

    def predict(self, data):
        probabilities = self.model.predict_proba(data)[:, 1]
        predictions = (probabilities > self.threshold).astype(int)
        return predictions