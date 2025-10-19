# python
from catboost import CatBoostClassifier, Pool
import logging
import pandas as pd

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

    def predict_scores(self, data):
        return self.model.predict_proba(data)[:, 1]

    def top_feature(self, data: pd.DataFrame | None = None, top_k: int = 5):
        try:
            if data is not None:
                cat_cols = list(data.select_dtypes(include=["object", "category"]).columns)
                pool = Pool(data, cat_features=cat_cols if cat_cols else None)
                importances = self.model.get_feature_importance(pool, type="PredictionValuesChange")
                feature_names = list(data.columns)
            else:
                importances = self.model.get_feature_importance(type="FeatureImportance")
                feature_names = getattr(self.model, "feature_names_", [f"feature_{i}" for i in range(len(importances))])

            pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_k]
            return {name: float(val) for name, val in pairs}
        except Exception as e:
            logging.error("Failed to compute feature importances: %s", e, exc_info=True)
            return {}