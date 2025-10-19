# python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from .models import ClassifierModel
from .preprocessing import preprocess_data
import pandas as pd
from pathlib import Path
import logging
import time
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

class FraudService:
    def __init__(self, input_dir='input', output_dir='output', model_path='models/base_fraudmodel.cbm'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        logging.info("Initializing FraudService...")
        self.fraud_model = ClassifierModel(
            model_path=model_path,
            output_path=output_dir,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("FraudService initialized.")

    def get_predictions(self, df: pd.DataFrame) -> pd.Series:
        logging.info("Starting prediction...")
        start_time = time.time()
        predictions = self.fraud_model.predict(df)
        end_time = time.time()
        logging.info(f"Prediction finished in {end_time - start_time:.2f} seconds.")
        return predictions

    def get_data_from_file(self, filename: Path) -> pd.DataFrame:
        logging.info(f"Reading data from '{filename}'...")
        df = pd.read_csv(filename)
        logging.info(f"Data read successfully. Shape: {df.shape}")
        return df

    def _save_artifacts(self, scores, importances_top5: dict, name: str):
        try:
            json_path = self.output_dir.joinpath(Path(f'importances_{name}.json'))
            logging.info(f"Saving feature importances to '{json_path}'...")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(importances_top5, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error("Failed to save feature importances json: %s", e, exc_info=True)

        try:
            img_path = self.output_dir.joinpath(Path(f'scores_density_{name}.png'))
            logging.info(f"Saving scores density plot to '{img_path}'...")

            arr = np.asarray(scores, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                logging.warning("No finite scores to plot; skipping density plot.")
                return

            plt.figure(figsize=(6, 4), dpi=150)

            plt.hist(arr, bins=50, density=True, color='tab:blue', alpha=0.35, label='hist')
            try:
                if np.unique(arr).size > 1 and arr.size >= 2:
                    kde = gaussian_kde(arr)
                    xs = np.linspace(arr.min(), arr.max(), 200)
                    ys = kde(xs)
                    plt.plot(xs, ys, color='tab:blue', linewidth=2, label='kde')
            except Exception as kde_err:
                logging.warning("KDE failed: %s. Continue with histogram.", kde_err)

            plt.title('Density of predicted scores')
            plt.xlabel('Score')
            plt.ylabel('Density')
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()
        except Exception as e:
            logging.error("Failed to save scores density plot: %s", e, exc_info=True)

    def save_predictions(self, df: pd.DataFrame, name: str = "") -> pd.Series:
        predictions = self.get_predictions(df)
        scores = self.fraud_model.predict_scores(df)
        importances_top5 = self.fraud_model.top_feature(df, top_k=5)

        predictions_df = pd.DataFrame({'index': df.index, 'prediction': predictions})

        output_file = self.output_dir.joinpath(Path(f'predictions_{str(name)}.csv'))
        logging.info(f"Saving predictions to '{output_file}'...")
        start_time = time.time()
        predictions_df.to_csv(output_file, index=False)
        end_time = time.time()
        logging.info(f"Predictions saved in {end_time - start_time:.2f} seconds.")

        self._save_artifacts(scores, importances_top5, name=name)

        return predictions

    def process_data(self, file: Path) -> pd.Series:
        try:
            data = self.get_data_from_file(file)
            preprocessed_data = preprocess_data(data)
            predictions = self.save_predictions(preprocessed_data, name=file.stem)
            return predictions
        except Exception as e:
            logging.error('Error processing file %s: %s', file, e, exc_info=True)
            return