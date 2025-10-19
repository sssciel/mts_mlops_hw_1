import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from .models import ClassifierModel
from .preprocessing import preprocess_data
import pandas as pd
from pathlib import Path
import logging
import time
import os

class FraudService:
    def __init__(self, input_dir='input', output_dir='output', model_path='models/base_fraudmodel.cbm'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        logging.info("Initializing FraudService...")
        self.fraud_model = ClassifierModel(
            model_path=model_path,
            output_path=output_dir,
        )
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

    def save_predictions(self, df: pd.DataFrame, name: str = "") -> pd.Series:
        predictions = self.get_predictions(df)
        predictions_df = pd.DataFrame({'index': df.index, 'prediction': predictions})

        output_file = self.output_dir.joinpath(Path(f'predictions_{str(name)}.csv'))
        logging.info(f"Saving predictions to '{output_file}'...")
        start_time = time.time()
        predictions_df.to_csv(output_file, index=False)

        end_time = time.time()
        logging.info(f"Predictions saved in {end_time - start_time:.2f} seconds.")
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
