import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import logging
from core.fraudservice import FraudService
from core.datahandler import DataHandler
from core.logging import setup_logging
import time
from pathlib import Path

from watchdog.observers import Observer

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting service...")
    # service = FraudService()

    BASE_DIR = Path(__file__).resolve().parent.parent

    service = FraudService(
        input_dir=BASE_DIR / 'input',
        output_dir=BASE_DIR / 'output',
        model_path=BASE_DIR / 'models' / 'base_fraudmodel.cbm'
    )

    observer = Observer()
    observer.schedule(DataHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logging.info(f"Observer started, watching directory: '{service.input_dir}'")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info('Service stopped by user')
        observer.stop()

    logging.info("Observer stopped.")
    observer.join()
