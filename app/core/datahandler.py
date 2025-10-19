from watchdog.events import FileSystemEventHandler
from .fraudservice import FraudService
from pathlib import Path
import threading

class DataHandler(FileSystemEventHandler):
    def __init__(self, service: FraudService = None):
        super().__init__()
        self.service = service or FraudService()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            self.service.process_data(Path(event.src_path))