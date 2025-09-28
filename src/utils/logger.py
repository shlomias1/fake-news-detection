import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime

from src.config import LOG_DIR

def _create_log(log_msg, log_type, log_file = "logs.txt"):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_file)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as log:
        log.write(f'{log_type} : {log_msg} | {current_time} \n')