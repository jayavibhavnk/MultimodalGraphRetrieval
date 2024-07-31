import json
import logging
import os
from joblib import Parallel, delayed

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parallel_process(func, iterable, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in iterable)
