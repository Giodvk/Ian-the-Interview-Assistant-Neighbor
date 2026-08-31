import pandas as pd 
import numpy as np 
from pathlib import Path

DATASET_PATH = "data/interview_forge_v3_complete.csv"

def load_dataset(dataset_path: Path):

    if not dataset_path.exists():
        raise ValueError(f"The provided path : {dataset_path} doesn't exists")

    return pd.DataFrame(dataset_path)

