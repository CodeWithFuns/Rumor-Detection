import os
import pandas as pd
from typing import Tuple

def load_dataset(path: str, text_col: str = "text", label_col: str = "label") -> Tuple[pd.Series, pd.Series]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Columns not found. Have={list(df.columns)}, need text_col='{text_col}', label_col='{label_col}'")
    return df[text_col].astype(str), df[label_col].astype(str)
