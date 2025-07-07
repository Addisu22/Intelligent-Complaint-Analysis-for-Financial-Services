import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import os

def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f" Error loading file: {e}")
        return None

df = load_dataset("Data/complaints.csv")  # update with your actual file path
