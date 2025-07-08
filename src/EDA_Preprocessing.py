import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
from collections import Counter
import os


def load_data(path, chunksize=100_000, max_chunks=3):
    """
    Loads and displays the first few rows of each chunk from a large CSV file.
    Returns the last chunk loaded as a DataFrame (df).
    
    Args:
        file_path (str): Path to the CSV file.
        chunksize (int): Number of rows per chunk to read.
        max_chunks (int): Number of chunks to display before stopping.
    
    Returns:
        pd.DataFrame: The last loaded chunk (or None if error).
    """
    try:
        df = None
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
            print(f"\n🔹 Chunk {i+1} - First 5 Rows:")
            print(chunk.head())
            
            df = chunk  # Store current chunk

            if i + 1 >= max_chunks:
                break

            del chunk
            gc.collect()

        return df

    except Exception as e:
        print(f"Error reading file: {e}")
        return None