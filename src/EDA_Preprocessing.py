import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import os


def load_data(path, chunksize=100_000, max_chunks=None):
    """
    Load a large CSV file in chunks and return a concatenated DataFrame.
    
    Args:
        path (str): Path to the CSV file.
        chunksize (int): Number of rows per chunk. Default is 100,000.
        max_chunks (int or None): Max number of chunks to load (useful for testing). None means load all.
    
    Returns:
        DataFrame: Combined DataFrame from all loaded chunks.
    """
    try:
        chunk_iter = pd.read_csv(path, chunksize=chunksize)
        chunks = []
        for i, chunk in enumerate(chunk_iter):
            chunks.append(chunk)
            # print(f"Loaded chunk {i+1} with shape {chunk.shape}")
            if max_chunks and i + 1 >= max_chunks:
                break

        df = pd.concat(chunks, ignore_index=True)
        print(f" Combined dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    except Exception as e:
        print(f"Error loading file: {e}")
        return None

