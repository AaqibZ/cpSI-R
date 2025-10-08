import pandas as pd
import os

def load_temporal_edge_list(path, time_col='timestamp', src_col='src', tgt_col='tgt', granularity=20):
    """
    Load a CSV temporal edge list with columns (src,tgt,timestamp).
    Normalize timestamps to integer multiples of granularity.
    Returns pandas DataFrame sorted by time.
    """
    df = pd.read_csv(path)
    # basic cleaning
    df = df[[src_col, tgt_col, time_col]].dropna()
    df.columns = ['src','tgt','timestamp']
    # normalize timestamps to integer multiples of granularity
    df['timestamp'] = (df['timestamp'] // granularity).astype(int) * granularity
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def save_sampled_edges(df, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
