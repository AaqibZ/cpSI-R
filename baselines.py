import pandas as pd
from collections import Counter
import math

def degree_discount_dynamic(edges_df, k):
    """
    Simple dynamic degree-based heuristic:
    - Count unique neighbors over the whole temporal window and pick top-k degrees.
    This is a placeholder inspired by degree discount methods.
    """
    nodes = set(edges_df['src']).union(set(edges_df['tgt']))
    deg = Counter()
    for r in edges_df.itertuples():
        deg[int(r.src)] += 1
        deg[int(r.tgt)] += 1
    topk = [n for n,_ in deg.most_common(k)]
    return set(topk)

def top_k_degree(edges_df, k):
    return degree_discount_dynamic(edges_df, k)
