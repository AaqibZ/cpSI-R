import pandas as pd
import numpy as np

def jaccard_similarity(setA, setB):
    if len(setA)==0 and len(setB)==0:
        return 1.0
    inter = len(setA.intersection(setB))
    uni = len(setA.union(setB))
    return inter/uni if uni>0 else 0.0

def kulczynski_similarity(setA, setB):
    # Kulczynski index: (|A∩B|/|A| + |A∩B|/|B|) / 2 (symmetric)
    if len(setA)==0 or len(setB)==0:
        return 0.0
    inter = len(setA.intersection(setB))
    return 0.5 * (inter/len(setA) + inter/len(setB))

def select_time_windows(df, window_size, step, eta=0.7, alpha=0.5, beta=0.5):
    """
    A simple temporal sampling procedure:
    - Partition timeline into windows of length window_size (seconds).
    - Represent each window as set of edges (unordered pairs).
    - Compute pairwise similarity, select top fraction eta of windows by diversity.
    Returns list of selected windows (start_times).
    """
    # map timestamp -> window index
    times = df['timestamp'].unique()
    start = df['timestamp'].min()
    end = df['timestamp'].max()
    windows = []
    t = start
    while t <= end:
        win_df = df[(df['timestamp'] >= t) & (df['timestamp'] < t+window_size)]
        edge_sets = set([tuple(sorted((int(r.src), int(r.tgt)))) for r in win_df.itertuples()])
        windows.append({'start': t, 'edges': edge_sets, 'count':len(edge_sets)})
        t += step
    # compute a simple diversity score: average similarity to others (lower is more diverse)
    scores = []
    for i,w in enumerate(windows):
        sims = []
        for j,u in enumerate(windows):
            if i==j: continue
            s_j = alpha * jaccard_similarity(w['edges'], u['edges']) + beta * kulczynski_similarity(w['edges'], u['edges'])
            sims.append(s_j)
        mean_sim = np.mean(sims) if sims else 0.0
        scores.append((i, mean_sim, w['start']))
    # lower mean_sim means more unique window; keep fraction eta of most unique windows
    scores_sorted = sorted(scores, key=lambda x: x[1])
    keep = int(np.ceil(len(scores_sorted)*eta))
    selected = [windows[scores_sorted[i][0]]['start'] for i in range(keep)]
    return selected
