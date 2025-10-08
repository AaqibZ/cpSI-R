import numpy as np
from cpsir import CpSIRSimulator

def estimate_marginal_gain(edges_df, node, seed_set, sim_params, trials=20):
    """
    Estimate marginal gain of adding `node` to `seed_set` by monte-carlo with small trials.
    This is a simple estimator for candidate selection; not optimized for large graphs.
    """
    base_sim = CpSIRSimulator(edges_df, lambda_p=sim_params['lambda'], mu=sim_params['mu'],
                              alpha=sim_params['alpha'], tau=sim_params['tau'])
    # baseline spread
    b_counts = []
    for i in range(trials):
        res = base_sim.run(seed_set, random_state=i)
        b_counts.append(res['infected_fraction'])
    base_mean = np.mean(b_counts)
    # with node
    w_counts = []
    for i in range(trials):
        res = base_sim.run(set(list(seed_set)+[node]), random_state=100+i)
        w_counts.append(res['infected_fraction'])
    add_mean = np.mean(w_counts)
    return add_mean - base_mean

def greedy_candidate_selection(edges_df, k, sim_params, candidate_nodes=None):
    """
    Simple greedy heuristic: estimate marginal gain for each candidate (or all nodes) and pick top-k.
    """
    if candidate_nodes is None:
        candidate_nodes = sorted(list(set(edges_df['src']).union(set(edges_df['tgt']))))
    gains = {}
    seeds = set()
    for _ in range(k):
        best_node = None
        best_gain = -1
        for v in candidate_nodes:
            if v in seeds: continue
            g = estimate_marginal_gain(edges_df, v, seeds, sim_params, trials=15)
            gains[v]=g
            if g>best_gain:
                best_gain=g
                best_node=v
        if best_node is None:
            break
        seeds.add(best_node)
    return seeds, gains
