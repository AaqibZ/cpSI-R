import argparse
import json
import os
import pandas as pd
import numpy as np

from data_prep import load_temporal_edge_list
from sampling import select_time_windows
from calc_influence import greedy_candidate_selection
from baselines import top_k_degree
from cpsir import CpSIRSimulator

def run_one(config):
    # load data
    df = load_temporal_edge_list(config['dataset'], granularity=config.get('time_granularity',20))
    # sample windows
    windows = select_time_windows(df, window_size=config['tau'], step=max(1,config['tau']//2), eta=config.get('eta',0.7))
    print(f"Selected {len(windows)} windows (example starts): {windows[:5]}")
    # merge edges for selected windows (simple approach)
    sel_df = df[df['timestamp'].isin(windows)]
    os.makedirs(config['output_dir'], exist_ok=True)
    # baseline: degree
    for k in config['k_list']:
        deg_seeds = top_k_degree(sel_df, k)
        # simulate baseline
        sim = CpSIRSimulator(sel_df, lambda_p=config['lambda'], mu=config['mu'], alpha=config['alpha'], tau=config['tau'])
        spreads = []
        for r in range(config['runs']):
            res = sim.run(deg_seeds, random_state=42+r)
            spreads.append(res['infected_fraction'])
        mean_spread = np.mean(spreads)
        print(f"Baseline degree k={k}: spread={mean_spread:.4f}")
        # save results minimal
        out = {'method':'degree', 'k':k, 'spread':mean_spread}
        pd.DataFrame([out]).to_csv(os.path.join(config['output_dir'], f"result_degree_k{k}.csv"), index=False)
    # our method: greedy candidate selection
    for k in config['k_list']:
        sim_params = {'lambda':config['lambda'], 'mu':config['mu'], 'alpha':config['alpha'], 'tau':config['tau']}
        seeds, gains = greedy_candidate_selection(sel_df, k, sim_params)
        sim = CpSIRSimulator(sel_df, lambda_p=config['lambda'], mu=config['mu'], alpha=config['alpha'], tau=config['tau'])
        spreads = []
        for r in range(config['runs']):
            res = sim.run(seeds, random_state=100+r)
            spreads.append(res['infected_fraction'])
        mean_spread = np.mean(spreads)
        print(f"cpSI-R greedy k={k}: spread={mean_spread:.4f}")
        out = {'method':'cpsir_greedy', 'k':k, 'spread':mean_spread}
        pd.DataFrame([out]).to_csv(os.path.join(config['output_dir'], f"result_cpsir_k{k}.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='example_config.json')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = json.load(f)
    run_one(config)
