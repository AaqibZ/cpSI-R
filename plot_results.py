import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_spread_vs_k(results_csv, outpath='spread_plot.pdf'):
    df = pd.read_csv(results_csv)
    plt.figure(figsize=(6,4))
    for method in df['method'].unique():
        sub = df[df['method']==method]
        plt.plot(sub['k'], sub['spread'], marker='o', label=method)
    plt.xlabel('k (seed set size)')
    plt.ylabel('Average infected fraction')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath)
    plt.close()
