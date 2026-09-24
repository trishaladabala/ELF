import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

def plot_margin_phase(steps, margins, rhos, out_path):
    setup_style()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Denoiser Step')
    ax1.set_ylabel('Decoder Margin ($m_i$)', color=color)
    ax1.plot(steps, margins, color=color, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel(r'Normalized Margin ($\rho_i$)', color=color)
    ax2.plot(steps, rhos, color=color, marker='s', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Phase Diagram: Margin and Sensitivity')
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_basin_histogram(entry_times, out_path):
    setup_style()
    plt.figure(figsize=(8, 5))
    sns.histplot(entry_times, bins=20, kde=True, color='purple')
    plt.xlabel('Basin Entry Step')
    plt.ylabel('Token Count')
    plt.title('Per-Token Basin Entry Staggering')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_ppl_entropy_frontier(methods, out_path):
    """
    methods: dict mapping method_name to list of dicts: {'cfg': val, 'ppl': val, 'entropy': val}
    """
    setup_style()
    plt.figure(figsize=(8, 6))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    for idx, (name, data) in enumerate(methods.items()):
        df = pd.DataFrame(data)
        if df.empty: continue
        df = df.sort_values('entropy')
        plt.plot(df['entropy'], df['ppl'], marker=markers[idx % len(markers)], label=name, linewidth=2, markersize=8)
        
    plt.xlabel('Entropy (Diversity)')
    plt.ylabel('Generative Perplexity (Quality)')
    plt.title('PPL–Entropy Frontier (Sweeping CFG)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_nfe_vs_quality(steps_list, bleus, baseline_bleu, oracle_bleu, out_path):
    setup_style()
    plt.figure(figsize=(8, 5))
    
    nfes = [64 - s for s in steps_list]
    plt.plot(nfes, bleus, marker='o', color='green', linewidth=2, label='Early-State Decoder')
    plt.axhline(baseline_bleu, color='red', linestyle='--', label='Baseline (64 NFE)')
    plt.axhline(oracle_bleu, color='gold', linestyle='--', label='Oracle Ceiling')
    
    plt.xlabel('Number of Function Evaluations (NFE)')
    plt.ylabel('BLEU Score')
    plt.title('NFE vs Quality (Early Exit)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_error_taxonomy(taxonomy_data, out_path):
    """taxonomy_data is a dict of error categories and counts"""
    setup_style()
    plt.figure(figsize=(8, 5))
    
    categories = list(taxonomy_data.keys())
    counts = list(taxonomy_data.values())
    
    sns.barplot(x=counts, y=categories, palette="viridis", hue=categories, legend=False)
    plt.xlabel('Error Count')
    plt.title('Error Taxonomy (Native vs Oracle)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
