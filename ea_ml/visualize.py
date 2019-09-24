#!/usr/bin/env python3
"""
Created on 9/23/19

@author: dillonshapiro

Functions for visualizing EA-ML results.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set()


def mcc_scatter(results, dpi=150):
    """
    Scatter plot of maxMCC results.

    Args:
        results (DataFrame): Scored results from EA-ML
        dpi (int): Figure resolution

    Returns:
        Figure
    """
    ranks = [i + 1 for i in range(len(results))]

    fig = plt.figure(dpi=dpi)
    plt.scatter(ranks, results.maxMCC, s=12, color='black')
    plt.xlabel('Rank')
    plt.ylabel('MCC')
    plt.tight_layout()
    return fig


def mcc_hist(results, dpi=150):
    """
    Histogram of maxMCC results.

    Args:
        results (DataFrame): Scored results from EA-ML
        dpi (int): Figure resolution

    Returns:
        Figure
    """
    weights = np.zeros_like(results.maxMCC) + 1 / len(results)

    fig = plt.figure(dpi=dpi)
    plt.hist(results.maxMCC, weights=weights, bins=20, color='black')
    plt.xlabel('MCC')
    plt.ylabel('Frequency')
    plt.tight_layout()
    return fig


def visualize(exp_dir, out_dir, prefix='maxMCC', dpi=150):
    results = pd.read_csv(exp_dir / 'maxMCC_summary.csv').sort_values('maxMCC', ascending=False)
    scatter = mcc_scatter(results, dpi=dpi)
    scatter.savefig(out_dir / f'{prefix}-scatter.png')
    hist = mcc_hist(results, dpi=dpi)
    hist.savefig(out_dir / f'{prefix}-hist.png')
