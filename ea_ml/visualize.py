#!/usr/bin/env python
"""Functions for visualizing EA-ML results"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set(context='talk', style='ticks')


def mcc_scatter(results, column='maxMCC', dpi=150):
    """
    Scatter plot of maxMCC results.

    Args:
        results (DataFrame): Scored results from EA-ML
        column (str): Score column to plot
        dpi (int): Figure resolution

    Returns:
        Figure
    """
    ranks = [i + 1 for i in range(len(results))]

    fig = plt.figure(dpi=dpi)
    plt.scatter(ranks, results[column], s=12, color='black')
    plt.xlabel('Rank')
    plt.ylabel('MCC')
    plt.tight_layout()
    sns.despine()
    return fig


def mcc_hist(results, column='maxMCC', dpi=150):
    """
    Histogram of maxMCC results.

    Args:
        results (DataFrame): Scored results from EA-ML
        column (str): Score column to plot
        dpi (int): Figure resolution

    Returns:
        Figure
    """
    weights = np.zeros(len(results)) + 1 / len(results)

    fig = plt.figure(dpi=dpi)
    plt.hist(results[column], weights=weights, bins=20, color='black')
    plt.xlabel('MCC')
    plt.ylabel('Frequency')
    plt.tight_layout()
    sns.despine()
    return fig


def visualize(exp_dir, out_dir, prefix='', dpi=150):
    if prefix:
        prefix = prefix + '.'

    for col in ('maxMCC', 'meanMCC'):
        results = pd.read_csv(exp_dir / f'{col}_results.csv').sort_values(col, ascending=False)
        mcc_scatter(results, column=col, dpi=dpi).savefig(out_dir / f'{prefix}{col}-scatter.png')
        mcc_hist(results, column=col, dpi=dpi).savefig(out_dir / f'{prefix}{col}-hist.png')

        stat_results = pd.read_csv(exp_dir / f'{col}_results.nonzero-stats.csv').sort_values(col, ascending=False)
        mcc_scatter(stat_results, column=col, dpi=dpi).savefig(out_dir / f'{prefix}{col}-scatter.nonzero.png')
        mcc_hist(stat_results, column=col, dpi=dpi).savefig(out_dir / f'{prefix}{col}-hist.nonzero.png')
        mcc_scatter(stat_results, column='logMCC', dpi=dpi).savefig(out_dir / f'{prefix}{col}.logMCC-scatter.nonzero.png')
        mcc_hist(stat_results, column='logMCC', dpi=dpi).savefig(out_dir / f'{prefix}{col}.logMCC-hist.nonzero.png')
