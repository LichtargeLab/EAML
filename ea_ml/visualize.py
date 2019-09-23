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
    fig.scatter(ranks, df.maxMCC)
    fig.xlabel('Rank')
    fig.ylabel('MCC')
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
    fig.hist(results.maxMCC, weights=weights)
    fig.xlabel('MCC')
    fig.ylabel('Frequency')
    return fig


def visualize(out_dir, dpi=150):
    results = pd.read_csv(out_dir / 'maxMCC_summary.csv').sort_values('maxMCC', ascending=False)
    scatter = mcc_scatter(results, dpi=dpi)
    scatter.savefig(out_dir / 'maxMCC-scatter.png')
    hist = mcc_hist(results, dpi=dpi)
    hist.savefig(out_dir / 'maxMCC-hist.png')
