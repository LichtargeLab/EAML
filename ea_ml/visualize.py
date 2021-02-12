#!/usr/bin/env python
"""Functions for visualizing EA-ML results"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(context='talk', style='ticks')


def mcc_scatter(results, column='MCC', dpi=150):
    """
    Scatterplot of maxMCC results

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


def mcc_hist(results, column='MCC', dpi=150):
    """
    Histogram of MCC scores

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


def manhattan_plot(mcc_df, reference, dpi=300):
    """
    Generates a Manhattan plot, given DataFrames of MCC rankings and gene positions

    Args:
        mcc_df (pd.DataFrame): Scored results from EA-ML, must have column corresponding to p-value, indexed by gene
        reference (pd.DataFrame): RefGene-formatted reference info for only tested genes,
            including chromosome and positions
        dpi (int): Figure resolution

    Returns:
        Figure

    Note: points close together may have overlapping labels (can be modified in Illustrator/Inkscape)
    """
    reference = reference[~reference.index.duplicated()].loc[mcc_df.index]
    reference['chrom'] = reference['chrom'].str.strip('chr').astype(int)
    reference.sort_values(['chrom', 'cdsStart'], inplace=True)
    reference['pos'] = range(len(reference))
    fdr_cutoff = mcc_df.loc[mcc_df.qvalue <= 0.1, 'pvalue'].max()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    colors = ['black', 'grey']
    ticks = []
    for i, chrom in enumerate(np.sort(reference.chrom.unique())):
        chr_df = reference.loc[reference.chrom == chrom]
        xmin = chr_df.pos.min() + (300 * i)
        xmax = chr_df.pos.max() + (300 * i)
        ticks.append((xmin + xmax) / 2)
        ax.scatter(chr_df['pos'] + (300 * i), -np.log10(mcc_df.loc[chr_df.index.unique(), 'pvalue']),
                   alpha=0.7, s=18, color=colors[i % len(colors)])

    def _label_point(g):
        gene_ref = reference.loc[g]
        x = gene_ref.pos + (gene_ref.chrom - 1) * 300
        y = -np.log10(mcc_df.loc[gene, 'pvalue'])
        ax.annotate(str(g), (x, y), bbox=bbox_props, textcoords='offset points', xytext=(0, 10), ha='center',
                    fontsize=8)
    # top gene annotation
    bbox_props = dict(boxstyle='round', fc='w', ec='0.5')
    if len(mcc_df.loc[mcc_df.qvalue <= 0.1]) < 30:
        for gene in mcc_df.loc[mcc_df.pvalue <= fdr_cutoff].index:
            _label_point(gene)
    else:
        for gene in mcc_df.nsmallest(30, 'pvalue').index:
            _label_point(gene)
    ax.axhline(-np.log10(fdr_cutoff), ls='--', color='black')
    ax.set_xlabel('Chromosome', fontsize=16)
    ax.set_ylabel(r'$-log_{10}$(p-value)', fontsize=16)
    ax.set_xticks(ticks)
    ax.set_xticklabels([chrom for chrom in reference.chrom.unique()])
    ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylim(0, -np.log10(mcc_df.pvalue.min()) + 1)
    fig.tight_layout()
    sns.despine()
    return fig
