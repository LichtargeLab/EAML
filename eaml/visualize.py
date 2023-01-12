#!/usr/bin/env python
"""Functions for visualizing EAML results"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

sns.set_theme(context='talk', style='ticks')
plt.rcParams['pdf.fonttype'] = 42  # so text is exported correctly to Illustrator


def mcc_scatter(results, column='MCC', fig_params=None):
    """
    Scatterplot of maxMCC results

    Args:
        results (DataFrame): Scored results from EAML
        column (str): Score column to plot
        fig_params (dict): Parameters for customizing figure

    Returns:
        Figure
    """
    with sns.axes_style('ticks', rc=fig_params), sns.plotting_context('talk', rc=fig_params):
        ranks = [i + 1 for i in range(len(results))]

        fig = plt.figure()
        plt.scatter(ranks, results[column], s=12, color='black')
        plt.xlabel('Rank')
        plt.ylabel('MCC')
        plt.tight_layout()
        sns.despine()
        return fig


def mcc_hist(results, column='MCC', fig_params=None):
    """
    Histogram of MCC scores

    Args:
        results (DataFrame): Scored results from EAML
        column (str): Score column to plot
        fig_params (dict): Parameters for customizing figure

    Returns:
        Figure
    """
    with sns.axes_style('ticks', rc=fig_params), sns.plotting_context('talk', rc=fig_params):
        weights = np.zeros(len(results)) + 1 / len(results)

        fig = plt.figure()
        plt.hist(results[column], weights=weights, bins=20, color='black')
        plt.xlabel('MCC')
        plt.ylabel('Frequency')
        plt.tight_layout()
        sns.despine()
        return fig


def downsample_enrichment_plot(hypergeom_df, fig_params=None):
    with sns.axes_style('ticks', rc=fig_params), sns.plotting_context('talk', rc=fig_params):
        fig = plt.figure()
        plt.plot(hypergeom_df.index, -np.log10(hypergeom_df['hypergeometric_pvalue']))
        plt.hlines(-np.log10(0.05), 0, np.max(hypergeom_df.index))
        plt.xticks(hypergeom_df.index, rotation=45)
        for n, row in hypergeom_df.iterrows():
            label = f'{row.mean_overlap} / {row.mean_predictions}'
            plt.annotate(label, (n, -np.log10(row.hypergeometric_pvalue)), textcoords='offset points', xytext=(0, 10),
                         ha='center', fontsize=10)
        plt.xlabel('Number of samples')
        plt.ylabel('-log10(Hypergeometric p-value)')
        sns.despine()
        plt.tight_layout()
        return fig


def manhattan_plot(mcc_df, reference, fig_params=None):
    """
    Generates a Manhattan plot, given DataFrames of MCC rankings and gene positions

    Args:
        mcc_df (pd.DataFrame): Scored results from EAML, must have column corresponding to p-value, indexed by gene
        reference (pd.DataFrame): RefGene-formatted reference info for only tested genes,
            including chromosome and positions
        fig_params (dict): Parameters for customizing figure

    Returns:
        Figure

    Note: points close together may have overlapping labels (can be modified in Illustrator/Inkscape)
    """
    with sns.axes_style('ticks', rc=fig_params), sns.plotting_context('talk', rc=fig_params):
        reference = reference.loc[mcc_df.index]
        reference.loc[reference.chrom == 'X'] = 23
        reference.loc[reference.chrom == 'Y'] = 24
        reference = reference.astype({'chrom': int})
        reference.sort_values(['chrom', 'start'], inplace=True)
        reference['pos'] = range(len(reference))

        fig, ax = plt.subplots()
        colors = ['black', 'grey']
        ticks = []
        for i, chrom in enumerate(reference.chrom.unique()):
            chr_df = reference.loc[reference.chrom == chrom]
            xmin = chr_df.pos.min() + (300 * i)
            xmax = chr_df.pos.max() + (300 * i)
            ticks.append((xmin + xmax) / 2)
            ax.scatter(chr_df['pos'] + (300 * i), -np.log10(mcc_df.loc[chr_df.index.unique(), 'pvalue']),
                       alpha=0.7, s=18, color=colors[i % len(colors)])

        def _label_point(g):
            gene_ref = reference.loc[g]
            x = gene_ref.pos + (gene_ref.chrom - 1) * 300
            y = -np.log10(mcc_df.loc[g, 'pvalue'])
            return ax.text(x, y, str(g), fontsize=8)

        # top gene annotation
        fdr_cutoff = mcc_df.loc[mcc_df.qvalue <= 0.1, 'pvalue'].max()
        texts = [_label_point(gene) for gene in mcc_df.loc[mcc_df.pvalue <= fdr_cutoff].index]
        ax.axhline(-np.log10(fdr_cutoff), ls='--', color='black')

        ax.set_xlabel('Chromosome', fontsize=16)
        ax.set_ylabel('-log10(p-value)', fontsize=16)
        ax.set_xticks(ticks)
        ax.set_xticklabels(['X' if chrom == 23 else 'Y' if chrom == 24 else str(chrom)
                            for chrom in reference.chrom.unique()])
        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylim(0, -np.log10(mcc_df.pvalue.min()) + 1)
        fig.tight_layout()
        sns.despine()
        adjust_text(texts)
        return fig
