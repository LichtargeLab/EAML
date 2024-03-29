#!/usr/bin/env python
"""Module for running downsampling experiments to estimate statistical power in comparison to alternative association
methods.

Note: This can be a long-running process, depending on the dataset size.
"""
from collections import defaultdict

import pandas as pd
from scipy.stats import hypergeom
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .pipeline import Pipeline, compute_stats
from .visualize import downsample_enrichment_plot
from .weka import eval_gene


class DownsamplingPipeline(Pipeline):
    """The main pipeline object for downsampling.

    Inherits from `Pipeline` class.

    Args:
        expdir (Path-like): Filepath to experiment directory
        data_fn (Path-like): Filepath to input VCF or pre-computed design matrix
        targets_fn (Path-like): Filepath to two-column CSV with sample IDs and disease status
        true_results_fn (Path-like): Filepath to final results CSV from full cohort experiment
        sample_sizes (list[int]): Sample sizes to test
        nrepeats (int): Number of replicates to perform for each sample size (more = longer compute time)
        reference (str/Path-like): Genome reference version or custom file (hg19, hg38, GRCh37, GRCh38)
        cpus (int): Number of CPUs to use
        kfolds (int): Number of cross-validation folds for each classifier
        seed (int): Random seed for cross-validation
        weka_path (Path-like): Filepath to Weka installation
        min_af (float): Sets minimum allele frequency threshold for including variants
        max_af (float): Sets maximum allele frequency threshold for including variants
        af_field (str): VCF field with allele frequency
        include_X (bool): Includes X chromosome in analysis
        parse_EA (str): How to parse EA scores when there are multiple transcripts (max, mean, all, canonical).
            Default is to only use the predefined 'canonical' transcript.
        memory (str): Memory argument for JVM used by Weka
        annotation (str): Variant annotation software used (ANNOVAR, VEP)
    """
    def __init__(self, expdir, data_fn, targets_fn, true_results_fn, sample_sizes, nrepeats=10, reference='hg19',
                 cpus=1, kfolds=10, seed=111, weka_path='~/weka', min_af=None, max_af=None, af_field='AF',
                 include_X=False, parse_EA='canonical', memory='Xmx2g', annotation='ANNOVAR'):
        """Initializes DownsamplingPipeline from input data"""
        super().__init__(expdir, data_fn, targets_fn, reference=reference, cpus=cpus, kfolds=kfolds, seed=seed,
                         weka_path=weka_path, min_af=min_af, max_af=max_af, af_field=af_field, include_X=include_X,
                         parse_EA=parse_EA, memory=memory, annotation=annotation)
        self.sample_sizes = sample_sizes
        self.n_repeats = nrepeats
        self.true_results = pd.read_csv(true_results_fn, index_col='gene')
        self.write_data = False

    def eval_gene(self, gene):
        """Parse input data for a given gene and evaluate with Weka repeatedly for each sample size"""
        # first check if gene was scored by whole cohort run
        if gene in self.true_results.index:
            sampled_results = defaultdict(list)
            if self.data_fn.is_dir():
                gene_dmatrix = pd.read_csv(self.data_fn / f'{gene}.csv', index_col=0)
            else:
                gene_dmatrix = self.compute_gene_dmatrix(gene)
            for sample_size in self.sample_sizes:
                splits = downsample_gene(gene_dmatrix, self.targets, sample_size, n_splits=self.n_repeats)
                for split_idx, _ in splits:
                    sub_X = gene_dmatrix.iloc[split_idx]
                    sub_y = self.targets.iloc[split_idx]
                    mcc_results = eval_gene(gene, sub_X, sub_y, self.class_params, seed=self.seed, cv=self.kfolds,
                                            expdir=self.expdir, weka_path=self.weka_path, memory=self.weka_mem)
                    sampled_results[sample_size].append(mcc_results)
                    (self.expdir / f'tmp/{gene}.arff').unlink()
            return gene, sampled_results
        else:
            return None

    def report_results(self):
        """Computes results file for each sampled experiment"""
        # lots of nesting here, is there a better way to do this?
        self.nonzero_results = {n: [] for n in self.sample_sizes}
        for n in self.sample_sizes:
            subdir = self.expdir / str(n)
            subdir.mkdir()
            for i in range(self.n_repeats):
                mcc_df_dict = defaultdict(list)
                for gene, sampled_results in self.raw_results:
                    mcc_df_dict['gene'].append(gene)
                    for clf, mcc in sampled_results[n][i].items():
                        mcc_df_dict[clf].append(mcc)
                mcc_df = pd.DataFrame(mcc_df_dict).set_index('gene')
                clfs = mcc_df.columns
                mcc_df['mean'] = mcc_df.mean(axis=1)
                mcc_df['std'] = mcc_df[clfs].std(axis=1)
                mcc_df.sort_values('mean', ascending=False, inplace=True)
                nonzero_df = compute_stats(mcc_df)
                self.nonzero_results[n].append(nonzero_df)
                nonzero_df.to_csv(subdir / f'meanMCC_results.nonzero-stats.{i}.csv')
        self.hypergeometric_results = self.hypergeometric_overlap()
        self.hypergeometric_results.to_csv(self.expdir / 'hypergeometric_results.csv')

    def hypergeometric_overlap(self):
        """Calculates hypergeometric overlap with true results at each sample size.

        For each sample size, we calculate the average number of genes identified across replicates and the average
        overlap between each replicate and the true results. These are then compared to number of genes identified in
        the true results and the total number of genes tested using a hypergeometric test.

        Returns:
            DataFrame: Summary of average overlap and hypergeometric enrichment for each sample size
        """
        true_results = self.true_results
        true_preds = set(true_results.loc[(true_results['MCC'] > 0) & (true_results['qvalue'] < 0.1)].index)
        hypergeom_pvalues, mean_overlaps, mean_preds = [], [], []
        for n, results_df_l in self.nonzero_results.items():
            n_overlaps, n_preds = [], []
            for results_df in results_df_l:
                sample_preds = set(results_df.loc[(results_df['MCC'] > 0) & (results_df['qvalue'] < 0.1)].index)
                n_overlaps.append(len(sample_preds & true_preds))
                n_preds.append(len(sample_preds))
            mean_overlap = np.mean(n_overlaps)
            mean_pred = np.mean(n_preds)
            mean_overlaps.append(mean_overlap)
            mean_preds.append(mean_pred)
            hypergeom_pvalues.append(hypergeom.sf(mean_overlap - 1, len(true_results), len(true_preds), mean_pred))
        hypergeom_results = pd.DataFrame(
            {'mean_overlap': mean_overlaps, 'mean_predictions': mean_preds, 'hypergeometric_pvalue': hypergeom_pvalues},
            index=self.nonzero_results.keys()
        )
        return hypergeom_results

    def visualize(self):
        """Generate summary plot of downsampling overlap with true results"""
        default_fig_params = {'figure.figsize': (8, 6)}
        downsample_enrichment_plot(self.hypergeometric_results, fig_params=default_fig_params)\
            .savefig(self.expdir / 'downsampled_hypergeometric-power.pdf')


# util functions
def downsample_gene(X, y, n, n_splits=10):
    return StratifiedShuffleSplit(n_splits=n_splits, train_size=n).split(X, y)
