#!/usr/bin/env python
"""
Functions that run a permutation experiment to assign significance z-score/p-value to top predictions from
EA-ML analysis.
"""
import shutil

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .pipeline import Pipeline


def permute_targets(targets_fn, run_dir):
    """
    Permute the case/control targets and write out new sample targets

    Args:
        targets_fn (Path-like): Filepath to original sample class targets
        run_dir (Path-like): Filepath to the specific permutation directory

    Returns:
        perm_labels_path (Path-like): Path to the output permuted sample list
    """
    targets = pd.read_csv(targets_fn, header=None, dtypes={0: str, 1: int}, index_col=0, squeeze=True)
    permuted_targets = pd.Series(np.random.permutation(targets), index=targets.index)
    perm_targets_path = run_dir / 'label_permutation.csv'
    permuted_targets.to_csv(perm_targets_path, header=False, index=False)
    return perm_targets_path


def merge_runs(exp_dir, n_runs=100):
    """
    Merge all of the permutation runs into a single DataFrame

    Args:
        exp_dir (Path-like): Filepath to the main experiment folder
        n_runs (int): Number of permutations

    Returns:
        DataFrame: A DataFrame of the random MCC distribution for each tested gene
    """
    merged_df = pd.read_csv(exp_dir / 'run1/classifier-MCC-summary.csv', index_col='gene', usecols=['gene', 'mean'],
                            names=['gene', 'run1'])
    for i in range(2, n_runs + 1):
        df = pd.read_csv(exp_dir / f'run{i}/classifier-MCC-summary.csv', index_col='gene', usecols=['gene', 'mean'],
                         names=['gene', f'run{i}'])
        merged_df = merged_df.join(df)
    return merged_df


def compute_stats(preds_fn, perm_results):
    """
    Calculate z-scores, p-values, and adjusted p-values for permutations

    Args:
        preds_fn (Path-like): Filepath to true EA-ML results
        perm_results (Path-like): Filepath to permuted MCC distributions

    Returns:
        DataFrame: True EA-ML results with permuted significance statistics
    """
    preds_df = pd.read_csv(preds_fn, index_col='gene')
    rand_means = perm_results.mean(axis=1)
    rand_stds = perm_results.std(axis=1)
    pvals = perm_results.apply(
        lambda row: np.sum(row > preds_df.loc[row.name, 'mean']) / perm_results.shape[1],
        axis=1
    )  # one-tailed test (effectively ignoring negative MCCs)
    rand_results = pd.DataFrame({
        'MCC': preds_df['mean'],
        'rand_mean': rand_means,
        'rand_std': rand_stds,
        'zscore': (preds_df['meanMCC'] - rand_means) / rand_stds,
        'pvalue': pvals,
        'qvalue': multipletests(pvals, method='fdr_bh')[1]
    }, index=preds_df.index)
    return rand_results.sort_values(['pvalue', 'MCC'], ascending=[True, False])


def run_permutations(exp_dir, data_fn, targets_fn, preds_fn, reference='hg19', cpus=1, seed=111, kfolds=10, n_runs=100,
                     restart=0, clean=False, include_X=False, min_af=None, max_af=None, af_field='AF',
                     weka_path='~/weka', memory='Xmx2g'):
    """
    Run permutation experiment

    Args:
        exp_dir (Path-like): Filepath to main experiment folder
        data_fn (Path-like): Filepath to VCF
        targets_fn (Path-like): Filepath to original sample class targets
        preds_fn (Path-like): Filepath to true EA-ML results
        reference (str): Genome reference name or filepath
        cpus (int): Number of CPUs for multiprocessing
        seed (int): Random seed
        kfolds (int): Number of cross-validation folds
        n_runs (int):  Number of permutations to run
        restart (int): If non-zero, restarts permutation count from here
        clean (bool): Cleans each permutation's results
        include_X (bool): Includes X chromosome genes in experiment
        min_af (float): Minimum allele frequency for variants
        max_af (float): Maximum allele frequency for variants
        af_field (str): Name of INFO field containing allele frequency information
        weka_path (Path-like): Filepath to Weka directory
        memory (str): Memory argument for each Weka JVM
    """
    if restart > 0:  # restart permutation count from here
        start = restart
    else:  # the default, no restart
        start = 0
    for i in range(start, n_runs + 1):
        run_dir = exp_dir / f'run{i}'
        run_dir.mkdir()
        new_targets_fn = permute_targets(targets_fn, run_dir)
        if '.vcf' in data_fn.suffixes:  # if starting with raw input data, save dmatrices for later permutations
            write_data = True
        else:
            write_data = False
        pipeline = Pipeline(run_dir, data_fn, new_targets_fn, reference=reference, cpus=cpus, kfolds=kfolds, seed=seed,
                            weka_path=weka_path, min_af=min_af, max_af=max_af, af_field=af_field, include_X=include_X,
                            write_data=write_data, memory=memory)
        pipeline.run()
        if '.vcf' in data_fn.suffixes:
            data_fn = (run_dir / 'dmatrices.h5').rename(exp_dir / 'dmatrices.h5')

    # aggregate background distributions
    perm_dist = merge_runs(exp_dir, n_runs)
    perm_dist.to_csv(exp_dir / 'random_distributions.csv')
    # compute stats on observed score vs. background
    perm_stats = compute_stats(preds_fn, perm_dist)
    perm_stats.to_csv(exp_dir / 'permutation_results.csv')
    if clean:
        data_fn.unlink()
        for i in range(1, n_runs + 1):
            shutil.rmtree(str(exp_dir / f'run{i}'), ignore_errors=True)
    else:
        (exp_dir / 'permute_exp').mkdir()
        for i in range(1, n_runs + 1):
            shutil.move(str(exp_dir / f'run{i}'), str(exp_dir / 'permute_exp'))
