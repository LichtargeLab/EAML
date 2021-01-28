#!/usr/bin/env python
"""
Functions that run a permutation experiment to assign significance z-score/p-value to top predictions from
EA-ML analysis.
"""
import shutil

import numpy as np
import pandas as pd

from .pipeline import run_ea_ml


def permute_labels(samples_fn, run_dir):
    """
    Permutes the case/control labels and writes out the new sample list.

    Args:
        samples_fn (str): Path to the original sample,target list
        run_dir (Path): Path to the specific permutation directory

    Returns:
        perm_labels_path (str): Path to the output permuted sample list
    """
    targets = pd.read_csv(samples_fn, header=None, dtypes={0: str, 1: int}, index_col=0, squeeze=True)
    permuted_targets = pd.Series(np.random.permutation(targets), index=targets.index)
    perm_labels_path = run_dir / 'label_permutation.csv'
    permuted_targets.to_csv(perm_labels_path, header=False, index=False)
    return perm_labels_path


def merge_runs(exp_dir, n_runs=100):
    """
    Merges all of the randomized runs into a single DataFrame.

    Args:
        exp_dir (Path): Path to the main experiment folder.
        n_runs (int): Number of permutations to incorporate into the random distributions.

    Returns:
        merged_df (pd.DataFrame): A DataFrame of the random MCC distribution for each tested gene.
    """
    merged_df = pd.read_csv(exp_dir / 'run1/meanMCC_results.csv', index_col='gene', names=['run1'])
    for i in range(2, n_runs + 1):
        df = pd.read_csv(exp_dir / 'run{i}/meanMCC_results.csv', index_col='gene', names=[f'run{i}'])
        merged_df = merged_df.join(df)
    return merged_df


def compute_zscores(preds_fn, perm_results):
    preds_df = pd.read_csv(preds_fn, index_col='gene')
    rand_means = perm_results.mean(axis=1)
    rand_stds = perm_results.std(axis=1)
    pvals = perm_results.apply(
        lambda row: np.sum(row > preds_df.loc[row.name, 'meanMCC']) / perm_results.shape[1],
        axis=1
    )  # one-tailed test (effectively ignoring negative MCCs)
    rand_results = pd.DataFrame({
        'meanMCC': preds_df['meanMCC'],
        'rand_mean': rand_means,
        'rand_std': rand_stds,
        'zscore': (preds_df['meanMCC'] - rand_means) / rand_stds,
        'pvalue': pvals
    }, index=preds_df.index)
    return rand_results


def run_permutations(exp_dir, data_fn, samples_fn, preds_fn, reference='hg19', cpus=1, seed=111, kfolds=10, n_runs=100,
                     restart=0, clean=False, include_X=False, min_af=None, max_af=None, af_field='AF'):
    if restart > 0:  # restart permutation count from here
        start = restart
    else:  # the default, no restart
        start = 0
    for i in range(start, n_runs + 1):
        run_dir = exp_dir / f'run{i}'
        run_dir.mkdir()
        new_labels = permute_labels(samples_fn, run_dir)
        run_ea_ml(run_dir, data_fn, new_labels, reference=reference, cpus=cpus, seed=seed, kfolds=kfolds,
                  keep_matrix=True, include_X=include_X, min_af=min_af, max_af=max_af, af_field=af_field)
        if '.vcf' in str(data_fn):
            data_fn = exp_dir / 'design_matrix.csv.gz'
            shutil.move(str(data_fn), str(exp_dir))
    # aggregate background distributions

    perm_dist = merge_runs(exp_dir, n_runs)
    perm_dist.to_csv(exp_dir / 'random_distributions.csv')
    # compute stats on observed score vs. background
    perm_stats = compute_zscores(preds_fn, perm_dist)
    perm_stats.to_csv(exp_dir / 'permutation_results.csv')
    if clean:
        data_fn.unlink()
        for i in range(1, n_runs + 1):
            shutil.rmtree(str(exp_dir / f'run{i}'), ignore_errors=True)
    else:
        (exp_dir / 'permute_exp').mkdir()
        for i in range(1, n_runs + 1):
            shutil.move(str(exp_dir / f'run{i}'), str(exp_dir / 'permute_exp'))
