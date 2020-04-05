#!/usr/bin/env python3
"""
Created on 9/26/19

@author: dillonshapiro

Functions that run a permutation experiment to assign significance z-score/p-value to top predictions from
EA-ML analysis.
"""
import shutil

import numpy as np
import pandas as pd
from scipy import stats

from .pipeline import run_ea_ml


def permute_labels(df_path, run_dir):
    """
    Permutes the case/control labels and writes out the new sample list.

    Args:
        df_path (str): Path to the original sample,label list
        run_dir (Path): Path to the specific permutation directory

    Returns:
        perm_labels_path (str): Path to the output permuted sample list
    """
    df = pd.read_csv(df_path, header=None)
    df[1] = np.random.permutation(df[1])
    perm_labels_path = run_dir / 'label_permutation.csv'
    df.to_csv(perm_labels_path, header=False, index=False)
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
    merged_df = pd.read_csv(exp_dir / 'run1/maxMCC_summary.csv')
    merged_df.columns = ['gene', 'run1']
    for i in range(2, n_runs + 1):
        df = pd.read_csv(exp_dir / f'run{i}/maxMCC_summary.csv')
        df.columns = ['gene', f'run{i}']
        merged_df = merged_df.merge(df, on='gene')
    merged_df.sort_values(by='gene').reset_index(drop=True, inplace=True)
    merged_df.set_index('gene', inplace=True)
    return merged_df


def _nonzero_shapiro(arr):
    """
    Removes MCCs that equal 0, as these skew the distribution and they are already removed from our
    whole-genome comparison
    """
    filt_arr = np.array([x for x in arr if x != 0])
    return stats.shapiro(filt_arr)[1]


def compute_zscores(preds_path, perm_results):
    preds_df = pd.read_csv(preds_path, index_col='gene').sort_index()
    rand_means = perm_results.mean(axis=1)
    rand_stds = perm_results.std(axis=1)
    norm_test = perm_results.apply(_nonzero_shapiro, axis=1)
    rand_results = pd.DataFrame({
        'maxMCC': preds_df['maxMCC'],
        'rand_mean': rand_means,
        'rand_std': rand_stds,
        'zscore': (preds_df['maxMCC'] - rand_means) / rand_stds,
        'shapiro_normality': norm_test
    }, index=preds_df.index)
    return rand_results


def run_permutations(exp_dir, data, samples, gene_list, preds_path, threads=1, seed=111, kfolds=10, n_runs=100):
    for i in range(1, n_runs + 1):
        run_dir = exp_dir / f'run{i}'
        run_dir.mkdir()
        new_labels = permute_labels(samples, run_dir)
        run_ea_ml(run_dir, data, new_labels, gene_list, threads=threads, seed=seed, kfolds=kfolds)
        if i == 1:
            shutil.move(run_dir / 'design_matrix.npz', exp_dir)
        else:
            data = exp_dir / 'design_matrix.npz'

    perm_results = merge_runs(exp_dir, n_runs)
    perm_results.to_csv(exp_dir / 'random_distributions.csv')
    rand_results = compute_zscores(preds_path, perm_results)
    rand_results.to_csv(exp_dir / 'randomization_results.csv')
    (exp_dir / 'random_exp').mkdir()
    for i in range(1, n_runs + 1):
        shutil.move(exp_dir / f'run{i}', exp_dir / 'random_exp')
