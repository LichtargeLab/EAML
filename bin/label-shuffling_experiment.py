#!/usr/bin/env python3
"""
Created on 2019-04-22

@author: dillonshapiro

Wrapper script that runs a label shuffling experiment to assign significance
z-score/p-value to top predictions from EA-ML analysis.
"""
import pandas as pd
import numpy as np
import os
import shutil
from sys import argv
import subprocess
import argparse
from pathlib import Path
from scipy import stats


def shuffle_labels(df_path, run_dir):
    """
    Shuffles the case/control labels and writes out the new sample list.

    Args:
        df_path (str): Path to the original sample,label list
        run_dir (str): Path to the specific randomization run directory

    Returns:
        shuff_labels_path (str): Path to the outputed shuffled sample list
    """
    df = pd.read_csv(df_path, header=None)
    df[1] = np.random.permutation(df[1])
    shuff_labels_path = f'{run_dir}/shuffled_samples.csv'
    df.to_csv(f'{run_dir}/shuffled_samples.csv', header=False, index=False)
    return shuff_labels_path


def merge_runs(exp_dir, n_runs=100):
    """
    Merges all of the randomized runs into a single DataFrame.

    Args:
        exp_dir (str): Path to the main experiment folder.
        n_runs (int): Number of shufflings to incorporate into the random
            distributions.

    Returns:
        merged_df (pd.DataFrame): A DataFrame of the random MCC distribution
            for each tested gene.
    """
    merged_df = pd.read_csv(f'{exp_dir}/run1/maxMCC_summary.csv')
    merged_df.columns = ['gene', 'run1']
    for i in range(2, n_runs + 1):
        df = pd.read_csv(f'{exp_dir}/run{i}/maxMCC_summary.csv')
        df.columns = ['gene', f'run{i}']
        merged_df = merged_df.merge(df, on='gene')
    merged_df.sort_values(by='gene').reset_index(drop=True, inplace=True)
    merged_df.set_index('gene', inplace=True)
    return merged_df


def compute_zscores(preds_path, shuffle_results):
    preds_df = pd.read_csv(preds_path, index_col='gene').sort_index()
    rand_means = shuffle_results.mean(axis=1)
    rand_stds = shuffle_results.std(axis=1)
    norm_test = shuffle_results.apply(stats.shapiro, axis=1)
    norm_test = [test[1] for test in norm_test]
    rand_results = pd.DataFrame({
        'maxMCC': preds_df['maxMCC'],
        'rand_mean': rand_means,
        'rand_std': rand_stds,
        'zscore': (preds_df['maxMCC'] - rand_means) / rand_stds,
        'shapiro_normality': norm_test
    }, index=preds_df.index)
    return rand_results


def main(exp_dir, labels_path, pipe_dir, vcf_path, gene_list, preds_path,
         n_workers=4, n_runs=100):
    for i in range(1, n_runs + 1):
        run_dir = f'{exp_dir}/run{i}'
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        new_labels = shuffle_labels(labels_path, run_dir)
        os.chdir(run_dir)
        subprocess.call([f'{pipe_dir}/run.sh', '-e', run_dir, '-d', vcf_path,
                         '-s', new_labels, '-g', gene_list, '-n', str(n_workers)])

    shuffle_results = merge_runs(exp_dir, n_runs)
    shuffle_results.to_csv(f'{exp_dir}/random_distributions.csv')
    rand_results = compute_zscores(preds_path, shuffle_results)
    rand_results.to_csv(f'{exp_dir}/randomization_results.csv')
    os.mkdir('random_exp')
    for i in range(1, n_runs + 1):
        shutil.move(f'{exp_dir}/run{i}', f'{exp_dir}/random_exp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Wrapper script that runs a label shuffling experiment to '
                    'assign signficance z-score/p-value to top predictions '
                    'from EA-ML analysis.'
    )
    parser.add_argument('exp_dir', type=Path,
                        help='Path to the overarching experiment directory')
    parser.add_argument('labels_path', type=Path,
                        help='Path to samples list containing original '
                             'corresponding labels')
    parser.add_argument('data', type=Path, help='Path to VCF file')
    parser.add_argument('gene_list', type=Path,
                        help='Path to single-column list of test genes')
    parser.add_argument('predictions', type=Path,
                        help='Path to real experiment results')
    parser.add_argument('--n_jobs', '-n', type=int, default=4,
                        help='Number of worker processes to use for Weka')
    parser.add_argument('--n_runs', '-r', type=int, default=100,
                        help='Number of shuffling runs to include in '
                             'distribution')
    args = parser.parse_args()

    pipe_dir = Path(argv[0]).parent.parent.expanduser().resolve()
    exp_dir = args.exp_dir.expanduser().resolve()
    labels_path = args.labels_path.expanduser().resolve()
    vcf = args.data.expanduser().resolve()
    gene_list = args.gene_list.expanduser().resolve()
    predictions = args.predictions.expanduser().resolve()

    main(exp_dir, labels_path, pipe_dir, vcf, gene_list, predictions,
         n_workers=args.n_jobs, n_runs=args.n_runs)
