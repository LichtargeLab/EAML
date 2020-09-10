#!/usr/bin/env python
"""Main script for EA-ML pipeline."""
import datetime
import os
import shutil
import time
from collections import OrderedDict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .vcf import parse_vcf
from .weka_wrapper import run_weka


class Pipeline(object):
    """
    Attributes:
        n_jobs (int): number of parallel jobs
        expdir (Path): filepath to experiment folder
        data_fn (Path): filepath to data input file (either a VCF or multi-indexed DataFrame)
        seed (int): Random seed for KFold sampling
        targets (np.ndarray): Array of target labels for training/prediction
        samples (list): List of samples to test
        test_genes (list): list of genes to test
        matrix (DesignMatrix): Object for containing feature information for each gene and sample
        clf_info (DataFrame): A DataFrame mapping classifier names to their corresponding Weka object names and
            hyperparameters
        kfolds (int): Number of folds for cross-validation
    """
    feature_names = ('D1', 'D30', 'D70', 'R1', 'R30', 'R70')
    ft_cutoffs = list(product((1, 2), (1, 30, 70)))

    def __init__(self, expdir, data_fn, sample_fn, gene_list, n_jobs=1, seed=111, kfolds=10):
        self.n_jobs = n_jobs
        self.expdir = expdir
        self.data_fn = data_fn
        self.seed = seed
        self.kfolds = kfolds

        # load feature and sample info
        sample_df = pd.read_csv(sample_fn, header=None, dtype={0: str, 1: int}).sort_values(0)
        self.targets = sample_df[1]
        self.samples = list(sample_df[0])
        self.test_genes = list(pd.read_csv(gene_list, header=None, squeeze=True).sort_values())

        # load classifier information
        self.clf_info = pd.read_csv(Path(__file__).parent / 'classifiers.csv',
                                    converters={'options': lambda x: x[1:-1].split(',')})
        # Adaboost doesn't work for Leave-One-Out due to it's implicit sample weighting
        if self.kfolds == -1:
            self.clf_info = self.clf_info[self.clf_info.classifier != 'Adaboost']

    def compute_matrix(self):
        """Computes the full design matrix from an input VCF"""
        self.matrix = 1 - parse_vcf(self.data_fn, self.reference, self.samples, n_jobs=self.n_jobs)
        self.matrix.to_csv('design_matrix.csv.bz2')

    def load_matrix(self):
        """Load precomputed matrix with multi-indexed columns"""
        self.matrix = pd.read_csv(self.data_fn, header=[0, 1], index_col=0)

    def run_weka_exp(self):
        """Wraps call to weka_wrapper functions"""
        run_weka(self.expdir, self.matrix, self.targets, self.test_genes, self.n_jobs, self.clf_info, seed=self.seed,
                 n_splits=self.kfolds)

    def summarize_experiment(self):
        """Combines results from Weka experiment files"""
        worker_files = (self.expdir / 'temp').glob('worker-*.results.csv')
        dfs = [pd.read_csv(fn, header=None) for fn in worker_files]
        result_df = pd.concat(dfs, ignore_index=True)
        if self.kfolds == -1:
            result_df.columns = ['gene', 'classifier', 'TP', 'TN', 'FP', 'FN', 'MCC']
        else:
            result_df.columns = ['gene', 'classifier'] + [str(i) for i in range(self.kfolds)]
            result_df['meanMCC'] = result_df.mean(axis=1)
        result_df.sort_values(['gene', 'classifier'], inplace=True)
        result_df.set_index(['gene', 'classifier'], inplace=True)
        result_df.to_csv(self.expdir / 'gene_MCC_summary.csv')
        self.full_result_df = result_df

    def write_results(self):
        clf_d = OrderedDict([('gene', self.test_genes)])

        # write summary file for each classifier
        for clf in self.clf_info['classifier']:
            clf_df = self.full_result_df.xs(clf, level='classifier')
            clf_df.to_csv(self.expdir / (clf + '-recap.csv'))
            if self.kfolds == -1:
                # if LOO, only a single MCC is present per gene
                clf_d[clf] = list(clf_df['MCC'])
            else:
                clf_d[clf] = list(clf_df['meanMCC'])

        # aggregate meanMCCs from each classifier
        cv_df = pd.DataFrame.from_dict(clf_d)
        cv_df.to_csv(self.expdir / 'all-classifier_means.csv', index=False)

        # fetch max and mean MCC for each gene and write final rankings files
        maxMCC_df = pd.DataFrame({'gene': self.test_genes, 'maxMCC': cv_df.max(axis=1)})
        maxMCC_df.sort_values('maxMCC', ascending=False, inplace=True)
        maxMCC_df.to_csv(self.expdir / 'maxMCC_results.csv', index=False)

        meanMCC_df = pd.DataFrame({'gene': self.test_genes, 'meanMCC': cv_df.mean(axis=1), 'std': cv_df.std(axis=1)})
        meanMCC_df.sort_values('meanMCC', ascending=False, inplace=True)
        meanMCC_df.to_csv(self.expdir / 'meanMCC_results.csv', index=False)

        # generate z-score and p-value stats
        max_stats = compute_stats(maxMCC_df, ensemble_type='max')
        max_stats.to_csv(self.expdir / 'maxMCC_results.nonzero-stats.csv', index=False)
        mean_stats = compute_stats(meanMCC_df, ensemble_type='mean')
        mean_stats.to_csv(self.expdir / 'meanMCC_results.nonzero-stats.csv', index=False)


def compute_stats(results_df, ensemble_type='max'):
    """Generate z-score and p-value statistics for all non-zero MCC scored genes"""
    nonzero = results_df.loc[results_df[f'{ensemble_type}MCC'] != 0].copy()
    nonzero['logMCC'] = np.log(nonzero[f'{ensemble_type}MCC'] + 1 - np.min(nonzero[f'{ensemble_type}MCC']))
    nonzero['zscore'] = (nonzero.logMCC - np.mean(nonzero.logMCC)) / np.std(nonzero.logMCC)
    nonzero['pvalue'] = stats.norm.sf(abs(nonzero.zscore)) * 2
    nonzero['fdr'] = multipletests(nonzero.pvalue, method='fdr_bh')[1]
    return nonzero


def cleanup(expdir, keep_matrix=False):
    """Deletes intermediate worker files and temp directory."""
    shutil.rmtree(expdir / 'tmp/')
    if not keep_matrix:
        (expdir / 'design_matrix.csv.bz2').unlink()


def run_ea_ml(exp_dir, data_fn, sample_fn, gene_list, n_jobs=1, seed=111, kfolds=10, keep_matrix=False):
    # check for JAVA_HOME
    assert os.environ['JAVA_HOME'] is not None

    start = time.time()
    # either load existing design matrix or compute new one from VCF
    exp_dir = exp_dir.expanduser().resolve()
    data_fn = data_fn.expanduser().resolve()
    pipeline = Pipeline(exp_dir, data_fn, sample_fn, gene_list, n_jobs=n_jobs, seed=seed, kfolds=kfolds)
    if '.vcf' in data_fn:
        pipeline.compute_matrix()
    else:
        pipeline.load_matrix()
    print('Design matrix loaded.')

    print('Running experiment...')
    pipeline.run_weka_exp()
    pipeline.summarize_experiment()
    pipeline.write_results()
    print('Gene scoring completed. Analysis summary in experiment directory.')

    cleanup(exp_dir, keep_matrix=keep_matrix)
    end = time.time()
    elapsed = str(datetime.timedelta(seconds=end - start))
    print(f'Time elapsed: {elapsed}')
