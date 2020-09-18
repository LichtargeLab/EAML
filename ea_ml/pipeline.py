#!/usr/bin/env python
"""Main script for EA-ML pipeline."""
import datetime
import os
import shutil
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from pkg_resources import resource_filename
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
        targets (Series): Array of target labels for training/prediction
        reference (DataFrame): reference background for genes
        matrix (DesignMatrix): Object for containing feature information for each gene and sample
        clf_info (DataFrame): A DataFrame mapping classifier names to their corresponding Weka object names and
            hyperparameters
        kfolds (int): Number of folds for cross-validation
    """
    feature_names = ('D1', 'D30', 'D70', 'R1', 'R30', 'R70')
    ft_cutoffs = list(product((1, 2), (1, 30, 70)))

    def __init__(self, expdir, data_fn, sample_targets, reference, n_jobs=1, seed=111, kfolds=10):
        self.n_jobs = n_jobs
        self.expdir = expdir
        self.data_fn = data_fn
        self.targets = sample_targets
        self.reference = reference
        self.seed = seed
        self.kfolds = kfolds

        # load classifier information
        self.clf_info = pd.read_csv(resource_filename('ea_ml', 'data/classifiers.csv'),
                                    converters={'options': lambda x: x[1:-1].split(',')})
        # Adaboost doesn't work for Leave-One-Out due to it's implicit sample weighting
        if self.kfolds == -1:
            self.clf_info = self.clf_info[self.clf_info.classifier != 'Adaboost']

    def compute_matrix(self):
        """Computes the full design matrix from an input VCF"""
        self.matrix = 1 - parse_vcf(self.data_fn, self.reference, list(self.targets.index), n_jobs=self.n_jobs)

    def load_matrix(self):
        """Load precomputed matrix with multi-indexed columns"""
        self.matrix = pd.read_csv(self.data_fn, header=[0, 1], index_col=0)

    def run_weka_exp(self):
        """Wraps call to weka_wrapper functions"""
        run_weka(self.expdir, self.matrix, self.targets, self.n_jobs, self.clf_info, seed=self.seed,
                 n_splits=self.kfolds)

    def summarize_experiment(self):
        """Combines results from Weka experiment files"""
        worker_files = (self.expdir / 'tmp').glob('worker-*.results.csv')
        dfs = [pd.read_csv(fn, header=None, index_col=[0, 1]) for fn in worker_files]
        result_df = pd.concat(dfs).sort_index()
        result_df.index.rename(['gene', 'classifier'], inplace=True)
        if self.kfolds == -1:
            result_df.columns = ['TP', 'TN', 'FP', 'FN', 'MCC']
        else:
            result_df.columns = [str(i) for i in range(self.kfolds)]
            result_df['meanMCC'] = result_df.mean(axis=1)
        result_df.sort_values(['gene', 'classifier'], inplace=True)
        result_df.to_csv(self.expdir / 'full-worker-results.csv')
        self.full_result_df = result_df

    def write_results(self):
        cv_df = pd.DataFrame(index=self.reference.index.unique())

        # write summary file for each classifier and aggregate mean MCCs
        for clf in self.clf_info['classifier']:
            clf_df = self.full_result_df.xs(clf, level='classifier')
            clf_df.to_csv(self.expdir / (clf + '-recap.csv'))
            if self.kfolds == -1:
                # if LOO, only a single MCC is present per gene
                cv_df[clf] = clf_df['MCC']
            else:
                cv_df[clf] = clf_df['meanMCC']
        cv_df.to_csv(self.expdir / 'gene-MCC-summary.csv')

        # fetch max and mean MCC for each gene and write final rankings files
        maxMCC_df = cv_df.max(axis=1).sort_values(ascending=False).to_frame(name='maxMCC')
        maxMCC_df.to_csv(self.expdir / 'maxMCC-results.csv')

        meanMCC_df = pd.concat([cv_df.mean(axis=1), cv_df.std(axis=1)], axis=1)
        meanMCC_df.columns = ['meanMCC', 'std']
        meanMCC_df.sort_values('meanMCC', ascending=False, inplace=True)
        meanMCC_df.to_csv(self.expdir / 'meanMCC-results.csv')

        # generate z-score and p-value stats
        max_stats = compute_stats(maxMCC_df, ensemble_type='max')
        max_stats.to_csv(self.expdir / 'maxMCC-results.nonzero-stats.csv')
        mean_stats = compute_stats(meanMCC_df, ensemble_type='mean')
        mean_stats.to_csv(self.expdir / 'meanMCC-results.nonzero-stats.csv')

    def cleanup(self, keep_matrix=False):
        """Deletes intermediate worker files and tmp directory."""
        shutil.rmtree(self.expdir / 'tmp/')
        if keep_matrix:
            self.matrix.to_csv(self.expdir / 'design-matrix.csv.gz')


def compute_stats(results_df, ensemble_type='max'):
    """Generate z-score and p-value statistics for all non-zero MCC scored genes"""
    nonzero = results_df.loc[results_df[f'{ensemble_type}MCC'] != 0].copy()
    nonzero['logMCC'] = np.log(nonzero[f'{ensemble_type}MCC'] + 1 - np.min(nonzero[f'{ensemble_type}MCC']))
    nonzero['zscore'] = (nonzero.logMCC - np.mean(nonzero.logMCC)) / np.std(nonzero.logMCC)
    nonzero['pvalue'] = stats.norm.sf(abs(nonzero.zscore)) * 2
    nonzero['fdr'] = multipletests(nonzero.pvalue, method='fdr_bh')[1]
    return nonzero


def _load_reference(reference, X_chrom=False):
    if reference == 'hg19':
        reference_fn = resource_filename('ea_ml', 'data/hg19-refGene.protein-coding.txt')
    elif reference == 'hg38':
        reference_fn = resource_filename('ea_ml', 'data/hg38-refGene.protein-coding.txt')
    else:
        reference_fn = reference
    reference_df = pd.read_csv(reference_fn, sep='\t', index_col='name2')
    if X_chrom is False:
        reference_df = reference_df[reference_df.chrom != 'chrX']
    return reference_df


def run_ea_ml(exp_dir, data_fn, sample_fn, reference='hg19', n_jobs=1, seed=111, kfolds=10, keep_matrix=False,
              X_chrom=False):
    # check for JAVA_HOME
    assert os.environ['JAVA_HOME'] is not None

    start = time.time()

    # load input data
    exp_dir = exp_dir.expanduser().resolve()
    data_fn = data_fn.expanduser().resolve()
    samples = pd.read_csv(sample_fn, header=None, dtype={0: str, 1: int}, index_col=0, squeeze=True)
    reference_df = _load_reference(reference, X_chrom=X_chrom)

    # initialize pipeline
    pipeline = Pipeline(exp_dir, data_fn, samples, reference_df, n_jobs=n_jobs, seed=seed, kfolds=kfolds)
    # either compute design matrix from VCF or load existing one
    if '.vcf' in str(data_fn):
        pipeline.compute_matrix()
    else:
        pipeline.load_matrix()
    print('Design matrix loaded.')

    print('Running experiment...')
    pipeline.run_weka_exp()
    print('Scoring results...')
    pipeline.summarize_experiment()
    pipeline.write_results()
    print('Gene scoring completed. Analysis summary in experiment directory.')

    pipeline.cleanup(keep_matrix=keep_matrix)
    end = time.time()
    elapsed = str(datetime.timedelta(seconds=end - start))
    print(f'Time elapsed: {elapsed}')
