#!/usr/bin/env python
"""Main script for EA-ML pipeline."""
import datetime
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from pysam import VariantFile
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .utils import fetch_variants, load_matrix, convert_zygo
from .design_matrix import DesignMatrix
from .weka_wrapper import run_weka


class Pipeline(object):
    """
    Attributes:
        threads (int): Number of cores to be used by Weka
        expdir (Path): filepath to experiment folder
        data (Path): filepath to VCF file
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

    def __init__(self, expdir, data, sample_f, gene_list, threads=1, seed=111, kfolds=10):
        self.threads = threads
        self.expdir = expdir
        self.data = data
        self.seed = seed
        self.kfolds = kfolds

        # load feature and sample info
        sample_df = pd.read_csv(sample_f, header=None, dtype={0: str, 1: int}).sort_values(0)
        self.targets = np.array(sample_df[1])
        self.samples = list(sample_df[0])
        self.test_genes = list(pd.read_csv(gene_list, header=None, squeeze=True).sort_values())
        self._gene_features = [f'{gene}_{feature}' for gene in self.test_genes for feature in self.feature_names]

        # initialize design matrix
        if self.data.suffix == '.npz':
            arr = load_matrix(self.data)
        else:
            arr = np.ones((len(self.samples), len(self._gene_features)))
        self.matrix = DesignMatrix(arr, self.targets, self._gene_features, self.samples,
                                   feature_names=self.feature_names)

        # load classifier information
        self.clf_info = pd.read_csv(Path(__file__).parent / 'classifiers.csv',
                                    converters={'options': lambda x: x[1:-1].split(',')})
        # Adaboost doesn't work for Leave-One-Out due to it's implicit sample weighting
        if self.kfolds == -1:
            self.clf_info = self.clf_info[self.clf_info.classifier != 'Adaboost']

    def process_contig(self, vcf, contig=None):
        """
        Reads and updates gene features in DesignMatrix based on a single contig.

        Args:
            vcf (VariantFile): A VariantFile object
            contig (str): A specific contig/chromosome to fetch from the VCF. If None, will iterate through the entire
                VCF file
        """
        for gene, ea, sample_info in fetch_variants(vcf, contig=contig):
            if gene not in self.test_genes or np.isnan(ea).all():
                continue
            # check genotypes and update matrix
            gts = np.array([convert_zygo(sample_info[sample]['GT']) for sample in self.samples])
            # duplicate EA scores to do numpy operations across all samples
            ea_arr = np.repeat(ea[:, np.newaxis], len(gts), axis=1)
            feature_arrs = []
            for cutoff in self.ft_cutoffs:
                mask = (ea_arr >= cutoff[1]) & (gts >= cutoff[0])  # EA and zygosity mask for each feature
                prod_arr = np.nanprod((1 - (ea_arr * mask) / 100)**gts, axis=0)  # pEA calculation
                feature_arrs.append(prod_arr)
            feature_arrs = np.vstack(feature_arrs).T
            self.matrix.update(feature_arrs, gene)

    def process_vcf(self):
        """The overall method for processing the entire VCF file."""
        vcf = VariantFile(self.data)
        vcf.subset_samples(self.samples)
        for contig in list(range(1, 23)) + ['X', 'Y']:
            try:
                print('Processing chromosome {}...'.format(contig))
                self.process_contig(vcf, contig=str(contig))
            except ValueError as e:
                if 'invalid contig' in str(e) and contig in ['X', 'Y']:
                    print(f'No {contig} chromosome data.')
                    continue
        self.matrix.X = 1 - self.matrix.X  # final pEA computation
        self.matrix.write_matrix(self.expdir / 'design_matrix.npz')

    def run_weka_exp(self):
        """Wraps call to weka_wrapper functions"""
        run_weka(self.expdir, self.matrix, self.test_genes, self.threads, self.clf_info, seed=self.seed,
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
    nonzero['logMCC'] = np.log(nonzero[f'{ensemble_type}MCC'] + 1 - np.min(nonzero.maxMCC))
    nonzero['zscore'] = (nonzero.logMCC - np.mean(nonzero.logMCC)) / np.std(nonzero.logMCC)
    nonzero['pvalue'] = stats.norm.sf(abs(nonzero.zscore)) * 2
    nonzero['fdr'] = multipletests(nonzero.pvalue, method='fdr_bh')[1]
    return nonzero


def cleanup(expdir, keep_matrix=False):
    """Deletes intermediate worker files and temp directory."""
    shutil.rmtree(expdir / 'temp/')
    if not keep_matrix:
        (expdir / 'design_matrix.npz').unlink()


def run_ea_ml(exp_dir, data, sample_f, gene_list, threads=1, seed=111, kfolds=10, keep_matrix=False):
    # check for JAVA_HOME
    assert os.environ['JAVA_HOME'] is not None

    start = time.time()
    # either load existing design matrix or compute new one from VCF
    pipeline = Pipeline(exp_dir, data, sample_f, gene_list, threads=threads, seed=seed, kfolds=kfolds)
    if not data.suffix == '.npz':
        pipeline.process_vcf()
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
