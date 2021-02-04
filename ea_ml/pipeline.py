#!/usr/bin/env python
"""Main script for EA-ML pipeline."""
import datetime
import os
import shutil
import time
from pathlib import Path
from joblib import delayed, Parallel
from tqdm import tqdm

import numpy as np
import pandas as pd
from pkg_resources import resource_filename
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .vcf import parse_gene
from .weka_wrapper import eval_gene
# TODO: finish and update docstrings


class Pipeline(object):
    """
    Attributes:
        n_jobs (int): number of parallel jobs
        expdir (Path): filepath to experiment folder
        data_fn (Path): filepath to data input file (either a VCF or multi-indexed DataFrame)
        targets (Series): Array of target labels for training/prediction
        reference (DataFrame): reference background for genes
        kfolds (int): Number of folds for cross-validation
    """
    class_params = {
        'PART': '-M 5 -C 0.25 -Q 1',
        'JRip': '-F 3 -N 2.0 -O 2 -S 1 -P',
        'RandomForest': '-I 10 -K 0 -S 1',
        'J48': '-C 0.25 -M 5',
        'NaiveBayes': '',
        'Logistic': '-R 1.0E-8 -M -1',
        'IBk': '-K 3 -W 0 -A \".LinearNNSearch -A \\\".EuclideanDistance -R first-last\\\"\"',
        'AdaBoostM1': '-P 100 -S 1 -I 10 -W .DecisionStump',
        'MultilayerPerceptron': '-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a'
    }

    def __init__(self, expdir, data_fn, targets_fn, reference='hg19', n_jobs=1, kfolds=10, seed=111,
                 weka_path='/opt/weka', min_af=None, max_af=None, af_field='AF', include_X=False, write_data=False):
        # data arguments
        self.expdir = expdir.expanduser().resolve()
        self.data_fn = data_fn.expanduser().resolve()
        self.targets = pd.read_csv(targets_fn, header=None, dtype={0: str, 1: int}).set_index(0).squeeze().sort_index()
        self.reference = load_reference(reference, include_X=include_X)

        # config arguments
        self.kfolds = kfolds
        self.seed = seed
        self.weka_path = weka_path
        self.min_af = min_af
        self.max_af = max_af
        self.af_field = af_field
        self.n_jobs = n_jobs
        self.write_data = write_data

    def run(self):
        gene_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.eval_gene)(gene) for gene in tqdm(self.reference.index.unique())
        )
        self.report_results(gene_results)

    def compute_gene_dmatrix(self, gene):
        """Computes the full design matrix from an input VCF"""
        gene_reference = self.reference.loc[gene]
        dmatrix = parse_gene(self.data_fn, gene, gene_reference, self.targets.index, min_af=self.min_af,
                             max_af=self.max_af, af_field=self.af_field)
        if self.write_data:
            dmatrix.to_hdf(self.expdir / 'dmatrices.h5', key=gene, complevel=5, complib='zlib', format='fixed')
        return dmatrix

    def eval_gene(self, gene):
        """
        Parses input data for a given gene and evaluates it using Weka

        Args:
            gene (str): HGSC gene symbol

        Returns:
            dict(float): Mapping of classifier to MCC from cross validation
        """
        if self.expdir.suffix == '.h5':
            hdf_fn = self.expdir / 'dmatrices.h5'
            gene_dmatrix = pd.read_hdf(hdf_fn, key=gene)
        else:
            gene_dmatrix = self.compute_gene_dmatrix(gene)
        mcc_results = eval_gene(gene, gene_dmatrix, self.targets, self.class_params, seed=self.seed, cv=self.seed,
                                expdir=self.expdir, weka_path=self.weka_path)
        return gene, mcc_results

    def report_results(self, gene_results):
        """
        Summarize and rank gene scores

        Args:
            gene_results (list): Results of Weka fitting, in the format of (gene, score dict)
        """
        # TODO: finish reporting refactor
        pass

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
        cv_df.index.rename('gene', inplace=True)

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

        # fetch mean MCC for each gene and write final rankings files
        meanMCC_df = pd.concat([cv_df.mean(axis=1), cv_df.std(axis=1)], axis=1)
        meanMCC_df.columns = ['meanMCC', 'std']
        meanMCC_df.sort_values('meanMCC', ascending=False, inplace=True)
        meanMCC_df.to_csv(self.expdir / 'meanMCC-results.csv')

        # generate z-score and p-value stats
        mean_stats = compute_stats(meanMCC_df, ensemble_type='mean')
        mean_stats.to_csv(self.expdir / 'meanMCC-results.nonzero-stats.csv')

    def cleanup(self):
        """Cleans tmp directory"""
        shutil.rmtree(self.expdir / 'tmp/')


def compute_stats(results_df, ensemble_type='mean'):
    """Generate z-score and p-value statistics for all non-zero MCC scored genes"""
    nonzero = results_df.loc[results_df[f'{ensemble_type}MCC'] != 0].copy()
    nonzero['logMCC'] = np.log(nonzero[f'{ensemble_type}MCC'] + 1 - np.min(nonzero[f'{ensemble_type}MCC']))
    nonzero['zscore'] = (nonzero.logMCC - np.mean(nonzero.logMCC)) / np.std(nonzero.logMCC)
    nonzero['pvalue'] = stats.norm.sf(abs(nonzero.zscore)) * 2
    nonzero['fdr'] = multipletests(nonzero.pvalue, method='fdr_bh')[1]
    return nonzero


def load_reference(reference, include_X=False):
    if reference == 'hg19':
        reference_fn = resource_filename('ea_ml', 'data/hg19-refGene.protein-coding.txt')
    elif reference == 'hg38':
        reference_fn = resource_filename('ea_ml', 'data/hg38-refGene.protein-coding.txt')
    else:
        reference_fn = reference
    reference_df = pd.read_csv(reference_fn, sep='\t', index_col='name2')
    if include_X is False:
        reference_df = reference_df[reference_df.chrom != 'chrX']
    return reference_df


def run_ea_ml(exp_dir, data_fn, targets_fn, reference='hg19', cpus=1, seed=111, kfolds=10, write_data=False,
              include_X=False, min_af=None, max_af=None, af_field='AF'):
    start = time.time()

    # initialize pipeline
    pipeline = Pipeline(exp_dir, data_fn, targets_fn, reference=reference, n_jobs=cpus, kfolds=kfolds, seed=seed,
                        min_af=min_af, max_af=max_af, af_field=af_field, include_X=include_X, write_data=write_data)
    pipeline.run()
    print('Gene scoring completed. Analysis summary in experiment directory.')

    pipeline.cleanup()
    end = time.time()
    elapsed = str(datetime.timedelta(seconds=end - start))
    print(f'Time elapsed: {elapsed}')
