#!/usr/bin/env python
"""Main script for EA-ML pipeline."""
import datetime
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from pkg_resources import resource_filename
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .vcf import parse_gene
from .weka_wrapper import eval_gene
from .visualize import mcc_hist, mcc_scatter, manhattan_plot


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

    def __init__(self, expdir, data_fn, targets_fn, reference='hg19', n_jobs=1, kfolds=10, seed=111, dpi=150,
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
        self.dpi = dpi
        self.write_data = write_data

    def run(self):
        start = time.time()
        gene_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.eval_gene)(gene) for gene in tqdm(self.reference.index.unique())
        )
        self.raw_results = gene_results
        self.report_results()
        print('Gene scoring completed. Analysis summary in experiment directory.')
        self.visualize()
        self.cleanup()
        end = time.time()
        elapsed = str(datetime.timedelta(seconds=end - start))
        print(f'Time elapsed: {elapsed}')

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

    def compute_stats(self):
        """Generate z-score and p-value statistics for all non-zero MCC scored genes"""
        mcc_df = self.full_results[['mean', 'std']]
        nonzero = mcc_df.loc[mcc_df[f'mean'] != 0].copy()
        nonzero.rename(columns={'mean': 'meanMCC'}, inplace=True)
        nonzero['logMCC'] = np.log(nonzero.meanMCC + 1 - np.min(nonzero.meanMCC))
        nonzero['zscore'] = (nonzero.logMCC - np.mean(nonzero.logMCC)) / np.std(nonzero.logMCC)
        nonzero['pvalue'] = stats.norm.sf(abs(nonzero.zscore)) * 2
        nonzero['qvalue'] = multipletests(nonzero.pvalue, method='fdr_bh')[1]
        return nonzero

    def report_results(self):
        """Summarize and rank gene scores"""
        mcc_df_dict = defaultdict(list)
        for gene, mcc_results in self.raw_results:
            mcc_df_dict[gene].append(gene)
            for clf, mcc in mcc_results.items():
                mcc_df_dict[clf].append(mcc)
        mcc_df = pd.DataFrame(mcc_df_dict).set_index('gene')
        clfs = mcc_df.columns
        mcc_df['mean'] = mcc_df.mean(axis=1)
        mcc_df['std'] = mcc_df[clfs].std(axis=1)
        mcc_df.sort_values('mean', ascending=False, inplace=True)
        mcc_df.to_csv(self.expdir / 'classifier-MCC-summary.csv')
        self.full_results = mcc_df
        stats_df = self.compute_stats()
        stats_df.to_csv(self.expdir / 'meanMCC-results.nonzero-stats.rankings')
        self.nonzero_results = stats_df

    def visualize(self):
        mcc_scatter(self.raw_results, dpi=self.dpi).savefig(self.expdir / f'meanMCC-scatter.png')
        mcc_hist(self.raw_results, dpi=self.dpi).savefig(self.expdir / f'meanMCC-hist.png')
        mcc_scatter(self.nonzero_results, dpi=self.dpi).savefig(self.expdir / 'meanMCC-scatter.nonzero.png')
        mcc_hist(self.nonzero_results, dpi=self.dpi).savefig(self.expdir / 'meanMCC-hist.nonzero.png')
        manhattan_plot(self.nonzero_results, self.reference, dpi=self.dpi).savefig(self.expdir / 'MCC-manhattan.svg')

    def cleanup(self):
        """Cleans tmp directory"""
        shutil.rmtree(self.expdir / 'tmp/')


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
