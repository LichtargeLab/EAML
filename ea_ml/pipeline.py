#!/usr/bin/env python
"""Main script for EA-ML pipeline."""
import datetime
import shutil
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from pkg_resources import resource_filename
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .vcf import parse_ANNOVAR, parse_VEP
from .visualize import mcc_hist, mcc_scatter, manhattan_plot
from .weka import eval_gene


class Pipeline:
    # TODO: add Pipeline docstrings
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

    def __init__(self, expdir, data_fn, targets_fn, reference='hg19', cpus=1, kfolds=10, seed=111, dpi=150,
                 weka_path='~/weka', min_af=None, max_af=None, af_field='AF', include_X=False, write_data=False,
                 parse_EA='all', memory='Xmx2g'):
        # data arguments
        self.expdir = expdir.expanduser().resolve()
        self.data_fn = data_fn.expanduser().resolve()
        self.targets = pd.read_csv(targets_fn, header=None, dtype={0: str, 1: int}).set_index(0).squeeze().sort_index()
        self.reference = load_reference(reference, include_X=include_X)

        # config arguments
        self.kfolds = kfolds
        self.seed = seed
        self.weka_path = weka_path
        self.weka_mem = memory
        self.min_af = min_af
        self.max_af = max_af
        self.af_field = af_field
        self.cpus = cpus
        self.dpi = dpi
        self.write_data = write_data
        self.EA_parser = parse_EA

    def run(self):
        """Run full pipeline from start to finish"""
        start = time.time()
        (self.expdir / 'tmp').mkdir(exist_ok=True)
        gene_results = Parallel(n_jobs=self.cpus)(
            delayed(self.eval_gene)(gene) for gene in tqdm(self.reference.index.unique())
        )
        self.raw_results = [result for result in gene_results if result]
        self.report_results()
        print('\nGene scoring completed. Analysis summary in experiment directory.')
        self.visualize()
        self.cleanup()
        end = time.time()
        elapsed = str(datetime.timedelta(seconds=end - start))
        print(f'Time elapsed: {elapsed}')

    def compute_gene_dmatrix(self, gene):
        """
        Computes the full design matrix from an input VCF

        Args:
            gene (str): HGSC gene symbol

        Returns:
            DataFrame: EA design matrix for gene-of-interest
        """
        gene_reference = self.reference.loc[gene]
        dmatrix = parse_gene(self.data_fn, gene, gene_reference, list(self.targets.index), min_af=self.min_af,
                             max_af=self.max_af, af_field=self.af_field, EA_parser=self.EA_parser)
        if self.write_data:
            dmatrix_dir = self.expdir / 'dmatrices'
            dmatrix_dir.mkdir(exist_ok=True)
            dmatrix.to_csv(dmatrix_dir / f'{gene}.csv')
        return dmatrix

    def eval_gene(self, gene):
        """
        Parses input data for a given gene and evaluates it using Weka

        Args:
            gene (str): HGSC gene symbol

        Returns:
            dict(float): Mapping of classifier to MCC from cross validation
        """
        if self.data_fn.is_dir():
            gene_dmatrix = pd.read_csv(self.data_fn / f'{gene}.csv', index_col=0)
        else:
            gene_dmatrix = self.compute_gene_dmatrix(gene)
        if (gene_dmatrix != 0).any().any():
            mcc_results = eval_gene(gene, gene_dmatrix, self.targets, self.class_params, seed=self.seed, cv=self.kfolds,
                                    expdir=self.expdir, weka_path=self.weka_path, memory=self.weka_mem)
            (self.expdir / f'tmp/{gene}.arff').unlink()  # clear intermediate ARFF file after gene scoring completes
            return gene, mcc_results
        else:
            return None

    def compute_stats(self):
        """
        Generate z-score and p-value statistics for all non-zero MCC scored genes

        Returns:
            DataFrame: EA-ML results with non-zero MCCs and computed z-scores, p-values, and adjusted p-values
        """
        mcc_df = self.full_results[['mean', 'std']]
        nonzero = mcc_df.loc[mcc_df[f'mean'] != 0].copy()
        nonzero.rename(columns={'mean': 'MCC'}, inplace=True)
        nonzero['logMCC'] = np.log(nonzero.MCC + 1 - np.min(nonzero.MCC))
        nonzero['zscore'] = (nonzero.logMCC - np.mean(nonzero.logMCC)) / np.std(nonzero.logMCC)
        nonzero['pvalue'] = stats.norm.sf(abs(nonzero.zscore)) * 2  # two-sided test
        nonzero['qvalue'] = multipletests(nonzero.pvalue, method='fdr_bh')[1]
        return nonzero

    def report_results(self):
        """Summarize and rank gene scores"""
        mcc_df_dict = defaultdict(list)
        for gene, mcc_results in self.raw_results:
            mcc_df_dict['gene'].append(gene)
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
        """
        Generate summary figures of EA-ML results, including a Manhattan plot of p-values and scatterplots and
        histograms of MCC scores
        """
        mcc_scatter(self.full_results, column='mean', dpi=self.dpi).savefig(self.expdir / f'meanMCC-scatter.png')
        mcc_hist(self.full_results, column='mean', dpi=self.dpi).savefig(self.expdir / f'meanMCC-hist.png')
        mcc_scatter(self.nonzero_results, column='MCC', dpi=self.dpi).savefig(self.expdir / 'meanMCC-scatter.nonzero.png')
        mcc_hist(self.nonzero_results, column='MCC', dpi=self.dpi).savefig(self.expdir / 'meanMCC-hist.nonzero.png')
        manhattan_plot(self.nonzero_results, self.reference, dpi=self.dpi).savefig(self.expdir / 'MCC-manhattan.svg')

    def cleanup(self):
        """Cleans tmp directory"""
        shutil.rmtree(self.expdir / 'tmp/')


def load_reference(reference, include_X=False):
    """
    Loads reference file of gene positions

    Args:
        reference (str): Either human genome reference name (hg19 or hg38), or filepath to custom reference
        include_X (bool): Whether or not to include X chromosome genes in analysis

    Returns:
        DataFrame: DataFrame with chromosome and position information for each annotated transcript
    """
    if reference == 'hg19':
        reference_fn = resource_filename('ea_ml', 'data/refGene-lite_hg19.txt')
    elif reference == 'hg38':
        reference_fn = resource_filename('ea_ml', 'data/refGene-lite_hg38.txt')
    elif reference == 'GRCh37':
        reference_fn = resource_filename('ea_ml', 'data/ENSEMBL-lite_GRCh37.txt')
    elif reference == 'GRCh38':
        reference_fn = resource_filename('ea_ml', 'data/ENSEMBL-lite_GRCh38.txt')
    else:
        reference_fn = reference
    reference_df = pd.read_csv(reference_fn, sep='\t', index_col='gene', dtype={'chrom': str})
    if include_X is False:
        chroms = [str(chrom) for chrom in range(1, 23)]
        reference_df = reference_df[reference_df.chrom.isin(chroms)]
    return reference_df
