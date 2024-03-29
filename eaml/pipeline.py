#!/usr/bin/env python
"""Main script for EAML pipeline."""
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
    """The main pipeline object, which stores all input and output data and runs analysis methods.

    Standard use is to instantiate the Pipeline object, then call `Pipeline.run()`, which runs the entire EAML pipeline
    from start to finish. Example files are in the 'examples' directory.

    Args:
        expdir (Path-like): Filepath to experiment directory
        data_fn (Path-like): Filepath to input VCF or pre-computed design matrix
        targets_fn (Path-like): Filepath to two-column CSV with sample IDs and disease status
        reference (str/Path-like): Genome reference version or custom file (hg19, hg38, GRCh37, GRCh38)
        cpus (int): Number of CPUs to use
        kfolds (int): Number of cross-validation folds for each classifier
        seed (int): Random seed for cross-validation
        weka_path (Path-like): Filepath to Weka installation
        min_af (float): Sets minimum allele frequency threshold for including variants
        max_af (float): Sets maximum allele frequency threshold for including variants
        af_field (str): VCF field with allele frequency
        include_X (bool): Includes X chromosome in analysis
        write_data (bool): Keeps design matrix after completed analysis
        parse_EA (str): How to parse EA scores when there are multiple transcripts (max, mean, all, canonical).
            Default is to only use the predefined 'canonical' transcript.
        memory (str): Memory argument for JVM used by Weka
        annotation (str): Variant annotation software used (ANNOVAR, VEP)
    """
    # hyperparameters for Weka classifiers
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

    def __init__(self, expdir, data_fn, targets_fn, reference='hg19', cpus=1, kfolds=10, seed=111, weka_path='~/weka',
                 min_af=None, max_af=None, af_field='AF', include_X=False, write_data=False, parse_EA='canonical',
                 memory='Xmx2g', annotation='ANNOVAR'):
        """Initializes Pipeline from input data"""
        # data arguments
        self.expdir = expdir.expanduser().resolve()
        self.data_fn = data_fn.expanduser().resolve()
        self.targets = pd.read_csv(targets_fn, header=None, dtype={0: str, 1: int}).set_index(0).squeeze().sort_index()
        self.reference = load_reference(reference, include_X=include_X)

        # config arguments
        self.annotation = annotation
        self.kfolds = kfolds
        self.seed = seed
        self.weka_path = weka_path
        self.weka_mem = memory
        self.min_af = min_af
        self.max_af = max_af
        self.af_field = af_field
        self.cpus = cpus
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
        """Computes the full design matrix from an input VCF

        Args:
            gene (str): HGSC gene symbol

        Returns:
            DataFrame: EA design matrix for gene-of-interest
        """
        gene_reference = self.reference.loc[gene]
        if self.annotation == 'ANNOVAR':
            dmatrix = parse_ANNOVAR(self.data_fn, gene, gene_reference, list(self.targets.index), min_af=self.min_af,
                                    max_af=self.max_af, af_field=self.af_field, EA_parser=self.EA_parser)
        else:
            dmatrix = parse_VEP(self.data_fn, gene, gene_reference, list(self.targets.index), min_af=self.min_af,
                                max_af=self.max_af, af_field=self.af_field, EA_parser=self.EA_parser)
        if self.write_data:
            dmatrix_dir = self.expdir / 'dmatrices'
            dmatrix_dir.mkdir(exist_ok=True)
            dmatrix.to_csv(dmatrix_dir / f'{gene}.csv')
        return dmatrix

    def eval_gene(self, gene):
        """Parses input data for a given gene and evaluates it using Weka

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
        self.nonzero_results = compute_stats(self.full_results)
        self.nonzero_results.to_csv(self.expdir / 'meanMCC-results.nonzero-stats.csv')

    def visualize(self):
        """Generate summary figures of EAML results, including a Manhattan plot of p-values and scatterplots and
        histograms of MCC scores
        """
        default_fig_params = {'figure.figsize': (8, 6)}
        mcc_scatter(self.full_results, column='mean', fig_params=default_fig_params)\
            .savefig(self.expdir / f'meanMCC-scatter.pdf')
        mcc_hist(self.full_results, column='mean', fig_params=default_fig_params)\
            .savefig(self.expdir / f'meanMCC-hist.pdf')
        mcc_scatter(self.nonzero_results, column='MCC', fig_params=default_fig_params)\
            .savefig(self.expdir / 'meanMCC-scatter.nonzero.pdf')
        mcc_hist(self.nonzero_results, column='MCC', fig_params=default_fig_params)\
            .savefig(self.expdir / 'meanMCC-hist.nonzero.pdf')
        manhattan_plot(self.nonzero_results, self.reference, fig_params=default_fig_params)\
            .savefig(self.expdir / 'MCC-manhattan.pdf')

    def cleanup(self):
        """Cleans tmp directory"""
        shutil.rmtree(self.expdir / 'tmp/')


def compute_stats(full_results):
    """Generate z-score and p-value statistics for all non-zero MCC scored genes

    Args:
        full_results (DataFrame): DataFrame with all classifier scores and ensemble average, for each gene

    Returns:
        DataFrame: EAML results with non-zero MCCs and corresponding z-scores, p-values, and adjusted p-values
    """
    mcc_df = full_results[['mean', 'std']]
    nonzero = mcc_df.loc[mcc_df[f'mean'] != 0].copy()
    nonzero.rename(columns={'mean': 'MCC'}, inplace=True)
    nonzero['logMCC'] = np.log(nonzero.MCC + 1 - np.min(nonzero.MCC))
    nonzero['zscore'] = (nonzero.logMCC - np.mean(nonzero.logMCC)) / np.std(nonzero.logMCC)
    nonzero['pvalue'] = stats.norm.sf(abs(nonzero.zscore)) * 2  # two-sided test
    nonzero['qvalue'] = multipletests(nonzero.pvalue, method='fdr_bh')[1]
    return nonzero


def load_reference(reference, include_X=False):
    """Loads reference file of gene positions

    Args:
        reference (str): Either human genome reference name (hg19 or hg38), or filepath to custom reference
        include_X (bool): Whether or not to include X chromosome genes in analysis

    Returns:
        DataFrame: DataFrame with chromosome and position information for each annotated transcript
    """
    if reference == 'hg19':
        reference_fn = resource_filename('eaml', 'data/refGene-lite_hg19.May2013.txt')
    elif reference == 'hg38':
        reference_fn = resource_filename('eaml', 'data/refGene-lite_hg38.June2017.txt')
    elif reference == 'GRCh37':
        reference_fn = resource_filename('eaml', 'data/ENSEMBL-lite_GRCh37.v75.txt')
    elif reference == 'GRCh38':
        reference_fn = resource_filename('eaml', 'data/ENSEMBL-lite_GRCh38.v94.txt')
    else:
        reference_fn = reference
    reference_df = pd.read_csv(reference_fn, sep='\t', index_col='gene', dtype={'chrom': str})
    if include_X is False:
        chroms = [str(chrom) for chrom in range(1, 23)]
        reference_df = reference_df[reference_df.chrom.isin(chroms)]
    return reference_df
