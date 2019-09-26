#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 1/21/19

@author: dillonshapiro

Main script for EA-ML pipeline.
"""
import datetime
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from pysam import VariantFile

from . import utils
from .design_matrix import DesignMatrix
from .weka_wrapper import run_weka


class Pipeline(object):
    """
    Attributes:
        threads (int): Number of cores to be used by Weka
        expdir (Path): filepath to experiment folder
        data (Path): filepath to VCF file
        seed (int): Random seed for KFold sampling
        hypotheses (list): The EA/zygosity hypotheses to use as feature cutoffs
        tabix (str): filepath to index file for VCF
        targets (np.ndarray): Array of target labels for training/prediction
        samples (list): List of samples to test
        test_genes (list): list of genes to test
        matrix (DesignMatrix): Object for containing feature information for each gene and sample
        result_df (pd.DataFrame): The DataFrame that stores experiment results
        clf_info (DataFrame): A DataFrame mapping classifier names to their corresponding Weka object names and
            hyperparameters
        kfolds (int): Number of folds for cross-validation
    """
    def __init__(self, expdir, data, sample_f, gene_list, threads=1, seed=111, kfolds=10):
        self.threads = threads
        self.expdir = expdir
        self.data = data
        self.seed = seed
        self.kfolds = kfolds
        self.hypotheses = ('D1', 'D30', 'D70', 'R1', 'R30', 'R70')

        # import tabix-indexed file if possible
        if os.path.exists(str(self.data) + '.tbi'):
            self.tabix = Path(str(self.data) + '.tbi')
        else:
            self.tabix = None

        # load feature and sample info
        sample_df = pd.read_csv(sample_f, header=None, dtype={0: str, 1: int}).sort_values(0)
        self.targets = np.array(sample_df[1])
        self.samples = list(sample_df[0])
        self.test_genes = sorted(list(pd.read_csv(gene_list, header=None, squeeze=True)))
        self._ft_labels = self._convert_genes_to_hyp(self.hypotheses)

        # initialize feature matrix
        if self.data.suffix == '.npz':
            arr = utils.load_matrix(self.data)
        else:
            arr = np.ones((len(self.samples), len(self._ft_labels)))
        self.matrix = DesignMatrix(arr, self.targets, self._ft_labels, self.samples)
        self.result_df = None

        # load classifier information
        self.clf_info = pd.read_csv('classifiers.csv', converters={'options': lambda x: x[1:-1].split(',')})
        # Adaboost doesn't work for Leave-One-Out due to it's implicit sample weighting
        if self.kfolds == len(self.samples):
            self.clf_info = self.clf_info[self.clf_info.classifier != 'Adaboost']

    def _convert_genes_to_hyp(self, hyps):
        """
        Converts the test genes to actual feature labels based on test hypotheses.

        Args:
            hyps (list/tuple): EA "hypotheses" being tested

        Returns:
            list: The list of feature labels
        """
        ft_labels = []
        for gene in self.test_genes:
            for hyp in hyps:
                ft_labels.append(f'{gene}_{hyp}')
        return ft_labels

    def process_contig(self, vcf, contig=None):
        """
        Reads and updates features in DesignMatrix based on a single contig.

        Args:
            vcf (VariantFile): A VariantFile object
            contig (str): A specific contig/chromosome to fetch from the VCF. If None, will iterate through the entire
                VCF file
        """
        for rec in vcf.fetch(contig=contig):
            gene = rec.info['gene']
            score = utils.refactor_EA(rec.info['EA'])
            if not any(score):
                continue
            # check for overlapping gene annotations
            if isinstance(gene, tuple):
                if len(gene) == len(score):
                    g_ea_zip = list(zip(gene, score))
                elif len(score) == 1:
                    g_ea_zip = list(zip(gene, score * len(gene)))
                else:
                    raise ValueError("Length of EA tuple doesn't match expected sizes.")
                g_ea_match = [(g, ea) for g, ea in g_ea_zip if g in self.test_genes]
                if not g_ea_match:
                    continue
            else:
                if gene not in self.test_genes:
                    continue
                g_ea_match = list(zip([gene] * len(score), score))
            # check genotypes and update matrix
            for sample in self.samples:
                try:
                    gt = rec.samples[sample]['GT']
                except (IndexError, KeyError):
                    raise KeyError("Sample ID {} doesn't exists in VCF.".format(sample))
                zygo = utils.convert_zygo(gt)
                if zygo == 0:
                    continue
                for g, ea in g_ea_match:
                    if ea is not None:
                        for hyp in self.hypotheses:
                            if utils.check_hyp(zygo, ea, hyp):
                                self.matrix.update(utils.neg_pEA(ea, zygo), '_'.join([g, hyp]), sample)

    def process_vcf(self):
        """The overall method for processing the entire VCF file."""
        vcf = VariantFile(self.data, index_filename=self.tabix)
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
        run_weka(self.expdir, self.matrix, self.test_genes, self.threads, self.clf_info, hyps=self.hypotheses,
                 seed=self.seed, n_splits=self.kfolds)

    def summarize_experiment(self):
        """Combines results from Weka experiment files"""
        worker_files = self.expdir.glob('worker-*.results.csv')
        dfs = [pd.read_csv(fn, header=None) for fn in worker_files]
        result_df = pd.concat(dfs, ignore_index=True)
        if self.kfolds == len(self.samples):
            result_df.columns = ['gene', 'classifier', 'TP', 'TN', 'FP', 'FN', 'MCC']
        else:
            result_df.columns = ['gene', 'classifier'] + [str(i) for i in range(self.kfolds)]
            result_df['meanMCC'] = result_df.mean(axis=1)
        result_df.sort_values('gene', inplace=True)
        result_df.set_index(['gene', 'classifier'], inplace=True)
        result_df.to_csv(self.expdir / 'gene_MCC_summary.csv')
        self.result_df = result_df

    def write_results(self):
        clf_d = OrderedDict([('gene', self.test_genes)])

        # write summary file for each classifier
        for clf in self.clf_info['classifier']:
            clf_df = self.result_df.xs(clf, level='classifier')
            clf_df.to_csv(self.expdir / (clf + '-recap.csv'))
            if self.kfolds == len(self.samples):
                # if LOO, only a single MCC is present per gene
                clf_d[clf] = list(clf_df['MCC'])
            else:
                clf_d[clf] = list(clf_df['meanMCC'])

        # aggregate meanMCCs from each classifier
        mean_df = pd.DataFrame.from_dict(clf_d)
        mean_df.to_csv(self.expdir / 'all-classifier_summary.csv', index=False)

        # fetch maxMCC for each gene and write final rankings file
        max_vals = mean_df.max(axis=1)
        final_df = pd.DataFrame({'gene': self.test_genes, 'maxMCC': max_vals})
        final_df.sort_values('maxMCC', ascending=False, inplace=True)
        final_df.to_csv(self.expdir / 'maxMCC_summary.csv', index=False)

    def cleanup(self):
        """Deletes intermediate worker and ARFF files."""
        for i in range(self.threads):
            os.remove(self.expdir / f'worker-{i}.results.csv')
        shutil.rmtree(self.expdir / 'arffs/')


def run_ea_ml(exp_dir, data, sample_f, gene_list, threads=1, seed=111, kfolds=10):
    # check for JAVA_HOME
    assert os.environ['JAVA_HOME'] is not None

    start = time.time()
    # either load existing design matrix or compute new one from VCF
    pipeline = Pipeline(exp_dir, data, sample_f, gene_list, threads=threads, seed=seed, kfolds=kfolds)
    if not data.suffix == '.npz':
        pipeline.process_vcf()
    print('Feature matrix loaded.')

    print('Running experiment...')
    pipeline.run_weka_exp()
    pipeline.summarize_experiment()
    pipeline.write_results()
    print('Gene scoring completed. Analysis summary in experiment directory.')

    pipeline.cleanup()
    end = time.time()
    elapsed = str(datetime.timedelta(seconds=end - start))
    print(f'Time elapsed: {elapsed}')
