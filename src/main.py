#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 1/21/19

@author: dillonshapiro

Main script for EA-ML pipeline.
"""
import datetime
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from pysam import VariantFile

import utils
from design_matrix import DesignMatrix
from weka_wrapper import run_weka


class Pipeline(object):
    """
    Attributes:
        threads (int): Number of cores to be used by Weka
        expdir (Path): filepath to experiment folder
        data (str): filepath to VCF file
        seed (int): Random seed for KFold sampling
        hypotheses (list): The EA/zygosity hypotheses to use as feature cutoffs.
        tabix (str): filepath to index file for VCF
        targets (np.ndarray): Array of target labels for training/prediction
        samples (list): List of samples to test
        test_genes (list): list of genes to test
        matrix (DesignMatrix): Object for containing feature information for
            each gene and sample.
        result_df (pd.DataFrame): The DataFrame that stores experiment results
        clf_info (DataFrame): A DataFrame mapping classifier names to their
            corresponding Weka object names and hyperparameters.
    """
    def __init__(self, expdir, data, sample_f, gene_list, threads=1, seed=111):
        self.threads = threads
        self.expdir = expdir
        self.data = data
        self.seed = seed
        self.hypotheses = ['D1', 'D30', 'D70', 'R1', 'R30', 'R70']

        # import tabix-indexed file if possible
        if os.path.exists(self.data + '.tbi'):
            self.tabix = self.data + '.tbi'
        else:
            self.tabix = None

        # load feature and sample info
        sample_df = pd.read_csv(sample_f, header=None,
                                dtype={0: str, 1: int})
        self.targets = np.array(sample_df[1])
        self.samples = list(sample_df[0])
        self.test_genes = list(pd.read_csv(gene_list, header=None,
                                           squeeze=True))
        self._ft_labels = self._convert_genes_to_hyp(self.hypotheses)

        # initialize feature matrix
        if self.data.endswith('.npz'):
            arr = utils.load_matrix(self.data)
        else:
            arr = np.ones((len(self.samples), len(self._ft_labels)))
        self.matrix = DesignMatrix(arr, self.targets, self._ft_labels,
                                   self.samples)
        self.result_df = None

        # load classifier information
        pipe_path = os.path.dirname(sys.argv[0])
        self.clf_info = pd.read_csv(pipe_path + '/../classifiers.csv')

    def _convert_genes_to_hyp(self, hyps):
        """
        Converts the test genes to actual feature labels based on test
        hypotheses.

        Returns:
            list: The list of feature labels.
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
            contig (str): A specific contig/chromosome to fetch from the
                VCF. If None, will iterate through the entire VCF file.
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
                    raise ValueError("Length of EA tuple doesn't match "
                                     "expected sizes.")
                g_ea_match = [(g, ea) for g, ea in g_ea_zip if g in
                              self.test_genes]
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
                    raise KeyError("Sample ID {} doesn't exists in "
                                   "VCF.".format(sample))
                zygo = utils.convert_zygo(gt)
                if zygo == 0:
                    continue
                for g, ea in g_ea_match:
                    if ea is not None:
                        for hyp in self.hypotheses:
                            if utils.check_hyp(zygo, ea, hyp):
                                self.matrix.update(
                                    utils.neg_pEA(ea, zygo),
                                    '_'.join([g, hyp]), sample)

    def process_vcf(self):
        """The overall method for processing the entire VCF file."""
        vcf = VariantFile(self.data, index_filename=self.tabix)
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
        run_weka(self.matrix, self.test_genes, self.threads, self.clf_info,
                 hyps=self.hypotheses, seed=self.seed)

    def summarize_experiment(self):
        """Combines results from Weka experiment files."""
        worker_files = Path('.').glob('worker-*.results.csv')
        dfs = [pd.read_csv(fn, header=None) for fn in worker_files]
        result_df = pd.concat(dfs, ignore_index=True)
        result_df.columns = ['gene', 'classifier'] + [str(i) for i in range(10)]
        result_df['meanMCC'] = result_df.mean(axis=1)
        result_df.to_csv(self.expdir / 'gene_MCC_summary.csv')
        self.result_df = result_df

    def write_results(self):
        genes = [k for k in OrderedDict.fromkeys(
            self.result_df.index.get_level_values('gene')).keys()]
        clfs = set(self.result_df.index.get_level_values('classifier'))
        clf_d = OrderedDict([('gene', list(genes))])

        # write summary file for each classifier
        for clf in clfs:
            clf_df = self.result_df.xs(clf, level='classifier')
            clf_df.to_csv(self.expdir / (clf + '-recap.csv'))
            clf_d[clf] = list(clf_df['meanMCC'])

        # aggregate meanMCCs from each classifier
        mean_df = pd.DataFrame.from_dict(clf_d)
        mean_df.to_csv(self.expdir / 'all-classifier_summary.csv',
                       index=False)

        # fetch maxMCC for each gene and write final rankings file
        max_vals = mean_df.max(axis=1)
        final_df = pd.DataFrame({'gene': genes, 'maxMCC': max_vals})
        final_df.sort_values('maxMCC', ascending=False)
        final_df.to_csv(self.expdir / 'maxMCC_summary.csv', index=False)

    def cleanup(self):
        pass


def argparser():
    parser = argparse.ArgumentParser(
        description='Main script for pyEA-ML pipeline.')
    parser.add_argument(
        'run_folder', help='Directory where experiment is being run.')
    parser.add_argument('data', help='VCF file annotated by ANNOVAR and EA, '
                                     'or compressed sparse matrix.')
    parser.add_argument(
        'samples', help='Comma-delimited file with VCF sample '
                        'IDs and corresponding disease status (0 or 1)')
    parser.add_argument('gene_list',
                        help='Single-column list of genes to test.')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads to run Weka on.')
    parser.add_argument('-r', '--seed', type=int, default=111,
                        help='Random seed for generating KFold samples.')

    args = parser.parse_args()
    return args


def main():
    # parse console arguments
    exp_dir, data, sample_f, gene_list, threads, seed = argparser()

    # either load existing design matrix or compute new one from VCF
    pipeline = Pipeline(exp_dir, data, sample_f, gene_list,
                        threads=threads, seed=seed)
    if not data.endswith('.npz'):
        pipeline.process_vcf()
    print('Feature matrix loaded.')

    print('Running experiment...')
    pipeline.run_weka_exp()
    pipeline.summarize_experiment()
    pipeline.write_results()
    print('Gene scoring completed. Analysis summary in experiment directory.')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    elapsed = str(datetime.timedelta(seconds=end - start))
    print('Time elapsed: {}'.format(elapsed))
