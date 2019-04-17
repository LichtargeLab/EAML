#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 1/21/19

@author: dillonshapiro

Main script for EA-ML pipeline.

TODO:
    * convert Weka method to incorporate multi-run experiments
    * add check for existing .arff matrix
    * option to write large arff matrix or not
    * switch to arff gene-split method to direct numpy-split-arff conversion
"""
import datetime
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pysam import VariantFile
from sklearn.model_selection import StratifiedKFold

import utils
from design_matrix import DesignMatrix
from weka_wrapper import run_weka

load_dotenv(Path('.') / '.env')


class Pipeline(object):
    """
    Attributes:
        expdir (Path): filepath to experiment folder
        arff_dir (Path): filepath to folder for storing temporary .arff files
        data (str): filepath to VCF file
        tabix (str): filepath to index file for VCF
        targets (np.ndarray): Array of target labels for training/prediction
        samples (list): List of samples to test
        test_genes (list): list of genes to test
        result_df (pd.DataFrame): The DataFrame that stores experiment results
        matrix (DesignMatrix): Object for containing feature information for
            each gene and sample.
        classifiers (list): A list of tuples that maps classifiers from Weka to
            their hyperparameters.
        hypotheses (list): The EA/zygosity hypotheses to use as feature cutoffs.
    """
    def __init__(self):
        # Import env information
        self.nb_cores = int(os.getenv("CORES"))
        self.expdir = Path(os.getenv("EXPDIR"))
        self.data = os.getenv("DATA")
        self.arff_dir = self.expdir / 'arffs'
        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)
        if not os.path.exists(self.arff_dir):
            os.mkdir(self.arff_dir)
        if os.path.exists(self.data + '.tbi'):
            self.tabix = self.data + '.tbi'
        else:
            self.tabix = None
            print('Unable to use multiprocessing for VCF parsing without '
                  'index file.')

        # get feature and sample info
        sample_df = pd.read_csv(os.getenv("SAMPLES"), header=None,
                                dtype={0: str, 1: int})
        sample_df.sort_values(by=0, inplace=True)
        self.targets = np.array(sample_df[1])
        self.samples = list(sample_df[0])
        self.test_genes = sorted(list(pd.read_csv(os.getenv("GENELIST"),
                                                  header=None, squeeze=True)))
        self._ft_labels = self.convert_genes_to_hyp()

        # initialize feature matrix
        matrix = np.ones((len(self.samples), len(self._ft_labels)))
        self.matrix = DesignMatrix(matrix, self.targets, self._ft_labels,
                                   self.samples)
        self.result_df = None

        # map classifiers to hyperparameters
        pipe_path = os.path.dirname(sys.argv[0])
        self.clf_info = pd.read_csv(pipe_path + '/../classifiers.csv')
        clfs = list(self.clf_info['call_string'])
        params = pd.eval(self.clf_info['options'])
        self.classifiers = dict(zip(clfs, params))
        self.hypotheses = ['D1', 'D30', 'D70', 'R1', 'R30', 'R70']

    def convert_genes_to_hyp(self):
        """
        Converts the test genes to actual feature labels based on test
        hypotheses.

        Returns:
            list: The list of feature labels.
        """
        ft_labels = []
        for gene in self.test_genes:
            ft_labels.append(gene + '_D1')
            ft_labels.append(gene + '_D30')
            ft_labels.append(gene + '_D70')
            ft_labels.append(gene + '_R1')
            ft_labels.append(gene + '_R30')
            ft_labels.append(gene + '_R70')
        return ft_labels

    def process_contig(self, vcf, contig=None):
        """
        Reads and updates features in DesignMatrix based on a single contig.

        Args:
            vcf (VariantFile): A VariantFile object
            contig (int, None): A specific contig/chromosome to fetch from the
                VCF. If None, will iterate through the entire VCF file.

        Returns:
            dict: A dictionary of samples, with each sample corresponding to a
                dictionary of gene: variant pairs
        """
        for rec in vcf.fetch(contig=contig):
            gene = rec.info['gene']
            score = utils.refactor_EA(rec.info['EA'])
            if not any(score):
                continue
            # check for overlapping transcripts
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
                                    utils.neg_pAFF(ea, zygo),
                                    '_'.join([g, hyp]), sample)

    def process_vcf(self):
        """The overall method for processing the entire VCF file."""
        vcf = VariantFile(self.data, index_filename=self.tabix)
        for contig in list(range(1, 23)) + ['X', 'Y']:
            print('Processing chromosome {}...'.format(contig))
            self.process_contig(vcf, contig=str(contig))
        self.matrix.X = 1 - self.matrix.X

    def split_matrix(self):
        """Splits the DesignMatrix K times for each gene."""
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        folds = kf.split(self.matrix.X, self.matrix.y)
        for i, (train, test) in enumerate(folds):
            for gene in self.test_genes:
                split = self.matrix.get_gene(gene, hyps=self.hypotheses)
                ft_labels = ['_'.join([gene, hyp]) for hyp in self.hypotheses]
                Xtrain = split.X[train, :]
                ytrain = split.y[train]
                Xtest = split.X[test, :]
                ytest = split.y[test]
                utils.write_arff(Xtrain, ytrain, ft_labels, self.arff_dir / (
                    '{}_{}-train.arff'.format(gene, i)))
                utils.write_arff(Xtest, ytest, ft_labels, self.arff_dir / (
                    '{}_{}-test.arff'.format(gene, i)))

    def run_weka_exp(self):
        gene_result_d = run_weka(self.test_genes, self.nb_cores, self.arff_dir,
                                 self.classifiers)
        clf_remap = pd.Series(self.clf_info['classifier'].values,
                              index=self.clf_info['call_string']).to_dict()
        idx = [(gene, clf) for gene in self.test_genes
               for clf in self.classifiers.keys()]
        idx = pd.MultiIndex.from_tuples(idx, names=['gene', 'classifier'])
        df = pd.DataFrame(list(fold for clf in gene_result_d.values() for
                               fold in clf.values()), index=idx)
        df.rename(index=clf_remap, inplace=True)
        df['meanMCC'] = df.mean(axis=1)
        df.to_csv(self.expdir / 'gene_MCC_summary.csv')
        self.result_df = df

    def write_results(self):
        genes = [k for k in OrderedDict.fromkeys(
            self.result_df.index.get_level_values('gene')).keys()]
        clfs = set(self.result_df.index.get_level_values('classifier'))
        clf_d = OrderedDict([('gene', list(genes))])
        for clf in clfs:
            clf_df = self.result_df.xs(clf, level='classifier')
            clf_df.to_csv(self.expdir / (clf + '-recap.csv'))
            clf_d[clf] = list(clf_df['meanMCC'])
        mean_df = pd.DataFrame.from_dict(clf_d)
        mean_df.to_csv(self.expdir / 'all-classifier_summary.csv',
                       index=False)
        max_vals = mean_df.max(axis=1)
        final_df = pd.DataFrame({'gene': genes, 'maxMCC': max_vals})
        final_df.sort_values('maxMCC', ascending=False)
        final_df.to_csv(self.expdir / 'maxMCC_summary.csv', index=False)

    def cleanup(self):
        shutil.rmtree(self.arff_dir)
        os.remove(self.expdir / 'design_matrix.arff')


def main():
    pipeline = Pipeline()
    pipeline.process_vcf()
    print('Feature matrix loaded.')
    print('Splitting matrix by gene...')
    pipeline.split_matrix()
    print('Matrix split complete.')
    print('Running experiment...')
    pipeline.run_weka_exp()
    pipeline.write_results()
    pipeline.cleanup()
    print('Gene scoring completed. Analysis summary in experiment directory.')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    elapsed = str(datetime.timedelta(seconds=end-start))
    print('Time elapsed: {}'.format(elapsed))
