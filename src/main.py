#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 1/21/19

@author: dillonshapiro

Main script for EA-ML pipeline.

TODO:
    * convert Weka method to incorporate multi-run experiments
    * add check for existing .arff matrix
    * switch to arff gene-split method to direct numpy-split-arff conversion
"""
import os
import shutil
import sys
import utils
import signal
from itertools import product
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict, OrderedDict
from pathlib import Path
from pysam import VariantFile
from design_matrix import DesignMatrix
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from weka.core.converters import Loader
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
import time
import datetime
load_dotenv(Path('.') / '.env')


class Pipeline(object):
    """
    Attributes:
        expdir (Path): filepath to experiment folder
        resultsdir (Path): filepath to results summary folder
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
        self.resultsdir = self.expdir / 'RESULTS'
        self.arff_dir = self.expdir / 'arffs'
        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)
        if not os.path.exists(self.arff_dir):
            os.mkdir(self.arff_dir)
        if not os.path.exists(self.resultsdir):
            os.mkdir(self.resultsdir)
        self.data = os.getenv("DATA")
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
        utils.write_arff(self.matrix.X, self.matrix.y, self._ft_labels,
                         self.expdir / 'design_matrix.arff')

    def _split_worker(self, args):
        """Worker process for generating a gene's feature matrix split"""
        try:
            gene, (fold, (train_idx, test_idx)) = args
        except ValueError:
            raise ValueError('Not enough arguments in args tuple '
                             'for worker process to unpack.')
        hyps = ['D1', 'D30', 'D70', 'R1', 'R30', 'R70']
        split = self.matrix.get_gene(gene, hyps=hyps)
        ft_labels = ['_'.join([gene, hyp]) for hyp in hyps]
        Xtrain = split.X[train_idx, :]
        ytrain = split.y[train_idx]
        Xtest = split.X[test_idx, :]
        ytest = split.y[test_idx]
        utils.write_arff(Xtrain, ytrain, ft_labels, self.arff_dir / (
            '{}_{}-train.arff'.format(gene, fold)))
        utils.write_arff(Xtest, ytest, ft_labels, self.arff_dir / (
            '{}_{}-test.arff'.format(gene, fold)))

    def split_matrix(self):
        """Splits the DesignMatrix K times for each gene."""
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        folds = list(enumerate(kf.split(self.matrix.X, self.matrix.y)))
        args = list(product(self.test_genes, folds))
        pool = Pool(self.nb_cores, _init_worker)
        try:
            pool.map(self._split_worker, args,
                     chunksize=len(args)//self.nb_cores)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, terminating workers')
            pool.terminate()
            pool.join()
            sys.exit(1)

    def _weka_worker(self, chunk):
        """Worker process for Weka experiments"""
        jvm.start(bundled=False)
        result_d = defaultdict(lambda: defaultdict(list))
        for gene in chunk:
            for i in range(10):
                train = str(self.arff_dir / '{}_{}-train.arff'.format(gene, i))
                test = str(self.arff_dir / '{}_{}-test.arff'.format(gene, i))
                loader = Loader(classname='weka.core.converters.ArffLoader')
                train = loader.load_file(train)
                train.class_is_last()
                test = loader.load_file(test)
                test.class_is_last()
                for clf_str, opts in self.classifiers.items():
                    clf = Classifier(classname=clf_str, options=opts)
                    clf.build_classifier(train)
                    evl = Evaluation(train)
                    _ = evl.test_model(clf, test)
                    mcc = evl.matthews_correlation_coefficient(1)
                    if np.isnan(mcc):
                        mcc = 0
                    result_d[gene][clf_str].append(mcc)
        jvm.stop()
        return result_d

    def run_weka(self):
        """
        The overall Weka experiment. The steps include loading the K .arff files
        for each gene, building a new classifier based on the training set, then
        evaluating the model on the test set and outputting the MCC to a
        dictionary.
        """
        gene_result_d = {}
        jvm.add_bundled_jars()
        chunks = np.array_split(np.array(self.test_genes), self.nb_cores)
        pool = Pool(self.nb_cores, _init_worker, maxtasksperchild=1)
        try:
            results = pool.imap_unordered(self._weka_worker, chunks)
            pool.close()
            progress = 0
            for result in results:
                progress += len(result.keys())
                print('Progress: {} / {}'.format(progress,
                                                 len(self.test_genes)))
                gene_result_d.update(result)
            pool.join()
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, terminating workers')
            pool.terminate()
            pool.join()
            sys.exit(1)

        clf_remap = pd.Series(self.clf_info['classifier'].values,
                              index=self.clf_info['call_string']).to_dict()
        idx = [(gene, clf) for gene in self.test_genes
               for clf in self.classifiers.keys()]
        idx = pd.MultiIndex.from_tuples(idx, names=['gene', 'classifier'])
        df = pd.DataFrame(list(fold for clf in gene_result_d.values() for
                               fold in clf.values()), index=idx)
        df.rename(index=clf_remap, inplace=True)
        df['meanMCC'] = df.mean(axis=1)
        df.to_csv(self.resultsdir / 'gene_MCC_summary.csv')
        self.result_df = df

    def write_results(self):
        genes = [k for k in OrderedDict.fromkeys(
            self.result_df.index.get_level_values('gene')).keys()]
        clfs = set(self.result_df.index.get_level_values('classifier'))
        clf_d = OrderedDict([('gene', list(genes))])
        for clf in clfs:
            clf_df = self.result_df.xs(clf, level='classifier')
            clf_df.to_csv(self.resultsdir / (clf + '-recap.csv'))
            clf_d[clf] = list(clf_df['meanMCC'])
        mean_df = pd.DataFrame.from_dict(clf_d)
        mean_df.to_csv(self.resultsdir / 'all-classifier_summary.csv',
                       index=False)
        max_vals = mean_df.max(axis=1)
        final_df = pd.DataFrame({'gene': genes, 'maxMCC': max_vals})
        final_df.sort_values('maxMCC', ascending=False)
        final_df.to_csv(self.resultsdir / 'maxMCC_summary.csv', index=False)


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    pipeline = Pipeline()
    pipeline.process_vcf()
    print('Feature matrix loaded.')
    print('Splitting matrix by gene...')
    pipeline.split_matrix()
    print('Matrix split complete.')
    print('Running experiment...')
    pipeline.run_weka()
    pipeline.write_results()
    shutil.rmtree(pipeline.arff_dir)  # removes subsetted matrices
    print('Gene scoring completed. Analysis summary in RESULTS/ directory.')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    elapsed = str(datetime.timedelta(seconds=end-start))
    print('Time elapsed: {}'.format(elapsed))
