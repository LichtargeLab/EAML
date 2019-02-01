#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 1/21/19

@author: dillonshapiro

pyEA-ML
General Description:HERE

Example:
    Use reStructuredText here. For example::

        $ python Pyea-Ml.py

TODO:
    * convert process_vcf to multiprocessing
    * parse Weka result output
"""
# Import env information
import os
import utils
import pandas as pd
import numpy as np
from collections import defaultdict, namedtuple
from pathlib import Path
from pysam import VariantFile
from SupportClasses import DesignMatrix
from dotenv import load_dotenv
import weka.core.converters as converters
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.experiments import SimpleCrossValidationExperiment
load_dotenv()


class Pipeline(object):
    """


    Attributes:

    """
    def __init__(self):
        self.expdir = Path(os.getenv("EXPERIMENTDIR"))
        self.resultsdir = self.expdir / 'RESULTS'
        self.data = os.getenv("DATA")
        self.samples = pd.read_csv(os.getenv("SAMPLES"), header=None)
        self.test_genes = list(pd.read_csv(os.getenv("GENELIST"), header=None))
        self.arff_dir = self.expdir / 'arffs'
        if os.path.exists(self.data + '.tbi'):
            self._tabix = self.data + '.tbi'
        else:
            self._tabix = None
            print('Unable to use multiprocessing for VCF parsing without '
                  'index file.')
        self._ft_labels = self.convert_genes_to_hyp()
        matrix = np.ones((len(self.samples), len(self._ft_labels) + 1))
        matrix[:, -1] = self.samples[1]
        self.matrix = DesignMatrix(matrix, self._ft_labels, self.samples[0])
        if not os.path.exists(self.arff_dir):
            os.mkdir(self.arff_dir)
        if not os.path.exists(self.resultsdir):
            os.mkdir(self.resultsdir)
        classifiers = [
            'weka.classifiers.rules.PART',
            'weka.classifiers.rules.JRip',
            'weka.classifiers.trees.J48',
            'weka.classifiers.trees.RandomForest',
            'weka.classifiers.bayes.NaiveBayes',
            'weka.classifiers.functions.Logistic',
            'weka.classifiers.meta.AdaBoostM1',
            'weka.classifiers.functions.MultiLayerPerceptron',
            'weka.classifiers.lazy.IBk'
        ]
        params = [
            ['-M', '5', '-C', '0.25', '-Q', '1'],
            ['-F', '3', '-N', '2.0', '-O', '2', '-S', '1', '-P'],
            ['-C', '0.25', '-M', '5'],
            ['-I', '10', '-K', '0', '-S', '1'],
            [''],
            ['-R', '1.0E-8', '-M', '-1'],
            ['-P', '100', '-S', '1', '-I', '10', '-W',
             'weka.classifiers.trees.DecisionStump'],
            ['-L', '0.3', '-M', '0.2', '-N', '500', '-V', '0', '-S', '0', '-E',
             '20', '-H', 'a'],
            ['-K', '3', '-W', '0', '-A',
             'weka.core.neighboursearch.LinearNNSearch', '-A',
             'weka.core.EuclideanDistance', '-R', 'first-last']
        ]
        self.classifiers = dict(zip(classifiers, params))

    def convert_genes_to_hyp(self):
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
        variant = namedtuple('Variant', ['EA', 'zygo'])
        sample_gene_d = defaultdict(lambda: defaultdict(list))
        for rec in vcf.fetch(contig=contig):
            gene = rec.info['gene']
            EA = utils.refactor_EA(rec.info['EA'])
            if not EA or gene not in self.test_genes:
                continue
            for sample, info in rec.samples.items():
                zygo = utils.convert_zygo(info['GT'])
                if zygo == 0:
                    continue
                for score in EA:
                    sample_gene_d[sample][gene].append(variant(score, zygo))
        return sample_gene_d

    def process_vcf(self, threads=1):
        vcf = VariantFile(self.data, index_filename=self._tabix)
        for contig in range(1, 23):
            print('Processing chromosome {}...'.format(contig))
            sample_gene_d = self.process_contig(vcf, contig=str(contig))
            utils.update_matrix(self.matrix, sample_gene_d)

    def split_matrix(self):
        for gene in self.test_genes:
            hyps = ['D1', 'D30', 'D70', 'R1', 'R30', 'R70']
            split = self.matrix.get_gene(gene, hyps=hyps)
            ft_labels = ['_'.join([gene, hyp]) for hyp in hyps]
            utils.write_arff(split, ft_labels, self.arff_dir / (gene + '.arff'))

    def run_weka(self, n_runs=1):
        datasets = [str(arff) for arff in self.arff_dir.glob('*.arff')]
        classifiers = [Classifier(classname=clf, options=opts) for
                       clf, opts in self.classifiers.items()]
        result = str(self.resultsdir / 'full_results-cv.arff')
        jvm.start()
        exp = SimpleCrossValidationExperiment(
            classification=True,
            runs=n_runs,
            folds=10,
            datasets=datasets,
            classifiers=classifiers,
            result=result
        )
        exp.setup()
        exp.run()
        loader = converters.loader_for_file(result)
        cv_results = loader.load_file(result)
        jvm.stop()


def main():
    pipeline = Pipeline()
    pipeline.process_vcf(os.getenv("NB_CORES"))
    print('Feature matrix loaded.')
    pipeline.split_matrix()
    print('Matrix split by gene.')
    print('Running experiment...')
    pipeline.run_weka()
    print('Gene scoring completed. Analysis summary in RESULTS/ directory.')
