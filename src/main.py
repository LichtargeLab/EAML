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
    * complete matrix splitting method
    * write Weka experiment code
"""
# Import env information
import os
import utils
import pandas as pd
import numpy as np
from collections import defaultdict, namedtuple
from pysam import VariantFile
from SupportClasses import DesignMatrix
from dotenv import load_dotenv
load_dotenv()


class Pipeline(object):
    """


    Attributes:

    """
    def __init__(self):
        self.expdir = os.getenv("EXPERIMENTDIR")
        self.data = os.getenv("DATA")
        self.samples = pd.read_csv(os.getenv("SAMPLES"), header=None)
        self.test_genes = list(pd.read_csv(os.getenv("GENELIST"), header=None))
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
            sample_gene_d = self.process_contig(vcf, contig=contig)
            utils.update_matrix(self.matrix, sample_gene_d)

    def split_matrix(self):
        pass

    def run_weka(self):
        return True


def main():
    pipeline = Pipeline()
    pipeline.process_vcf(os.getenv("NB_CORES"))
    print('Feature matrix loaded.')
    pipeline.split_matrix()
    print('Matrix split by gene.')
    print('Running experiment...')
    pipeline.run_weka()
    print('Gene scoring completed. Analysis summary in RESULTS/ directory.')
