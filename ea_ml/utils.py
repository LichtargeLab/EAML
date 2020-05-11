#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-01-28

@author: dillonshapiro
"""
import re
from collections import defaultdict

import numpy as np
from scipy.sparse import load_npz


def refactor_EA(EA):
    """
    Refactors EA to a list of floats and NaN.

    Args:
        EA (list/tuple): The EA scores parsed from a variant

    Returns:
        tuple: The new list of scores, refactored as float or NaN

    Note: EA must be string type, otherwise regex search will raise an error.
    """
    newEA = []
    for score in EA:
        try:
            score = float(score)
        except (ValueError, TypeError):
            if re.search(r'fs-indel', score) or re.search(r'STOP', score):
                score = 100
            else:
                score = np.nan
        newEA.append(score)
    return np.array(newEA)


def convert_zygo(genotype):
    """
    Converts a genotype tuple to a zygosity integer.

    Args:
        genotype (tuple): The genotype of a variant for a sample

    Returns:
        int: The zygosity of the variant (0/1/2)
    """
    if genotype in [(1, 0), (0, 1), ('.', 1), (1, '.')]:
        zygo = 1
    elif genotype == (1, 1):
        zygo = 2
    else:
        zygo = 0
    return zygo


def load_matrix(matrix_f):
    """
    Loads an existing numpy matrix.

    Args:
        matrix_f (Path): Path to the compressed matrix

    Returns:
        ndarray: Precomputed design matrix
    """
    if not matrix_f.suffix == '.npz':
        raise Exception(f'{matrix_f} does not use .npz format.')
    sp_matrix = load_npz(matrix_f)
    return sp_matrix.toarray()


def fetch_variants(vcf, contig=None):
    """
    Iterates over all variants (splitting variants with overlapping gene annotations).

    Args:
        vcf (VariantFile): pysam VariantFile
        contig (str): Chromosome of interest

    Yields:
        str: Gene name
        ndarray: EA scores corresponding to the gene
        VariantRecordSamples: Container for sample genotypes and info
    """
    for rec in vcf.fetch(contig=contig):
        if type(rec.info['gene']) == tuple:  # check for overlapping gene annotations
            for var in _split_genes(rec):
                yield var
        else:
            ea = refactor_EA(rec.info['EA'])
            yield rec.info['gene'], ea, rec.samples


def _split_genes(rec):
    ea_d = defaultdict(list)
    score = rec.info['EA']
    genes = rec.info['gene']
    geneset = set(genes)
    if len(score) == len(genes):
        for g, s in zip(genes, score):
            ea_d[g].append(s)
    elif len(score) == 1:
        for g in genes:
            ea_d[g].append(score[0])
    elif len(genes) == 1:
        ea_d[genes[0]] = score
    else:
        raise ValueError('Size of values list does not match expected case.')

    for gene in geneset:
        ea = refactor_EA(ea_d[gene])
        yield gene, ea, rec.samples
