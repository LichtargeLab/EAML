#!/usr/bin/env python
import re
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysam import VariantFile
from tqdm import tqdm


def parse_vcf(vcf_fn, reference, samples, n_jobs=1, af_threshold=None, af_field='AF'):
    features = Parallel(n_jobs=n_jobs)(delayed(parse_gene)(vcf_fn, gene, reference.loc[gene], samples,
                                                           af_threshold=af_threshold, af_field=af_field)
                                       for gene in tqdm(reference.index.unique()))
    design_matrix = pd.concat(dict(features), axis=1)
    return design_matrix


def parse_gene(vcf_fn, gene, gene_reference, samples, af_threshold=None, af_field='AF'):
    feature_names = ('D1', 'D30', 'D70', 'R1', 'R30', 'R70')
    ft_cutoffs = list(product((1, 2), (1, 30, 70)))
    vcf = VariantFile(vcf_fn)
    vcf.subset_samples(samples)
    if type(gene_reference) == pd.DataFrame:
        contig = gene_reference['chrom'].iloc[0].strip('chr')
    else:
        contig = gene_reference['chrom'].strip('chr')
    cds_start = gene_reference['cdsStart'].min()
    cds_end = gene_reference['cdsEnd'].max()
    gene_df = pd.DataFrame(np.ones((len(samples), 6)), index=samples, columns=('D1', 'D30', 'D70', 'R1', 'R30', 'R70'))

    variants = fetch_variants(vcf, contig=contig, start=cds_start, stop=cds_end, af_field=af_field)
    for anno_gene, ea, sample_info, af in variants:
        if not ea or gene != anno_gene or (af_threshold and af > af_threshold):
            continue
        gts = pd.Series([convert_zygo(sample_info[sample]['GT']) for sample in samples], index=samples)
        for i, ft_name in enumerate(feature_names):
            cutoff = ft_cutoffs[i]
            for score in ea:
                mask = (score >= cutoff[1]) & (gts >= cutoff[0])
                gene_df[ft_name] *= (1 - (score * mask) / 100)**gts
    return gene, gene_df


# util functions
def fetch_variants(vcf, contig=None, start=None, stop=None, af_field='AF'):
    """
    Iterates over all variants (splitting variants with overlapping gene annotations).

    Args:
        vcf (VariantFile): pysam VariantFile
        contig (str): Chromosome of interest
        start (int): The starting nucleotide position of the region of interest
        stop (int): The ending nucleotide position of the region of interest
        af_field (str): Column containing AF info

    Yields:
        str: Gene name
        ndarray: EA scores corresponding to the gene
        VariantRecordSamples: Container for sample genotypes and info
        float: Allele frequency
    """
    for rec in vcf.fetch(contig=contig, start=start, stop=stop):
        if type(rec.info['gene']) == tuple:  # check for overlapping gene annotations
            for var in _split_genes(rec, af_field=af_field):
                yield var
        else:
            ea = refactor_EA(rec.info['EA'])
            af = _get_af(rec, af_field=af_field)
            yield rec.info['gene'], ea, rec.samples, af


def _get_af(rec, af_field='AF'):
    if af_field == 'AF':
        try:  # if AF isn't annotated, compute using AC and AN fields
            af = rec.info['AF'][0]
        except KeyError:
            af = rec.info['AC'] / rec.info['AN']
    else:
        af = rec.info[af_field]
    return af


def _split_genes(rec, af_field='AF'):
    ea_d = defaultdict(list)
    score = rec.info['EA']
    genes = rec.info['gene']
    af = _get_af(rec, af_field=af_field)
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
        yield gene, ea, rec.samples, af


def refactor_EA(EA):
    """
    Refactors EA to a list of floats and NaN.

    Args:
        EA (list/tuple): The EA scores parsed from a variant

    Returns:
        list: The new list of scores, refactored as float or NaN

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
                continue
        newEA.append(score)
    return newEA


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
