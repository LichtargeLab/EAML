#!/usr/bin/env python
"""Module for parsing VCFs that have been annotated by a custom ANNOVAR & EA annotation pipeline"""
from itertools import product

import numpy as np
import pandas as pd
from pysam import VariantFile

from .utils import pEA, af_check, convert_zygo, validate_EA


def parse_ANNOVAR(vcf_fn, gene, gene_ref, samples, min_af=None, max_af=None, af_field='AF', EA_parser='canonical'):
    """Parse EA scores and compute pEA design matrix for a given gene with custom ANNOVAR annotations

    Args:
        vcf_fn (Path-like): Filepath to VCF
        gene (str): HGSC gene symbol
        gene_ref (Series): Reference information for given gene's transcripts
        samples (list): sample IDs
        min_af (float): Minimum allele frequency for variants
        max_af (float): Maximum allele frequency for variants
        af_field (str): Name of INFO field containing allele frequency information
        EA_parser (str): How to parse EA scores from multiple transcripts

    Returns:
        DataFrame: pEA design matrix
    """
    feature_names = ('D0', 'D30', 'D70', 'R0', 'R30', 'R70')
    ft_cutoffs = list(product((1, 2), (0, 30, 70)))
    vcf = VariantFile(vcf_fn)
    vcf.subset_samples(samples)
    dmatrix = pd.DataFrame(np.ones((len(samples), 6)), index=samples, columns=feature_names)

    for rec in fetch_variants(vcf, contig=str(gene_ref.chrom), start=gene_ref.start, stop=gene_ref.end):
        ea = fetch_EA(rec.info['EA'], rec.info['NM'], gene_ref.canonical, EA_parser=EA_parser)
        pass_af_check = af_check(rec, af_field=af_field, max_af=max_af, min_af=min_af)
        if not np.isnan(ea).all() and gene == rec.info['gene'] and pass_af_check:
            gts = pd.Series([convert_zygo(rec.samples[sample]['GT']) for sample in samples], index=samples, dtype=int)
            for i, ft_name in enumerate(feature_names):
                cutoff = ft_cutoffs[i]
                if type(ea) == list:
                    for score in ea:
                        if not np.isnan(score):
                            pEA(dmatrix, score, gts, cutoff, ft_name)
                else:
                    pEA(dmatrix, ea, gts, cutoff, ft_name)
    return 1 - dmatrix


# util functions
def fetch_variants(vcf, contig=None, start=None, stop=None):
    """Iterates over variant records for a given region, splitting up records that are annotated with multiple genes.

    Args:
        vcf (VariantFile): pysam VariantFile
        contig (str): Chromosome of interest
        start (int): The starting nucleotide position of the region of interest
        stop (int): The ending nucleotide position of the region of interest

    Yields:
        VariantRecord
    """
    for rec in vcf.fetch(contig=contig, start=start, stop=stop):
        if type(rec.info['gene']) == tuple:  # check for overlapping gene annotations
            for var in split_genes(rec):
                yield var
        else:
            yield rec


def split_genes(rec):
    """If a variant has overlapping gene annotations, it will be split into separate records with correct corresponding
    transcripts, substitutions, and EA scores

    Args:
        rec (VariantRecord)

    Yields:
        VariantRecord
    """
    def _gene_map(gene_idxs_dict, values):
        genes = gene_idxs_dict.keys()
        if len(values) == len([i for gene_idxs in gene_idxs_dict.values() for i in gene_idxs]):
            val_d = {g: [values[i] for i in gene_idxs_dict[g]] for g in genes}
        elif len(values) == 1:
            val_d = {g: values for g in genes}
        else:
            raise ValueError('Size of values list does not match expected case.')
        return val_d

    ea = rec.info['EA']
    gene = rec.info['gene']
    nm = rec.info['NM']
    geneset = set(gene)
    idxs = {genekey: [i for i, g in enumerate(gene) if g == genekey] for genekey in geneset}
    ea_dict = _gene_map(idxs, ea)
    nm_dict = _gene_map(idxs, nm)
    for g in geneset:
        var = rec.copy()
        var.info['gene'] = g
        var.info['EA'] = tuple(ea_dict[g])
        var.info['NM'] = tuple(nm_dict[g])
        yield var


def fetch_EA(EA, nm_ids, canon_nm, EA_parser='canonical'):
    """Parse EA scores for a given variant

    Args:
        EA (tuple): The EA scores parsed from a variant
        nm_ids (tuple): Transcript IDs corresponding to EA scores
        canon_nm (str): Canonical NM ID based on smallest numbered transcript
        EA_parser (str): How to aggregate multiple transcript EA scores

    Returns:
        float/list: Valid EA scores, refactored as floats

    Note: EA must be string type, since our ANNOVAR annotations don't have a separate field with nonsense and fs-indel
        annotations.
    """
    if EA_parser == 'canonical' and canon_nm in nm_ids:
        if len(EA) > 1:
            return validate_EA(EA[nm_ids.index(canon_nm)])
        else:
            return validate_EA(EA[0])
    elif EA_parser != 'canonical':
        newEA = []
        for score in EA:
            newEA.append(validate_EA(score))
        if np.isnan(newEA).all():
            return np.nan
        elif EA_parser == 'mean':
            return np.nanmean(newEA)
        elif EA_parser == 'max':
            return np.nanmax(newEA)
        else:
            return newEA
    else:
        return np.nan
