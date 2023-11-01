#!/usr/bin/env python
"""Module for parsing VCFs that have been annotated by a custom VEP & EA annotation pipeline"""
from itertools import product

import numpy as np
import pandas as pd
from pysam import VariantFile

from .utils import validate_EA, af_check, convert_zygo, pEA


def fetch_EA(EA, canon_ensp, all_ensp, csq, EA_parser='canonical'):
    """Parse EA scores for a given variant

    Args:
        EA (tuple): The EA scores parsed from a variant
        canon_ensp (str): Canonical ENSP ID based on VEP criteria
        all_ensp (tuple): All ENSP transcript IDs corresponding to the EA scores
        csq (str): The functional consequence of the variant
        EA_parser (str): How to aggregate multiple transcript EA scores

    Returns:
        float/list: Valid EA scores, refactored as floats
    """
    if 'stop_gained' in csq or 'frameshift_variant' in csq or 'stop_lost' in csq:
        return 100
    if EA_parser == 'canonical':
        try:
            canon_idx = all_ensp.index(canon_ensp)
        except ValueError:
            return np.nan
        else:
            return validate_EA(EA[canon_idx])
    else:
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


def parse_VEP(vcf_fn, gene, gene_ref, samples, min_af=None, max_af=None, af_field='AF', EA_parser='canonical'):
    """Parse EA scores and compute pEA design matrix for a given gene with custom ENSEMBL-VEP annotations

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

    def _fetch_anno(anno):
        # for fields that could return either direct value or tuple depending on header
        if type(anno) == tuple:
            return anno[0]
        else:
            return anno

    for rec in vcf.fetch(contig=str(gene_ref.chrom), start=gene_ref.start, stop=gene_ref.end):
        all_ea = rec.info.get('EA', (None,))
        all_ensp = rec.info.get('Ensembl_proteinid', (rec.info['ENSP'][0],))
        canon_ensp = _fetch_anno(rec.info['ENSP'])
        csq = _fetch_anno(rec.info['Consequence'])
        rec_gene = _fetch_anno(rec.info['SYMBOL'])
        ea = fetch_EA(all_ea, canon_ensp, all_ensp, csq, EA_parser=EA_parser)
        pass_af_check = af_check(rec, af_field=af_field, max_af=max_af, min_af=min_af)
        if not np.isnan(ea).all() and gene == rec_gene and pass_af_check:
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
