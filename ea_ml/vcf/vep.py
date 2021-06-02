#!/usr/bin/env python
"""Module for parsing VCFs that have been annotated by a custom VEP & EA annotation pipeline"""
from itertools import product

import numpy as np
import pandas as pd
from pysam import VariantFile

from .utils import validate_EA, af_check, convert_zygo, pEA


def fetch_EA(EA, canon_ensp, all_ensp, csq, EA_parser='canonical'):
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
    feature_names = ('D0', 'D30', 'D70', 'R0', 'R30', 'R70')
    ft_cutoffs = list(product((1, 2), (0, 30, 70)))
    vcf = VariantFile(vcf_fn)
    vcf.subset_samples(samples)
    dmatrix = pd.DataFrame(np.ones((len(samples), 6)), index=samples, columns=feature_names)

    for rec in vcf.fetch(contig=str(gene_ref.chrom), start=gene_ref.start, stop=gene_ref.end):
        ea = fetch_EA(rec.info['EA'], rec.info['ENSP'][0], rec.info['Ensembl_proteinid'][0], rec.info['Consequence'][0],
                      EA_parser=EA_parser)
        pass_af_check = af_check(rec, af_field=af_field, max_af=max_af, min_af=min_af)
        if not np.isnan(ea).all() and gene == rec.info['SYMBOL'][0] and pass_af_check:
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
