#!/usr/bin/env python
import numpy as np


def validate_EA(ea):
    """Checks for valid EA score

    Args:
        ea (str/float/None): EA score as string

    Returns:
        float: EA score between 0-100 if valid, otherwise returns NaN
    """
    try:
        ea = float(ea)
    except ValueError:
        if type(ea) == str and (ea == 'fs-indel' or 'STOP' in ea):
            ea = 100
        else:
            ea = np.nan
    except TypeError:
        ea = np.nan
    return ea


def pEA(dmatrix, ea, gts, cutoff, ft_name):
    mask = (ea >= cutoff[1]) & (gts >= cutoff[0])
    dmatrix[ft_name] *= (1 - (ea * mask) / 100) ** gts


def af_check(rec, af_field='AF', max_af=None, min_af=None):
    """Check if variant allele frequency passes filters

    Args:
        rec (VariantRecord)
        af_field (str): Name of INFO field containing allele frequency information
        max_af (float): Maximum allele frequency for variant
        min_af (float): Minimum allele frequency for variant

    Returns:
        bool: True of AF passes filters, otherwise False
    """
    if max_af is None and min_af is None:
        return True
    elif max_af is None:
        max_af = 1
    elif min_af is None:
        min_af = 0
    af = rec.info[af_field]
    if type(af) == tuple:
        af = af[0]
    return min_af < af < max_af


def convert_zygo(genotype):
    """Convert a genotype tuple to a zygosity integer

    Args:
        genotype (tuple): The genotype of a variant for a sample

    Returns:
        int: The zygosity of the variant (0/1/2)
    """
    if genotype in [(1, 0), (0, 1), (None, 1), (1, None)]:
        zygo = 1
    elif genotype == (1, 1):
        zygo = 2
    else:
        zygo = 0
    return zygo
