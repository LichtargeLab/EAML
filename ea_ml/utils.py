#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-01-28

@author: dillonshapiro
"""
import re
from scipy.sparse import load_npz


def refactor_EA(EA):
    """
    Refactors EA to a list of floats and None.

    Args:
        EA (tuple): The EA scores parsed from a variant.

    Returns:
        list: The new list of scores, refactored as float or None.

    Note: EA must be string type, otherwise regex search will raise an error.
    """
    newEA = []
    for score in EA:
        try:
            score = float(score)
        except (ValueError, TypeError):
            if (re.search(r'fs-indel', score) or
                    re.search(r'STOP', score)):
                score = 100
            else:
                score = None
        newEA.append(score)
    return newEA


def neg_pEA(EA, zygo):
    return (1 - (EA / 100))**zygo


def check_hyp(zygo, EA, hyp):
    if zygo == 0:  # ignore wildtype variants
        return False
    hyp = re.split(r'(\d+)', hyp)
    if hyp[0] == 'R' and zygo != 2:
        return False
    return float(EA) >= float(hyp[1])


def convert_zygo(genotype):
    """
    Converts a genotype tuple to a zygosity integer.

    Args:
        genotype (tuple): The genotype of a variant for a sample.

    Returns:
        int: The zygosity of the variant (0/1/2)
    """
    if genotype in [(1, 0), (0, 1)]:
        zygo = 1
    elif genotype == (1, 1):
        zygo = 2
    else:
        zygo = 0
    return zygo

def load_matrix(matrix_f):
    """
    Loads an existing numpy matrix, replacing the existing X attribute.

    Args:
        matrix_f (Path): Path to the compressed matrix.

    Returns:
        ndarray: Precomputed design matrix
    """
    if not matrix_f.suffix == '.npz':
        raise Exception(f'{matrix_f} does not use .npz format.')
    sp_matrix = load_npz(matrix_f)
    return sp_matrix.toarray()
