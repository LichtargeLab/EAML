#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-01-28

@author: dillonshapiro
"""
import re
import csv


def refactor_EA(EA):
    """
    Refactors EA to a list of floats and None.

    Args:
        EA (tuple): The EA scores parsed from a variant.

    Returns:
        list: The new list of scores, refactored as float or None.
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


def neg_pAFF(EA, zygo):
    return (1 - (EA / 100))**zygo


def check_hyp(zygo, EA, hyp):
    if zygo == 0:
        return False
    hyp = re.split(r'(\d+)', hyp)
    if hyp[0] == 'R' and zygo != 2:
        return False
    if EA >= hyp[1]:
        return True


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


def update_matrix(matrix, sample_gene_d):
    """
    Updates a DesignMatrix given a dictionary of samples to variants.

    Args:
        matrix (DesignMatrix): The DesignMatrix to update
        sample_gene_d (dict): A nested dictionary containing samples as keys
            with genes as values, then a list of variants for each gene key.

    Returns:

    """
    for sample, gene_d in sample_gene_d.items():
        for gene, variants in gene_d.items():
            for var in variants:
                for hyp in ['D1', 'D30', 'D70', 'R1', 'R30', 'R70']:
                    if check_hyp(var.zygo, var.EA, hyp):
                        matrix.update(neg_pAFF(var.EA, var.zygo),
                                      '_'.join([gene, hyp]), sample)


def write_arff(X, y, ft_labels, output):
    """
    Outputs an .arff file corresponding to a feature matrix of interest.

    Args:
        X (ndarray): The feature matrix
        y (ndarray): The classification labels
        ft_labels (list): The label for each feature
        output (str/Path): The filepath for the output
    """
    attrs = []
    examples = []
    attrs.append(['@relation '
                  '{}'.format(str(output).split('/')[-1].split('.')[0])])
    for ft in ft_labels:
        attrs.append(['@attribute {} REAL'.format(ft)])
    attrs.append(['@data'])
    for i, row in X:
        example = [str(x) for x in row]
        example.append(str(y[i]))
        examples.append(example)
    attrs.append(['@attribute class {0,1}'])
    with open(output, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        rows = attrs + examples
        writer.writerows(rows)
