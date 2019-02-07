#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-01-28

@author: dillonshapiro
"""
import re
import csv


def refactor_EA(EA, var_ann, var_type):
    newEA = []
    for score in EA:
        try:
            score = float(score)
        except TypeError:
            if ((re.search(r'frameshift', var_ann) or
                 re.search(r'stop_gained', var_ann) or
                 re.search(r'stop_lost', var_ann)) and
                    var_type == 'protein_coding'):
                score = 100
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
    if genotype in [(1, 0), (0, 1)]:
        zygo = 1
    elif genotype == (1, 1):
        zygo = 2
    else:
        zygo = 0
    return zygo


def update_matrix(matrix, sample_gene_d):
    for sample, gene_d in sample_gene_d.items():
        for gene, variants in gene_d.items():
            for hyp in ['D1', 'D30', 'D70', 'R1', 'R30', 'R70']:
                for var in variants:
                    if check_hyp(var.zygo, var.EA, hyp):
                        matrix.update(neg_pAFF(var.EA, var.zygo),
                                      '_'.join([gene, hyp]), sample)


def write_arff(matrix, ft_labels, output):
    attrs = []
    examples = []
    attrs.append(['@relation '
                  '{}'.format(str(output).split('/')[-1].split('.')[0])])
    for ft in ft_labels:
        attrs.append(['@attribute {} REAL'.format(ft)])
    attrs.append(['@data'])
    for i, row in matrix.X:
        example = [str(x) for x in row]
        example.append(str(matrix.y[i]))
        examples.append(example)
    attrs.append(['@attribute class {0,1}'])
    with open(output, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        rows = attrs + examples
        writer.writerows(rows)
