#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-01-28

@author: dillonshapiro

These are the commonly used data containers within the pipeline.
"""
from collections import OrderedDict
from scipy.sparse import save_npz, csr_matrix


class DesignMatrix(object):
    """
    A container for holding information about a feature matrix.

    Args:
        X (ndarray): The 2D data matrix
        y (ndarray): 1D array of disease status labels corresponding to each sample
        feature_labels (list): The names for each column in X
        id_labels (Series): The list of names for each sample

    Attributes:
        X (ndarray)
        y (ndarray)
    """
    def __init__(self, X, y, feature_labels, id_labels):
        self.feature_labels = feature_labels
        self.id_labels = id_labels
        self.X = X
        self.y = y
        self._ft_map = OrderedDict((feature, i) for i, feature in enumerate(feature_labels))
        self._id_map = OrderedDict((ID, i) for i, ID in enumerate(id_labels))

    def __len__(self):
        return self.X.shape[0]

    def update(self, val, ft_label, sample):
        """
        Updates a specific cell in the X matrix.

        Args:
            val (float): value to modify feature by
            ft_label (str): The label of the feature column (i.e. gene name)
            sample (str): The specific sample ID corresponding to a example row
        """
        ft_idx = self._ft_map[ft_label]
        id_idx = self._id_map[sample]
        self.X[id_idx, ft_idx] *= val

    def get_gene(self, gene, hyps=None):
        if hyps:
            col_names = ['_'.join([gene, hyp]) for hyp in hyps]
            col_idxs = [self._ft_map[ft] for ft in col_names]
            return DesignMatrix(
                self.X[:, col_idxs],
                self.y,
                [self.feature_labels[idx] for idx in col_idxs],
                self.id_labels
            )
        else:
            col_idx = self._ft_map[gene]
            return DesignMatrix(
                self.X[:, col_idx],
                self.y,
                self.feature_labels[col_idx],
                self.id_labels
            )

    def write_matrix(self, f_out):
        """
        Writes X matrix out as compressed sparse matrix in .npz format.

        Args:
            f_out (Path): Filepath to save compressed matrix to.
        """
        sp_matrix = csr_matrix(self.X)
        save_npz(f_out, sp_matrix)

    def write_arff(self, f_out, gene=None, row_idxs=None, hyps=None):
        """
            Outputs an .arff file corresponding to the DesignMatrix object.

            Args:
                f_out (Path): The filepath for the output
                gene (str, optional): Gene to subset from matrix.
                row_idxs (ndarray, optional): Specifies specific samples to write to output.
                hyps (list): EA/variant hypotheses being used as features.
            """
        def _write_rows(examples):
            for example, label in examples:
                example = [str(x) for x in example]
                example.append(str(label))
                f.write(','.join(example) + '\n')

        if gene:
            matrix = self.get_gene(gene=gene, hyps=hyps)
        else:
            matrix = self
        with open(f_out, 'w') as f:
            relation = f_out.stem
            f.write(f'@relation {relation}\n')
            for ft in matrix.feature_labels:
                f.write(f'@attribute {ft} REAL\n')
            f.write('@attribute class {0,1}\n')
            f.write('@data\n')
            if row_idxs is not None:
                rows = zip(matrix.X[row_idxs, :], matrix.y[row_idxs])
            else:
                rows = zip(matrix.X, matrix.y)
            _write_rows(rows)
