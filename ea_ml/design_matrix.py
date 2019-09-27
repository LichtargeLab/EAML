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
    A container for holding information about a design matrix.

    Args:
        X (ndarray): The 2D data matrix
        y (ndarray): 1D array of disease status labels corresponding to each sample
        gene_features (list): The names for each column in X
        id_labels (Series): The list of names for each sample
        feature_names (tuple/list): The EA/zygosity feature labels for each gene

    Attributes:
        X (ndarray)
        y (ndarray)
    """
    def __init__(self, X, y, gene_features, id_labels, feature_names=None):
        self.gene_features = gene_features
        self.feature_names = feature_names
        self.id_labels = id_labels
        self.X = X
        self.y = y
        self._ft_map = OrderedDict((feature, i) for i, feature in enumerate(self.gene_features))
        self._id_map = OrderedDict((ID, i) for i, ID in enumerate(id_labels))

    def __len__(self):
        return self.X.shape[0]

    def update(self, val_arr, gene):
        """
        Updates a specific gene in the X matrix.

        Args:
            val_arr (ndarray): Array of pEA calculations
            gene (str): The gene of interest to modify feature_names of
        """
        if self.feature_names:
            col_names = ['_'.join([gene, ft_name]) for ft_name in self.feature_names]
            col_idxs = [self._ft_map[ft] for ft in col_names]
            self.X[:, col_idxs] *= val_arr
        else:
            col_idx = self._ft_map[gene]
            self.X[:, col_idx] *= val_arr

    def get_gene(self, gene):
        if self.feature_names:
            col_names = ['_'.join([gene, ft_name]) for ft_name in self.feature_names]
            col_idxs = [self._ft_map[ft] for ft in col_names]
            return DesignMatrix(
                self.X[:, col_idxs],
                self.y,
                [self.gene_features[idx] for idx in col_idxs],
                self.id_labels
            )
        else:
            col_idx = self._ft_map[gene]
            return DesignMatrix(
                self.X[:, col_idx],
                self.y,
                self.gene_features[col_idx],
                self.id_labels
            )

    def write_matrix(self, f_out):
        """
        Writes X matrix out as compressed sparse matrix in .npz format.

        Args:
            f_out (Path): Filepath to save compressed matrix to
        """
        sp_matrix = csr_matrix(self.X)
        save_npz(f_out, sp_matrix)

    def write_arff(self, f_out, gene=None, row_idxs=None):
        """
            Outputs an .arff file corresponding to the DesignMatrix object.

            Args:
                f_out (Path): The filepath for the output
                gene (str, optional): Gene to subset from matrix
                row_idxs (ndarray, optional): Specifies specific samples to write to output
            """
        def _write_rows(examples):
            for example, label in examples:
                example = [str(x) for x in example]
                example.append(str(label))
                f.write(','.join(example) + '\n')

        if gene:
            matrix = self.get_gene(gene=gene)
        else:
            matrix = self
        with open(f_out, 'w') as f:
            relation = f_out.stem
            f.write(f'@relation {relation}\n')
            for ft in matrix.gene_features:
                f.write(f'@attribute {ft} REAL\n')
            f.write('@attribute class {0,1}\n')
            f.write('@data\n')
            if row_idxs is not None:
                rows = zip(matrix.X[row_idxs, :], matrix.y[row_idxs])
            else:
                rows = zip(matrix.X, matrix.y)
            _write_rows(rows)
