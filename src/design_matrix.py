#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-01-28

@author: dillonshapiro

These are the commonly used data containers within the pipeline.
"""
from collections import OrderedDict


class DesignMatrix(object):
    """
    A container for holding information about a feature matrix.

    Args:
        X (ndarray): The 2D data matrix
        y (ndarray): 1D array of disease status labels corresponding to each
            sample
        feature_labels (list): The names for each column in X
        id_labels (Series): The list of names for each sample

    Attributes:
        X (ndarray)
        y (ndarray)
    """
    def __init__(self, X, y, feature_labels, id_labels):
        self._feature_labels = feature_labels
        self._id_labels = id_labels
        self.X = X
        self.y = y
        self._ft_map = OrderedDict((feature, i) for i, feature in
                                   enumerate(feature_labels))
        self._id_map = OrderedDict((ID, i) for i, ID in enumerate(id_labels))

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
            col_idxs.append(self.X.shape[1] - 1)
            return DesignMatrix(self.X[:, col_idxs], self.y,
                                self._feature_labels[:, col_idxs],
                                self._id_labels)
        else:
            idx = self._ft_map[gene]
            idx = [idx, self.X.shape[1] - 1]
            return DesignMatrix(self.X[:, idx], self.y,
                                self._feature_labels[:, idx], self._id_labels)