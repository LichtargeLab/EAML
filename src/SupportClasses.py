#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-01-28

@author: dillonshapiro

These are the commonly used data containers within the pipeline.
"""
from collections import OrderedDict
import numpy as np


class DesignMatrix(object):
    def __init__(self, matrix, feature_labels, id_labels):
        self._matrix = matrix
        self._feature_labels = feature_labels
        self._id_labels = id_labels
        self.X = matrix[:, :-1]
        self.y = matrix[:, -1]
        self._ft_map = OrderedDict((feature, i) for i, feature in
                                   enumerate(feature_labels))
        self._id_map = OrderedDict((ID, i) for i, ID in enumerate(id_labels))

    def update(self, val, ft_label, sample, ft_type='pAFF'):
        ft_idx = self._ft_map[ft_label]
        id_idx = self._id_map[sample]
        if ft_type == 'pAFF':
            self.X[id_idx, ft_idx] *= val
        elif ft_type == 'burden':
            self.X[id_idx, ft_idx] += val
        else:
            raise TypeError('Invalid feature type used.')

    def get_gene(self, gene, hyps=None):
        if hyps:
            col_names = ['_'.join([gene, hyp]) for hyp in hyps]
            col_idxs = [self._ft_map[ft] for ft in col_names]
            col_idxs.append(self._matrix.shape[1] - 1)
            return DesignMatrix(self._matrix[:, col_idxs],
                                self._feature_labels[:, col_idxs],
                                self._id_labels)
        else:
            idx = self._ft_map[gene]
            idx = [idx, self._matrix.shape[1] - 1]
            return DesignMatrix(self._matrix[:, idx],
                                self._feature_labels[:, idx], self._id_labels)


class ConfusionMatrix(object):
    """
    Container for the confusion matrix output from a single trained
    classification experiment.

    Args:
        tp (int): Number of true positive predictions
        tn (int): Number of true negative predictions
        fp (int): Number of false positive predictions
        fn (int): Number of false negative predictions

    Attributes:
        tp (int): Number of true positive predictions
        tn (int): Number of true negative predictions
        fp (int): Number of false positive predictions
        fn (int): Number of false negative predictions
    """
    def __init__(self, y_pred, y_truth):
        self.y_pred = y_pred
        self.y_truth = y_truth
        self.tp = np.sum(y_pred == y_truth == 1)
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def MCC(self):
        """
        Calculates the Matthew Correlation Coefficient.

        The general equation for MCC is:
            ((TP * TN) - (FP * FN)) / (sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN)))

        Returns:
            float: The MCC score for this confusion matrix.
        """
        num = (self.tp * self.tn) - (self.fp * self.fn)
        denom = np.sqrt((self.tp + self.fp) * (self.tp + self.fp) *
                        (self.tn + self.fp) * (self.tn + self.fn))
        return num / denom

    def accuracy(self):
        """
        Calculates the accuracy score.

        The general equation for accuracy is:
            (TP + TN) / (TP + TN + FP + FN)

        Returns:
            float: The accuracy score for this confusion matrix.
        """
        num = self.tp + self.tn
        denom = self.tp + self.tn + self.fp + self.fn
        return num / denom
