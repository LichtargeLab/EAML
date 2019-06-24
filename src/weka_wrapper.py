#!/usr/bin/env python3
"""
Created on 2019-04-17

@author: dillonshapiro
"""
import signal
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import StratifiedKFold
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import ndarray_to_instances
from weka.filters import Filter
from weka.core.dataset import Instance


def _init_worker(arr, clf_dict, folds, hyps):
    """
    Initializes each worker process. This makes the design matrix data available
    as a shared global variable within the pool.

    Args:
        arr (DesignMatrix): The DesignMatrix object containing the feature data.
        clf_dict (dict): Dictionary of Weka classifier call and its options.
        folds (list): Train/test indices for kfolds.
        hyps (list): The hypotheses for fetching features.
    """
    global data_matrix
    data_matrix = arr
    global clf_calls
    clf_calls = clf_dict
    global kfolds
    kfolds = folds
    global hypotheses
    hypotheses = hyps
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _weka_worker(gene_split):
    """Worker process for Weka experiments"""
    jvm.start(bundled=False)
    result_d = {}
    for gene in gene_split:
        sub_matrix = data_matrix.get_genes([gene], hyps=hypotheses)
        col_names = sub_matrix.feature_labels
        col_names.append('class')
        result_d[gene] = defaultdict(list)
        for i, (train, test) in enumerate(kfolds):
            train_X = sub_matrix.X[train, :]
            train_y = sub_matrix.y[train]
            test_X = sub_matrix.X[test, :]
            test_y = sub_matrix.y[test]

            train_arff = convert_array(train_X, train_y, gene, col_names)
            test_arff = convert_array(test_X, test_y, gene, col_names)

            for clf_str, opts in clf_calls.items():
                clf = Classifier(classname=clf_str, options=opts)
                clf.build_classifier(train_arff)
                evl = Evaluation(train_arff)
                _ = evl.test_model(clf, test_arff)
                mcc = evl.matthews_correlation_coefficient(1)
                if np.isnan(mcc):
                    mcc = 0
                result_d[gene][clf_str].append(mcc)
    jvm.stop()
    return result_d


def convert_array(X, y, gene, col_names):
    # reshape y and append to X
    labels = y.reshape((len(y), 1))
    data = np.append(X, labels, axis=1)
    # convert to ARFF format
    try:
        arff = ndarray_to_instances(data, gene, att_list=col_names)
    except TypeError as e:  # catches and logs any errors with Instance creation
        with open('error.log', 'a+') as f:
            error_idx = _instance_error_check(data)
            f.write(f'Instance creation failed at {gene}, row {error_idx}\n')
            f.write(f'{data[error_idx]}\n')
            f.write(f'{e}\n')
        raise e

    # convert label attribute to nominal type
    nominal = Filter(
        classname='weka.filters.unsupervised.attribute.NumericToNominal',
        options=['-R', 'last'])
    nominal.inputformat(arff)
    arff_reform = nominal.filter(arff)
    arff_reform.class_is_last()
    return arff_reform


def _instance_error_check(array):
    rows = array.shape[0]
    for i in range(rows):
        try:
            Instance.create_instance(array[i])
        except TypeError:
            print(f'Row index at error: {i}')
            print(array[i])
            return i


def run_weka(design_matrix, test_genes, n_workers, clf_dict,
             hyps=None, seed=111):
    """
    The overall Weka experiment. The steps include loading the K .arff files
    for each gene, building a new classifier based on the training set, then
    evaluating the model on the test set and outputting the MCC to a
    dictionary.
    """
    gene_result_d = {}
    jvm.add_bundled_jars()
    # split genes into chunks by number of workers
    gene_splits = np.array_split(np.array(test_genes), n_workers)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    splits = list(kf.split(design_matrix.X, design_matrix.y))
    # these are set as globals in the worker initializer
    global_args = [design_matrix, clf_dict, splits, hyps]
    pool = Pool(n_workers, initializer=_init_worker, initargs=global_args,
                maxtasksperchild=1)
    try:
        results = pool.map(_weka_worker, gene_splits)
        pool.close()
        pool.join()
        for result in results:
            gene_result_d.update(result)
    except KeyboardInterrupt:
        print('Caught KeyboardInterrupt, terminating workers')
        pool.terminate()
        pool.join()
        sys.exit(1)
    return gene_result_d
