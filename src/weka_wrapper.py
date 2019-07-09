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
from weka.core.dataset import Instance, Attribute, Instances


def _init_worker(arr, clf_info, folds, hyps):
    """
    Initializes each worker process. This makes the design matrix data available
    as a shared global variable within the pool.

    Args:
        arr (DesignMatrix): The DesignMatrix object containing the feature data.
        clf_info (DataFrame): DataFrame of Weka classifiers, their Weka
            object names, and hyperparameters.
        folds (list): Train/test indices for kfolds.
        hyps (list): The hypotheses for fetching features.
    """
    global data_matrix
    data_matrix = arr
    global clf_calls
    clf_calls = clf_info
    global kfolds
    kfolds = folds
    global hypotheses
    hypotheses = hyps
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _weka_worker(gene_split):
    """
    Worker process for Weka experiments

    Args:
        gene_split (ndarray): The split of genes to test through.
    """
    wrk_idx, genes = gene_split
    jvm.start(bundled=False)
    for gene in genes:
        sub_matrix = data_matrix.get_genes([gene], hyps=hypotheses)
        col_names = sub_matrix.feature_labels
        result_d = defaultdict(list)
        for i, (train, test) in enumerate(kfolds):
            # retrieve fold
            train_X = sub_matrix.X[train, :]
            train_y = sub_matrix.y[train]
            test_X = sub_matrix.X[test, :]
            test_y = sub_matrix.y[test]

            # convert numpy arrays to arff format
            train_arff = ndarray_to_instances(train_X, train_y, gene, col_names)
            test_arff = ndarray_to_instances(test_X, test_y, gene, col_names)
            train_arff.class_is_last()
            test_arff.class_is_last()

            # run each classifier
            for row in clf_calls.itertuples():
                _, clf, clf_str, opts = row
                clf_obj = Classifier(classname=clf_str, options=opts)
                clf_obj.build_classifier(train_arff)
                evl = Evaluation(train_arff)
                _ = evl.test_model(clf_obj, test_arff)
                mcc = evl.matthews_correlation_coefficient(1)
                if np.isnan(mcc):
                    mcc = 0
                result_d[clf].append(mcc)
        _append_results(f'worker-{wrk_idx}.results.csv', gene, result_d)
    jvm.stop()


def _append_results(worker_file, gene, gene_results):
    """"Append gene's scores to worker file"""
    with open(worker_file, 'a+') as f:
        for clf, scores in gene_results.items():
            scores = [str(score) for score in scores]
            row = ','.join([gene, clf] + scores)
            f.write(f'{row}\n')


def ndarray_to_instances(X, y, relation, att_list):
    # reshape y and append to X
    labels = y.reshape((len(y), 1))
    data = np.append(X, labels, axis=1)

    if len(np.shape(data)) != 2:
        raise Exception("Number of array dimensions must be 2!")
    nrows, ncols = np.shape(data)

    # header
    atts = []
    if len(att_list) + 1 != ncols:
        raise Exception(f"Number columns and provided attribute names differ: "
                        f"{ncols} != {len(att_list)}")
    for name in att_list:
        att = Attribute.create_numeric(name)
        atts.append(att)
    atts.append(Attribute.create_nominal('class', ['0', '1']))
    arff = Instances.create_instances(relation, atts, nrows)

    # data
    for i in range(nrows):
        try:
            inst = Instance.create_instance(data[i])
            arff.add_instance(inst)
        except TypeError as e:  # catches and logs any errors with Instance gen.
            with open('error.log', 'a+') as f:
                print(f'Row index at error: {i}')
                print(data[i])
                f.write(f'Instance creation failed at {relation}, row {i}\n')
                f.write(f'{data[i]}\n')
                f.write(f'{e}\n')
            raise e

    return arff


def run_weka(design_matrix, test_genes, n_workers, clf_info,
             hyps=None, seed=111):
    """
    The overall Weka experiment. The steps include loading the K .arff files
    for each gene, building a new classifier based on the training set, then
    evaluating the model on the test set and outputting the MCC to a
    dictionary.

    Args:
        design_matrix (DesignMatrix): The container with all feature and target
            label data.
        test_genes (list): The list of genes being tested.
        n_workers (int): Number of workers to generate in multiprocessing Pool.
        clf_info (DataFrame): Info about classifiers and their parameters.
        hyps (list): EA/variant hypotheses being used as features.
        seed (int): Random seed for generating KFold samples.
    """
    jvm.add_bundled_jars()
    # split genes into chunks by number of workers
    gene_splits = enumerate(np.array_split(np.array(test_genes), n_workers))
    # generate KFold groups for samples
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    splits = list(kf.split(design_matrix.X, design_matrix.y))
    # these are set as globals in the worker initializer
    global_args = [design_matrix, clf_info, splits, hyps]
    pool = Pool(n_workers, initializer=_init_worker, initargs=global_args,
                maxtasksperchild=1)
    try:
        pool.map(_weka_worker, gene_splits)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print('Caught KeyboardInterrupt, terminating workers')
        pool.terminate()
        pool.join()
        sys.exit(1)
