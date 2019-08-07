#!/usr/bin/env python3
"""
Created on 2019-04-17

@author: dillonshapiro
"""
import signal
import sys
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import StratifiedKFold
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader


def _init_worker(clf_info, folds, hyps):
    """
    Initializes each worker process. This makes the design matrix data available
    as a shared global variable within the pool.

    Args:
        clf_info (DataFrame): DataFrame of Weka classifiers, their Weka
            object names, and hyperparameters.
        folds (list): Train/test indices for kfolds.
        hyps (list): The hypotheses for fetching features.
    """
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
    n_genes = len(genes)
    n_done = 0
    for gene in genes:
        result_d = defaultdict(list)
        for i in range(len(kfolds)):
            # load train and test arffs
            train = str(f'arffs/{gene}_{i}-train.arff')
            test = str(f'arffs/{gene}_{i}-test.arff')
            loader = Loader(classname='weka.core.converters.ArffLoader')
            train_arff = loader.load_file(train)
            test_arff = loader.load_file(test)
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
        n_done += 1
        if n_done % 100 == 0:
            print(f'Worker {wrk_idx}: {n_done} / {n_genes}')
    print(f'Worker {wrk_idx} complete.')
    jvm.stop()


def _append_results(worker_file, gene, gene_results):
    """"Append gene's scores to worker file"""
    with open(worker_file, 'a+') as f:
        for clf, scores in gene_results.items():
            scores = [str(score) for score in scores]
            row = ','.join([gene, clf] + scores)
            f.write(f'{row}\n')


def split_matrix(folds, design_matrix, test_genes, hyps=None):
    if not os.path.exists('arffs/'):
        os.mkdir('arffs')
    for i, (train, test) in enumerate(folds):
        for gene in test_genes:
            design_matrix.write_arff(f'arffs/{gene}_{i}-train.arff', gene=gene,
                                     row_idxs=train, hyps=hyps)
            design_matrix.write_arff(f'arffs/{gene}_{i}-test.arff', gene=gene,
                                     row_idxs=test, hyps=hyps)


def run_weka(design_matrix, test_genes, n_workers, clf_info,
             hyps=None, seed=111, n_splits=10):
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
        n_splits (int): Number of folds for cross-validation.
    """
    jvm.add_bundled_jars()
    # split genes into chunks by number of workers
    gene_splits = enumerate(np.array_split(np.array(test_genes), n_workers))
    # generate KFold groups for samples
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kf.split(design_matrix.X, design_matrix.y))
    # write intermediate train/test arff files for each gene & fold
    split_matrix(splits, design_matrix, test_genes, hyps=hyps)
    # these are set as globals in the worker initializer
    global_args = [clf_info, splits, hyps]
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
