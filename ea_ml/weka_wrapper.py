#!/usr/bin/env python
import signal
import sys
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import StratifiedKFold
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader
from weka.core.classes import Random


# noinspection PyGlobalUndefined
def _init_worker(expdir, targets, clf_info, n_splits, kf_splits, seed=111):
    """
    Initializes each worker process. This makes some information available as shared memory.

    Args:
        expdir (Path): Path to experiment directory
        targets (Series): target labels
        clf_info (DataFrame): DataFrame of Weka classifiers, their Weka object names, and hyperparameters
        n_splits (int): Number of splits used for cross-validation
        kf_splits (list): List of tuples containing train/test indices for each fold
        seed (int): The random seed used for sampling
    """
    global exp_dir
    exp_dir = expdir
    global y
    y = targets
    global clf_calls
    clf_calls = clf_info
    global n_folds
    n_folds = n_splits
    global k_splits
    k_splits = kf_splits
    global rseed
    rseed = seed
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _weka_worker(args):
    wrk_idx, (genes, X) = args
    jvm.start(bundled=False)
    n_genes = len(genes)
    n_done = 0
    for gene in genes:
        result_d = defaultdict(list)
        gene_X = X[gene]
        for i, (train, test) in enumerate(k_splits):
            # write intermediate train and test arffs
            train_fn = exp_dir / 'tmp' / f'{gene}_{i}-train.arff'
            test_fn = exp_dir / 'tmp' / f'{gene}_{i}-test.arff'
            write_arff(gene_X.iloc[train], y.iloc[train], gene, train_fn, row_idxs=train)
            write_arff(gene_X.iloc[test], y.iloc[test], gene, test_fn, row_idxs=test)

            # load train and test arffs
            loader = Loader(classname='weka.core.converters.ArffLoader')
            train_arff = loader.load_file(str(train_fn))
            test_arff = loader.load_file(str(test_fn))
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

            train_fn.unlink()
            test_fn.unlink()

        _append_results(exp_dir / 'tmp' / f'worker-{wrk_idx}.results.csv', gene, result_d)
        n_done += 1
        if n_done % 100 == 0:
            print(f'Worker {wrk_idx}: {n_done} / {n_genes}')
    print(f'Worker {wrk_idx} complete.')
    jvm.stop()


def _loo_worker(args):
    wrk_idx, (genes, X) = args
    jvm.start(bundled=False)
    n_genes = len(genes)
    n_done = 0
    for gene in genes:
        result_d = {}
        gene_arff_fn = exp_dir / 'tmp' / f'{gene}.arff'
        write_arff(X, y, gene, gene_arff_fn)
        loader = Loader(classname='weka.core.converters.ArffLoader')
        arff = loader.load_file(f'{str(exp_dir)}/tmp/{gene}.arff')
        arff.class_is_last()
        # run each classifier
        for row in clf_calls.itertuples():
            _, clf, clf_str, opts = row
            clf_obj = Classifier(classname=clf_str, options=opts)
            evl = Evaluation(arff)
            evl.crossvalidate_model(clf_obj, arff, len(arff), Random(rseed))
            mcc = evl.matthews_correlation_coefficient(1)
            if np.isnan(mcc):
                mcc = 0
            tp = evl.num_true_positives(1)
            tn = evl.num_true_negatives(1)
            fp = evl.num_false_positives(1)
            fn = evl.num_false_negatives(1)
            result_d[clf] = (tp, tn, fp, fn, mcc)
        _append_results(exp_dir / 'tmp' / f'worker-{wrk_idx}.results.csv', gene, result_d)
        gene_arff_fn.unlink()
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


def run_weka(expdir, design_matrix, targets, n_workers, clf_info, seed=111, n_splits=10):
    """
    The overall Weka experiment. The steps include loading the K .arff files for each gene, building a new classifier
    based on the training set, then evaluating the model on the test set and outputting the MCC to a dictionary.

    Args:
        expdir (Path): Path to experiment directory.
        design_matrix (DataFrame): design matrix computed from VCF
        targets (Series): target labels
        n_workers (int): Number of workers to generate in multiprocessing Pool.
        clf_info (DataFrame): Info about classifiers and their parameters.
        seed (int): Random seed for generating KFold samples.
        n_splits (int): Number of folds for cross-validation.
    """
    (expdir / 'tmp').mkdir(exist_ok=True)
    jvm.add_bundled_jars()
    # split genes into chunks by number of workers
    gene_splits = np.array_split(list(design_matrix.columns.get_level_values(0).unique()), n_workers)
    if n_splits != -1:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        kf_splits = list(kf.split(design_matrix, targets))
    else:
        kf_splits = None
    matrix_splits = []
    for split in gene_splits:
        matrix_splits.append(design_matrix[split])
    worker_args = enumerate(list(zip(gene_splits, matrix_splits)))

    # these are set as globals in the worker initializer
    global_args = [expdir, targets, clf_info, n_splits, kf_splits, seed]
    pool = Pool(n_workers, initializer=_init_worker, initargs=global_args, maxtasksperchild=1)
    try:
        if n_splits == -1:
            pool.map(_loo_worker, worker_args)
        else:
            pool.map(_weka_worker, worker_args)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print('Caught KeyboardInterrupt, terminating workers')
        pool.terminate()
        pool.join()
        sys.exit(1)


def write_arff(design_matrix, targets, gene, f_out, row_idxs=None):
    """
        Outputs an .arff file corresponding to the DataFrame

        Args:
            design_matrix (DataFrame): design matrix from VCF
            targets (Series): disease labels
            gene (str): gene to subset from matrix
            f_out (Path): filepath for the output
            row_idxs (list-like): specifies specific samples to write to output
        """
    def _write_rows(examples):
        for example, label in examples:
            example = [str(x) for x in example]
            example.append(str(label))
            f.write(','.join(example) + '\n')

    gene_X = design_matrix[gene]
    with open(f_out, 'w') as f:
        relation = f_out.stem
        f.write(f'@relation {relation}\n')
        for ft in gene_X.columns:
            f.write(f'@attribute {ft} REAL\n')
        f.write('@attribute class {0,1}\n')
        f.write('@data\n')
        if row_idxs:
            rows = zip(gene_X.iloc[row_idxs].values, targets.iloc[row_idxs].values)
        else:
            rows = zip(gene_X.values, targets.values)
        _write_rows(rows)
