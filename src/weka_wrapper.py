#!/usr/bin/env python3
"""
Created on 2019-04-17

@author: dillonshapiro
"""
import signal
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _weka_worker(gene_list, arff_dir, clf_dict):
    """Worker process for Weka experiments"""
    jvm.start(bundled=False)
    result_d = {}
    for gene in gene_list:
        result_d[gene] = defaultdict(list)
        for i in range(10):
            train = str(arff_dir / '{}_{}-train.arff'.format(gene, i))
            test = str(arff_dir / '{}_{}-test.arff'.format(gene, i))
            loader = Loader(classname='weka.core.converters.ArffLoader')
            train = loader.load_file(train)
            train.class_is_last()
            test = loader.load_file(test)
            test.class_is_last()
            for clf_str, opts in clf_dict.items():
                clf = Classifier(classname=clf_str, options=opts)
                clf.build_classifier(train)
                evl = Evaluation(train)
                _ = evl.test_model(clf, test)
                mcc = evl.matthews_correlation_coefficient(1)
                if np.isnan(mcc):
                    mcc = 0
                result_d[gene][clf_str].append(mcc)
    jvm.stop()
    return result_d


def run_weka(test_genes, n_workers, arff_dir, clf_dict):
    """
    The overall Weka experiment. The steps include loading the K .arff files
    for each gene, building a new classifier based on the training set, then
    evaluating the model on the test set and outputting the MCC to a
    dictionary.
    """
    gene_result_d = {}
    jvm.add_bundled_jars()
    chunks = np.array_split(np.array(test_genes), n_workers)
    pool = Pool(n_workers, _init_worker, maxtasksperchild=1)
    worker = partial(_weka_worker, arff_dir=arff_dir, clf_dict=clf_dict)
    try:
        results = pool.map(worker, chunks)
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
