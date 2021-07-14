#!/usr/bin/env python
"""
Module for running downsampling experiments to estimate statistical power in comparison to alternative association
methods.

Note: This can be a long-running process, depending on the dataset size.

Steps:
calculate all matrices
for each sample size:
    subset matrix and targets
    run ea-ml

"""
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import train_test_split

from .pipeline import Pipeline
from .weka import eval_gene


class DownsamplingPipeline(Pipeline):
    def __init__(self, expdir, data_fn, targets_fn, true_results, sample_sizes, n_repeats=10, reference='hg19',
                 cpus=1, kfolds=10, seed=111, dpi=150, weka_path='~/weka', min_af=None, max_af=None, af_field='AF',
                 include_X=False, parse_EA='canonical', memory='Xmx2g', annotation='ANNOVAR'):
        super().__init__(expdir, data_fn, targets_fn, reference=reference, cpus=cpus, kfolds=kfolds, seed=seed, dpi=dpi,
                         weka_path=weka_path, min_af=min_af, max_af=max_af, af_field=af_field, include_X=include_X,
                         parse_EA=parse_EA, memory=memory, annotation=annotation)
        self.sample_sizes = sample_sizes
        self.n_repeats = n_repeats
        self.true_results = true_results
        self.write_data = False

    def run(self):
        start = time.time()
        (self.expdir / 'tmp').mkdir(exist_ok=True)

    def eval_gene(self, gene):
        """Parse input data for a given gene and evaluate with Weka repeatedly for each sample size"""

        if self.data_fn.is_dir():
            gene_dmatrix = pd.read_csv(self.data_fn / f'{gene}.csv', index_col=0)
        else:
            gene_dmatrix = self.compute_gene_dmatrix(gene)
        for sample_size in self.sample_sizes:
            sub_X, sub_y = downsample_matrix(gene_dmatrix, self.targets, sample_size)
            mcc_results = eval_gene(gene, sub_X, sub_y, self.class_params, seed=self.seed, cv=self.kfolds,
                                    expdir=self.expdir, weka_path=self.weka_path, memory=self.weka_mem)


    def compute_stats(self):
        pass

    def report_results(self):
        pass


def downsample_matrix(X, y, n):
    sub_X, _, sub_y, _ = train_test_split(X, y, train_size=n, stratify=y)
    return sub_X, sub_y
