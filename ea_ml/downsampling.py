#!/usr/bin/env python
"""
Module for running downsampling experiments to estimate statistical power in comparison to alternative association
methods.

Note: This can be a long-running process, depending on the dataset size.
"""
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

from .pipeline import Pipeline, compute_stats
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

    def eval_gene(self, gene):
        """Parse input data for a given gene and evaluate with Weka repeatedly for each sample size"""
        # first check if gene was scored by whole cohort run
        if gene in self.true_results.index:
            sampled_results = defaultdict(list)
            if self.data_fn.is_dir():
                gene_dmatrix = pd.read_csv(self.data_fn / f'{gene}.csv', index_col=0)
            else:
                gene_dmatrix = self.compute_gene_dmatrix(gene)
            for sample_size in self.sample_sizes:
                splits = downsample_gene(gene_dmatrix, self.targets, sample_size, n_splits=self.n_repeats)
                for split_idx, _ in splits:
                    sub_X = gene_dmatrix.iloc[split_idx]
                    sub_y = self.targets.iloc[split_idx]
                    mcc_results = eval_gene(gene, sub_X, sub_y, self.class_params, seed=self.seed, cv=self.kfolds,
                                            expdir=self.expdir, weka_path=self.weka_path, memory=self.weka_mem)
                    sampled_results[sample_size].append(mcc_results)
                    (self.expdir / f'tmp/{gene}.arff').unlink()
            return gene, sampled_results
        else:
            return None

    def report_results(self):
        # lots of nesting here, is there a better way to do this?
        self.nonzero_results = {n: [] for n in self.sample_sizes}
        for n in self.sample_sizes:
            subdir = self.expdir / str(n)
            subdir.mkdir()
            for i in range(self.n_repeats):
                mcc_df_dict = defaultdict(list)
                for gene, sampled_results in self.raw_results:
                    mcc_df_dict['gene'].append(gene)
                    for clf, mcc in sampled_results[n][i].items():
                        mcc_df_dict[clf].append(mcc)
                mcc_df = pd.DataFrame(mcc_df_dict).set_index('gene')
                clfs = mcc_df.columns
                mcc_df['mean'] = mcc_df.mean(axis=1)
                mcc_df['std'] = mcc_df[clfs].std(axis=1)
                mcc_df.sort_values('mean', ascending=False, inplace=True)
                nonzero_df = compute_stats(mcc_df)
                self.nonzero_results[n].append(nonzero_df)
                nonzero_df.to_csv(subdir / f'meanMCC_results.nonzero-stats.{i}.csv')

    def visualize(self):
        pass


def downsample_gene(X, y, n, n_splits=10):
    return StratifiedShuffleSplit(n_splits=n_splits, train_size=n).split(X, y)
