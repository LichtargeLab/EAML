#!/usr/bin/env python
"""Module for adapting EAML to pathways."""
import datetime
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from .pipeline import Pipeline, compute_stats
from .visualize import mcc_hist, mcc_scatter


class PathwayPipeline(Pipeline):
    def __init__(self, expdir, data_fn, targets_fn, pathways_fn, reference='hg19', cpus=1, kfolds=10, seed=111,
                 weka_path='~/weka', min_af=None, max_af=None, af_field='AF', include_X=False, write_data=False,
                 parse_EA='canonical', memory='Xmx2g', annotation='ANNOVAR'):
        super().__init__(expdir, data_fn, targets_fn, reference=reference, cpus=cpus, kfolds=kfolds, seed=seed,
                         weka_path=weka_path, min_af=min_af, max_af=max_af, af_field=af_field, include_X=include_X,
                         parse_EA=parse_EA, memory=memory, annotation=annotation, write_data=write_data)
        self.pathways_map, self.pathway_descriptions = load_pathways(pathways_fn, self.reference)
        self.write_pathway_data = write_data
        self.write_data = False

    def run(self):
        """Run full pipeline from start to finish"""
        start = time.time()
        (self.expdir / 'tmp').mkdir(exist_ok=True)
        results = Parallel(n_jobs=self.cpus)(
            delayed(self.eval_feature)(pathway) for pathway in tqdm(list(self.pathways_map.keys()))
        )
        self.raw_results = [result for result in results if result]
        self.report_results()
        print('\nPathway scoring completed. Analysis summary in experiment directory.')
        self.visualize()
        self.cleanup()
        end = time.time()
        elapsed = str(datetime.timedelta(seconds=end - start))
        print(f'Time elapsed: {elapsed}')

    def compute_pathway_dmatrix(self, pathway):
        """
        Computes design matrix for given pathway from an input VCF

        Args:
            pathway (str): Pathway name

        Returns:
            DataFrame: EA design matrix for pathway-of-interest
        """
        feature_names = ['D0', 'D30', 'D70', 'R0', 'R30', 'R70']
        dmatrix = pd.DataFrame(np.ones((len(self.targets), 6)), index=self.targets.index, columns=feature_names)
        gene_list = self.pathways_map[pathway]

        for gene in gene_list:
            # need to resubtract sub-matrix since it's subtracted in parser
            dmatrix *= (1 - self.compute_gene_dmatrix(gene))
        dmatrix = 1 - dmatrix
        if self.write_pathway_data:
            dmatrix_dir = self.expdir / 'dmatrices'
            dmatrix_dir.mkdir(exist_ok=True)
            dmatrix.to_csv(dmatrix_dir / f'{pathway}.csv')
        return dmatrix

    def compute_dmatrix(self, feature):
        return self.compute_pathway_dmatrix(feature)

    def report_results(self):
        """Summarize and rank pathway scores"""
        mcc_df_dict = defaultdict(list)
        for pathway, mcc_results in self.raw_results:
            mcc_df_dict['pathway'].append(pathway)
            for clf, mcc in mcc_results.items():
                mcc_df_dict[clf].append(mcc)
        mcc_df = pd.DataFrame(mcc_df_dict).set_index('pathway')
        mcc_df['mean'] = mcc_df.mean(axis=1)
        mcc_df.sort_values('mean', ascending=False, inplace=True)
        mcc_df.to_csv(self.expdir / 'classifier-MCC-summary.csv')
        self.full_results = mcc_df
        self.nonzero_results = compute_stats(self.full_results)
        self.nonzero_results['description'] = self.pathway_descriptions
        self.nonzero_results['n_genes'] = {pathway: len(gene_list) for pathway, gene_list in self.pathways_map.items()
                                           if pathway in self.nonzero_results.index}
        self.nonzero_results['genes'] = {pathway: ','.join(gene_list) for pathway, gene_list
                                         in self.pathways_map.items() if pathway in self.nonzero_results.index}
        self.nonzero_results.to_csv(self.expdir / 'meanMCC-results.nonzero-stats.csv')

    def visualize(self):
        """Generate summary figures of EAML results, including scatterplots and histograms of MCC scores"""
        default_fig_params = {'figure.figsize': (8, 6)}
        mcc_scatter(self.full_results, column='mean', fig_params=default_fig_params)\
            .savefig(self.expdir / f'meanMCC-scatter.pdf')
        mcc_hist(self.full_results, column='mean', fig_params=default_fig_params)\
            .savefig(self.expdir / f'meanMCC-hist.pdf')
        mcc_scatter(self.nonzero_results, column='MCC', fig_params=default_fig_params)\
            .savefig(self.expdir / 'meanMCC-scatter.nonzero.pdf')
        mcc_hist(self.nonzero_results, column='MCC', fig_params=default_fig_params)\
            .savefig(self.expdir / 'meanMCC-hist.nonzero.pdf')


# util functions
def load_pathways(pathways_fn, reference_df):
    """
    Loads map of gene lists to pathways and intersects with reference genes

    Args:
        pathways_fn (str/Path): File with two columns of pathways and gene lists
        reference_df (DataFrame): Reference of all coding genes and positions

    Returns:
        dict(list): Mapping of gene lists to pathways
    """
    pathway_map, pathway_descriptions = {}, {}
    with open(pathways_fn) as f:
        for ln in f:
            ln = ln.strip().split('\t')
            pathway = ln[0]
            genes = set(ln[2].split(',')).intersection(reference_df.index)
            pathway_descriptions[pathway] = ln[1]
            pathway_map[pathway] = genes
    return pathway_map, pathway_descriptions
