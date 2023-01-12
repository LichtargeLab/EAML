#!/usr/bin/env python
"""Command-line interface for EAML"""
import sys
import argparse
from pathlib import Path

from . import CLI, VERSION, DESCRIPTION
from .pipeline import Pipeline
from .downsampling import DownsamplingPipeline
from .pathways import PathwayPipeline


def main_args(parser):
    """Common arguments for all modules"""
    parser.add_argument('data', type=Path,
                        help='VCF file annotated by ANNOVAR or VEP and EA, or CSV with Pandas-multi-indexed columns')
    parser.add_argument('targets',
                        help='comma-delimited file with VCF sample IDs and corresponding disease status (0 or 1)')
    parser.add_argument('-e', '--experiment-dir', default='.', type=Path, help='root directory for experiment')
    parser.add_argument('-r', '--reference', default='hg19', help='genome reference name')
    parser.add_argument('-a', '--annotation', default='ANNOVAR', choices=('ANNOVAR', 'VEP'),
                        help='Variant annotation pipeline used. Must be either ANNOVAR or VEP')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating KFold samples')
    parser.add_argument('-k', '--kfolds', type=int, default=10, help='number of folds for cross-validation')
    parser.add_argument('-X', '--include-X', action='store_true', help='includes X chromosome in analysis')
    parser.add_argument('--parse-EA', default='canonical', choices=('all', 'max', 'mean', 'canonical'),
                        help='how to parse EA scores from different transcripts')
    parser.add_argument('--max-af', type=float, default=1, help='maximum allele frequency cutoff')
    parser.add_argument('--min-af', type=float, default=0, help='minimum allele frequency cutoff')
    parser.add_argument('--af-field', help='name of INFO field with AF values', default='AF')
    parser.add_argument('--cpus', type=int, default=1, help='number of CPUs to use for multiprocessing')
    parser.add_argument('-w', '--weka-path', default='~/weka', help='path to Weka install directory')
    parser.add_argument('--memory', default='Xmx2g', help='memory argument for each Weka JVM')


# noinspection PyTypeChecker
def main():
    """Process command-line arguments and run the program."""
    parser = argparse.ArgumentParser(prog=CLI, description=DESCRIPTION,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=VERSION)
    subs = parser.add_subparsers(dest='command')

    # Pipeline parser
    info = 'run the EAML analysis on genes'
    sub = subs.add_parser('run', help=info)
    main_args(sub)
    sub.add_argument('--write-data', action='store_true', help='keep design matrix after analysis')

    # Downsampling experiment parser
    info = 'evaluate statistical power by repeat stratified downsampling of cohort'
    sub = subs.add_parser('downsample', help=info)
    main_args(sub)
    sub.add_argument('true_results', type=Path, help='True MCC-ranked results from standard EAML experiment')
    sub.add_argument('--sample-sizes', type=int, nargs='+', help='sample sizes to test')
    sub.add_argument('--nrepeats', type=int, default=10, help='number of replicates per sample size')

    # Pathway pipeline parser
    info = 'run EAML analysis on defined pathways/communities'
    sub = subs.add_parser('pathways', help=info)
    main_args(sub)
    sub.add_argument('pathways_file', type=Path, help='Tab-separated file with pathways/communities and corresponding'
                                                      'comma-separated lists of genes')
    sub.add_argument('--write-data', action='store_true', help='keep design matrix after analysis')

    # Parse arguments
    namespace = parser.parse_args()
    # Run the program
    run_program(parser, namespace)


def run_program(parser, namespace):
    """
    Call specified module of pipeline

    Args:
        parser (ArgumentParser): The command-line parser
        namespace (Namespace): The parsed argument namespace
"""
    kwargs = vars(namespace)
    command = kwargs.pop('command')
    if command == 'run':
        args = [kwargs.pop(arg) for arg in ('experiment_dir', 'data', 'targets')]
        pipeline = Pipeline(*args, **kwargs)
        pipeline.run()
    elif command == 'downsample':
        args = [kwargs.pop(arg) for arg in ('experiment_dir', 'data', 'targets', 'true_results', 'sample_sizes')]
        pipeline = DownsamplingPipeline(*args, **kwargs)
        pipeline.run()
    elif command == 'pathways':
        args = [kwargs.pop(arg) for arg in ('experiment_dir', 'data', 'targets', 'pathways_file')]
        pipeline = PathwayPipeline(*args, **kwargs)
        pipeline.run()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
