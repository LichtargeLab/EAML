#!/usr/bin/env python
"""Command-line interface for pyEA-ML"""
import sys
import argparse
from pathlib import Path

from . import VERSION, DESCRIPTION, CLI
from .pipeline import Pipeline
from .permute import run_permutations


def main_args(parser):
    """Common arguments for all modules"""
    parser.add_argument('data', type=Path,
                        help='VCF file annotated by ANNOVAR and EA, or CSV with Pandas-multi-indexed columns')
    parser.add_argument('targets',
                        help='comma-delimited file with VCF sample IDs and corresponding disease status (0 or 1)')
    parser.add_argument('-e', '--experiment-dir', default='.', type=Path, help='root directory for experiment')
    parser.add_argument('-r', '--reference', default='hg19', help='genome reference name')
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


# noinspection PyTypeChecker
def main():
    """Process command-line arguments and run the program."""
    parser = argparse.ArgumentParser(prog=CLI, description=DESCRIPTION,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=VERSION)
    subs = parser.add_subparsers(dest='command', metavar="<command>")

    # Pipeline parser
    info = 'run the EA-ML analysis'
    sub = subs.add_parser('run', help=info)
    main_args(sub)
    sub.add_argument('--write-data', action='store_true', help='keep design matrix after analysis')
    sub.add_argument('--dpi', default=150, type=int, help='DPI for output figures')

    # Permutation experiment parser
    info = 'analyze significance of MCC scores through label permutations (experimental)'
    sub = subs.add_parser('permute', help=info)
    main_args(sub)
    sub.add_argument('predictions', help='Path to real experiment results')
    sub.add_argument('-n', '--n_runs', type=int, default=100,
                     help='Number of permutations to include in distribution')
    sub.add_argument('--restart', type=int, default=0, help='run to restart permutations at')
    sub.add_argument('-c', '--clean', action='store_true', help='clean design matrix and permutation files')

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

    if kwargs.pop('command') == 'run':
        args = [kwargs.pop(arg) for arg in ('experiment_dir', 'data', 'targets')]
        pipeline = Pipeline(*args, **kwargs)
        pipeline.run()
    elif kwargs.pop('command') == 'permute':
        args = [kwargs.pop(arg) for arg in ('experiment_dir', 'data', 'targets', 'predictions')]
        run_permutations(*args, **kwargs)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
