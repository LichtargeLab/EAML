#!/usr/bin/env python
"""Command-line interface for pyEA-ML"""
import sys
import argparse
from pathlib import Path

from . import VERSION, DESCRIPTION, CLI
from .pipeline import run_ea_ml
from .visualize import visualize
from .permute import run_permutations


def main_args(parser):
    parser.add_argument('data_fn', type=Path,
                        help='VCF file annotated by ANNOVAR and EA, or CSV with Pandas-multi-indexed columns')
    parser.add_argument('samples',
                        help='comma-delimited file with VCF sample IDs and corresponding disease status (0 or 1)')
    parser.add_argument('-e', '--experiment-dir', default='.', type=Path, help='root directory for experiment')
    parser.add_argument('-r', '--reference', default='hg19', choices=('hg19', 'hg38'), help='genome reference name')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating KFold samples')
    parser.add_argument('-k', '--kfolds', type=int, default=10, help='number of folds for cross-validation')
    parser.add_argument('-X', '--include-X', action='store_true', help='includes X chromosome in analysis')
    parser.add_argument('--max-af', type=float, default=1, help='maximum allele frequency cutoff')
    parser.add_argument('--min-af', type=float, default=0, help='minimum allele frequency cutoff')
    parser.add_argument('--af-field', help='name of INFO field with AF values', default='AF')
    parser.add_argument('--cpus', type=int, default=1, help='number of CPUs to use for multiprocessing')


# noinspection PyTypeChecker
def main(args=None, function=None):
    """Process command-line arguments and run the program."""
    parser = argparse.ArgumentParser(prog=CLI, description=DESCRIPTION)
    parser.add_argument('-v', '--version', action='version', version=VERSION)
    subs = parser.add_subparsers(dest='command', metavar="<command>")

    # Pipeline parser
    info = 'run the EA-ML analysis'
    sub = subs.add_parser('run', help=info)
    main_args(sub)
    sub.add_argument('--keep-matrix', action='store_true', help='keep design matrix after analysis')

    # Permutation experiment parser
    info = 'analyze significance of MCC scores through label permutations'
    sub = subs.add_parser('permute', help=info)
    main_args(sub)
    sub.add_argument('predictions', help='Path to real experiment results')
    sub.add_argument('-n', '--n_runs', type=int, default=100,
                     help='Number of permutations to include in distribution')
    sub.add_argument('--restart', type=int, default=0, help='run to restart permutations at')
    sub.add_argument('-c', '--clean', action='store_true', help='clean design matrix and permutation files')

    # Visualize parser
    info = 'visualize results of EA-ML analysis'
    sub = subs.add_parser('visualize', help=info)
    sub.add_argument('-e', '--experiment-dir', default='.', type=Path, help='root directory for experiment')
    sub.add_argument('-r', '--reference', default='hg19', choices=('hg19', 'hg38'), help='genome reference name')
    sub.add_argument('--dpi', default=150, type=int, help='DPI for output figures')
    sub.add_argument('-o', '--output', type=Path, help='location to output figures')
    sub.add_argument('-p', '--prefix', default='', help='prefix for output files')

    # Parse arguments
    namespace = parser.parse_args(args=args)

    # Run the program
    function, args, kwargs = _get_command(function, namespace)
    if function is None:
        parser.print_help()
        sys.exit(1)
    function(*args, **kwargs)


def _get_command(function, namespace):
    kwargs = vars(namespace)
    args = None

    if kwargs.pop('command') == 'run':
        function = run_ea_ml
        args = (kwargs.pop(arg) for arg in ('experiment_dir', 'data_fn', 'samples'))
    elif kwargs.pop('command') == 'permute':
        function = run_permutations
        args = (kwargs.pop(arg) for arg in ('experiment_dir', 'data_fn', 'samples', 'predictions'))
    elif kwargs.pop('command') == 'visualize':
        function = visualize
        args = (kwargs.pop(arg) for arg in ('experiment_dir', 'output'))

    return function, args, kwargs


if __name__ == '__main__':
    main()
