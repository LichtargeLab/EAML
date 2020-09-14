#!/usr/bin/env python
"""Command-line interface for pyEA-ML"""
import sys
import argparse
from pathlib import Path

from . import VERSION, DESCRIPTION, CLI
from .pipeline import run_ea_ml
from .visualize import visualize
from .permute import run_permutations


# noinspection PyTypeChecker
def main(args=None, function=None):
    """Process command-line arguments and run the program."""
    parser = argparse.ArgumentParser(prog=CLI, description=DESCRIPTION)
    parser.add_argument('-v', '--version', action='version', version=VERSION)
    subs = parser.add_subparsers(dest='command', metavar="<command>")

    # Pipeline parser
    info = 'run the EA-ML analysis'
    sub = subs.add_parser('run', help=info)
    sub.add_argument('experiment_dir', type=Path, help='root directory for experiment')
    sub.add_argument('data', type=Path,
                     help='VCF file annotated by ANNOVAR and EA, or CSV with Pandas-multi-indexed columns')
    sub.add_argument('samples', help='comma-delimited file with VCF sample IDs and corresponding disease status (0 or 1)')
    sub.add_argument('-r', '--reference', default='hg19', choices=('hg19', 'hg38'), help='genome reference name')
    sub.add_argument('-t', '--threads', type=int, default=1, help='number of parallel threads')
    sub.add_argument('-s', '--seed', type=int, default=111, help='random seed for generating KFold samples')
    sub.add_argument('-k', '--kfolds', type=int, default=10, help='number of folds for cross-validation')
    sub.add_argument('--keep-matrix', action='store_true', help='keep design matrix after analysis')

    # Permutation experiment parser
    info = 'analyze significance of MCC scores through label permutations'
    sub = subs.add_parser('permute', help=info)
    sub.add_argument('experiment_dir', type=Path, help='root directory for experiment')
    sub.add_argument('data', type=Path, help='Path to VCF file')
    sub.add_argument('samples', help='Path to samples list containing original corresponding labels')
    sub.add_argument('predictions', help='Path to real experiment results')
    sub.add_argument('-r', '--reference', default='hg19', choices=('hg19', 'hg38'), help='genome reference name')
    sub.add_argument('-t', '--threads', type=int, default=1, help='number of parallel threads')
    sub.add_argument('-s', '--seed', type=int, default=111, help='random seed for generating KFold samples')
    sub.add_argument('-k', '--kfolds', type=int, default=10, help='number of folds for cross-validation')
    sub.add_argument('-n', '--n_runs', type=int, default=100,
                     help='Number of permutations to include in distribution')
    sub.add_argument('--restart', type=int, default=0, help='run to restart permutations at')
    sub.add_argument('-c', '--clean', action='store_true', help='clean design matrix and permutation files')

    # Visualize parser
    info = 'visualize results of EA-ML analysis'
    sub = subs.add_parser('visualize', help=info)
    sub.add_argument('experiment_dir', type=Path, help='root directory for experiment')
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
    args = [namespace.experiment_dir]
    kwargs = {}

    if namespace.command == 'run':
        function = run_ea_ml
        args += [namespace.data, namespace.samples]
        kwargs.update(reference=namespace.reference, n_jobs=namespace.threads, seed=namespace.seed,
                      kfolds=namespace.kfolds, keep_matrix=namespace.keep_matrix)
    elif namespace.command == 'permute':
        function = run_permutations
        args += [namespace.data, namespace.samples, namespace.predictions]
        kwargs.update(reference=namespace.reference, n_jobs=namespace.threads, seed=namespace.seed,
                      kfolds=namespace.kfolds, n_runs=namespace.n_runs, restart=namespace.restart,
                      clean=namespace.clean)
    elif namespace.command == 'visualize':
        function = visualize
        output = namespace.output
        args.append(output) if output else args.append(namespace.experiment_dir)
        kwargs.update(dpi=namespace.dpi, prefix=namespace.prefix)

    return function, args, kwargs


if __name__ == '__main__':
    main()
