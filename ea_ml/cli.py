#!/usr/bin/env python3
"""
Created on 9/20/19

@author: dillonshapiro

Command-line interface for pyEA-ML.
"""
import sys
import argparse
from pathlib import Path

from . import VERSION, DESCRIPTION, CLI
from .pipeline import run_ea_ml
from .visualize import visualize
from .shuffle import run_shuffling


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
    sub.add_argument('data', type=Path, help='VCF file annotated by ANNOVAR and EA, or compressed sparse matrix')
    sub.add_argument('samples', type=Path,
                     help='comma-delimited file with VCF sample IDs and corresponding disease status (0 or 1)')
    sub.add_argument('gene_list', type=Path, help='single-column list of genes to test')
    sub.add_argument('-t', '--threads', type=int, default=1, help='number of threads to run Weka on')
    sub.add_argument('-s', '--seed', type=int, default=111, help='random seed for generating KFold samples')
    sub.add_argument('-k', '--kfolds', type=int, default=10, help='number of folds for cross-validation')

    # Shuffle experiment parser
    info = 'analyze significance of MCC scores through random shuffling'
    sub = subs.add_parser('shuffle', help=info)
    sub.add_argument('experiment_dir', type=Path, help='root directory for experiment')
    sub.add_argument('data', type=Path, help='Path to VCF file')
    sub.add_argument('samples', type=Path, help='Path to samples list containing original corresponding labels')
    sub.add_argument('gene_list', type=Path, help='Path to single-column list of test genes')
    sub.add_argument('predictions', type=Path, help='Path to real experiment results')
    sub.add_argument('-t', '--threads', type=int, default=1, help='number of threads to run Weka on')
    sub.add_argument('-s', '--seed', type=int, default=111, help='random seed for generating KFold samples')
    sub.add_argument('-k', '--kfolds', type=int, default=10, help='number of folds for cross-validation')
    sub.add_argument('-n', '--n_runs', type=int, default=100,
                     help='Number of shuffling runs to include in distribution')

    # Visualize parser
    info = 'visualize results of EA-ML analysis'
    sub = subs.add_parser('visualize', help=info)
    sub.add_argument('experiment_dir', type=Path, help='root directory for experiment')
    sub.add_argument('--dpi', default=150, help='DPI for output figures')
    sub.add_argument('-o', '--output', type=Path, help='location to output figures')
    sub.add_argument('-p', '--prefix', default='maxMCC', help='prefix for outfit files')

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
        args += [namespace.data, namespace.samples, namespace.gene_list]
        kwargs.update(threads=namespace.threads, seed=namespace.seed, kfolds=namespace.kfolds)
    elif namespace.command == 'shuffle':
        function = run_shuffling
        args += [namespace.data, namespace.samples, namespace.gene_list, namespace.predictions]
        kwargs.update(threads=namespace.threads, seed=namespace.seed, kfolds=namespace.kfolds, n_runs=namespace.n_runs)
    elif namespace.command == 'visualize':
        function = visualize
        output = namespace.output
        args.append(output) if output else args.append(namespace.experiment_dir)
        kwargs.update(dpi=namespace.dpi, prefix=namespace.prefix)

    return function, args, kwargs


if __name__ == '__main__':
    main()
