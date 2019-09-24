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
from .main import run_ea_ml
from .visualize import visualize


def main(args=None, function=None):
    """Process command-line arguments and run the program."""
    parser = argparse.ArgumentParser(prog=CLI, description=DESCRIPTION)
    parser.add_argument('experiment_dir', type=Path, help='root directory for experiment')
    parser.add_argument('-v', '--version', action='version', version=VERSION)
    subs = parser.add_subparsers(dest='command', metavar="<command>")

    # Pipeline parser
    info = 'run the EA-ML analysis'
    sub = subs.add_parser('run', help=info)

    sub.add_argument('data', type=Path, help='VCF file annotated by ANNOVAR and EA, or compressed sparse matrix')
    sub.add_argument('samples', type=Path,
                        help='comma-delimited file with VCF sample IDs and corresponding disease status (0 or 1)')
    sub.add_argument('gene_list', type=Path, help='single-column list of genes to test')
    sub.add_argument('-t', '--threads', type=int, default=1, help='number of threads to run Weka on')
    sub.add_argument('-r', '--seed', type=int, default=111, help='random seed for generating KFold samples')
    sub.add_argument('-k', '--kfolds', type=int, default=10, help='number of folds for cross-validation')

    # TODO: move shuffle code
    # Shuffle experiment parser
    #info = 'analyze significance of MCC scores through random shuffling'
    #sub = subs.add_parser('shuffle')

    # Visualize parser
    info = 'visualize results of EA-ML analysis'
    sub = subs.add_parser('visualize', help=info)
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
        pass
    elif namespace.command == 'visualize':
        function = visualize
        output = namespace.output
        args.append(output) if output else args.append(namespace.experiment_dir)
        kwargs.update(dpi=namespace.dpi, prefix=namespace.prefix)

    return function, args, kwargs


if __name__ == '__main__':
    main()
