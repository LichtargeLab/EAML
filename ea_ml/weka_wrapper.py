#!/usr/bin/env python
from pathlib import Path
from subprocess import run, DEVNULL, PIPE


def call_weka(clf, clf_params, arff_fn, weka_path='/opt/weka', cv=10, seed=111):
    """
    Wrapper that calls Weka JVM as subprocess

    Args:
        clf (str): Name of classifier class in Weka
        clf_params (str): String of classifier-specific hyperparameters
        arff_fn (Path-like): Filepath to intermediate ARFF file of gene design matrix
        weka_path (Path-like): Filepath to Weka directory
        cv (int): Number of folds for cross-validation
        seed (int): Random seed

    Returns:
        float: mean MCC score from cross-validation
    """
    weka_jar = Path(weka_path) / 'weka.jar'
    weka_call = f'java -Xmx1g -cp {weka_jar} weka.Run .{clf} {clf_params} -t {arff_fn} -v -o -x {cv} -s {seed}'
    weka_out = run(weka_call, shell=True, stderr=DEVNULL, stdout=PIPE, text=True)
    return parse_weka_output(weka_out.stdout)


def parse_weka_output(stdout):
    """
    Parse string stdout from single classifier's cross-validation

    Args:
        stdout (str): Stdout from Weka subprocess

    Returns:
        float: mean MCC score from cross-validation
    """
    stdout = stdout.split('\n')
    for i, ln in enumerate(stdout):
        ln = [string for string in ln.split(' ') if string]
        if 'MCC' in ln:
            score_row = i + 2  # class 1 is two rows after score header
            # two header names are split twice (TP Rate and FP Rate), so the -2 corrects the position
            score_col = ln.index('MCC') - 2
            score_ln = [string for string in stdout[score_row].split(' ') if string]
            score = float(score_ln[score_col])
            return score


def eval_gene(gene, dmatrix, targets, clf_calls, seed=111, cv=10, expdir=Path('.'), weka_path='/opt/weka'):
    """
    Evaluate gene's classification performance across all Pipeline classifiers

    Args:
        gene (str): HGSC gene symbol
        dmatrix (DataFrame): EA design matrix
        targets (Series): Target classes for all samples
        clf_calls (dict): Mapping of Weka classifier to hyperparameter string
        seed (int): Random seed
        cv (int): Number of folds for cross-validation
        expdir (Path-like): Filepath to experiment directory
        weka_path (Path-like): Filepath to Weka directory

    Returns:
        dict: Mapping of classifier to mean MCC score
    """
    mcc_results = {}
    if cv == -1:
        cv = len(dmatrix)
    arff_fn = expdir / 'tmp' / f'{gene}.arff'
    write_arff(dmatrix, targets, arff_fn)
    for clf, params in clf_calls.items():
        mcc_results[clf] = call_weka(clf, params, arff_fn, weka_path=weka_path, cv=cv, seed=seed)
    return mcc_results


def write_arff(design_matrix, targets, f_out):
    """
    Output a .arff file corresponding to given gene's design matrix

    Args:
        design_matrix (DataFrame): EA design matrix
        targets (Series): Target classes for all samples
        f_out (Path): filepath for the ARFF file
    """
    def _write_rows(examples):
        for example, label in examples:
            example = [str(x) for x in example]
            example.append(str(label))
            f.write(','.join(example) + '\n')

    with open(f_out, 'w') as f:
        relation = f_out.stem
        f.write(f'@relation {relation}\n')
        for ft in design_matrix.columns:
            f.write(f'@attribute {ft} REAL\n')
        f.write('@attribute class {0,1}\n')
        f.write('@data\n')
        rows = zip(design_matrix.sort_index().values, targets.sort_index().values)
        _write_rows(rows)
