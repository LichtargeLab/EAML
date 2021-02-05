#!/usr/bin/env python
from pathlib import Path
from subprocess import run, STDOUT, DEVNULL


def call_weka(clf, clf_params, arff_fn, weka_path='/opt/weka', cv=10, seed=111):
    weka_jar = f'{weka_path}/weka.jar'
    weka_call = f'java -Xmx1g -cp {weka_jar} weka.Run .{clf} {clf_params} -t {arff_fn} -v -o -x {cv} -s {seed}'
    weka_out = run(weka_call, shell=True, stdout=STDOUT, stderr=DEVNULL).stdout
    return parse_weka_output(weka_out)


def parse_weka_output(stdout):
    stdout = stdout.split('\n')
    for i, ln in enumerate(stdout):
        ln = [string for string in ln.split(' ') if string]
        if 'MCC' in ln:
            score_row = i + 2
            score_col = ln.index('MCC')
            score_ln = [string for string in stdout[score_row].split(' ') if string]
            score = float(score_ln[score_col])
            return score


def eval_gene(gene, dmatrix, targets, clf_calls, seed=111, cv=10, expdir=Path('.'), weka_path='/opt/weka'):
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
        Outputs an .arff file corresponding to the DataFrame

        Args:
            design_matrix (DataFrame): design matrix for a single gene
            targets (Series): disease labels
            f_out (Path): filepath for the output
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
