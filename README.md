# pyEA-ML

The Python version of the lab EA-ML pipeline.

This pipeline uses an ensemble of supervised learning algorithms to score a gene's contribution to disease risk based on
it's case/control separation ability. This is all based on a probability score (pEA) computed using EA.

The output is a ranked list of genes scored by the Matthew's Correlation Coefficient (MCC), effectively using
classification accuracy as a proxy for a gene's relevance in a specified disease context.

## Setup

### Requirements

- OS X or Linux
- Anaconda3
- Python 3.7+
- Java JDK 1.8.0_92 or greater

### Installation

To install the conda environment:
```bash
chmod +x pyEA-ML/install_env.sh
bash pyEA-ML/install_env.sh
```

## Usage

It's highly recommended to run any pyEA-ML analysis inside a virtual environment, this is created from the install script.

See available commands:
```bash
ea-ml --help
```

Before running the main pipeline, be sure that the `JAVA_HOME` variable is set:
```bash
conda activate pyEA-ML
export JAVA_HOME=${CONDA_PREFIX}
```

EA-ML can then be run by calling `ea-ml` and one of its commands:

| command     | description                                               |
|-------------|-----------------------------------------------------------|
| run         | run the EA-ML analysis                                    |
| visualize   | visualize results of EA-ML analysis                       |
| permute     | permute sample labels for evaluating confidence intervals |

### Main Pipeline

Required arguments (note: don't include the argument name in the command line):

| argument       | type          | description                                       |
|----------------|---------------|---------------------------------------------------|
| experiment_dir | \<directory\> | experiment directory                              |
| data           | \<file\>      | VCF or .npz matrix                                |
| samples        | \<file\>      | two-column CSV with sample IDs and disease status |
| gene_list      | \<file\>      | single-column list of genes                       |

Optional arguments:

| argument       | type      | description                                       |
|----------------|-----------|---------------------------------------------------|
| -t, --threads  | \<int\>   | experiment directory                              |
| -s, --seed     | \<int\>   | VCF or .npz matrix                                |
| -k, --kfolds   | \<int\>   | number of cross-validation folds                  |
| --keep-matrix  | \<bool\>  | keeps design matrix after analysis completes      |

*Note: To specify leave-one-out cross-validation, set the number of folds equal to -1*

### Permutation Analysis

Required arguments:

| argument       | type          | description                                       |
|----------------|---------------|---------------------------------------------------|
| experiment_dir | \<directory\> | experiment directory                              |
| data           | \<file\>      | VCF or .npz matrix                                |
| samples        | \<file\>      | two-column CSV with sample IDs and disease status |
| gene_list      | \<file\>      | single-column list of genes                       |
| predictions    | \<file\>      | EA-ML results                                     |

Optional arguments:

| argument       | type      | description                                       |
|----------------|-----------|---------------------------------------------------|
| -t, --threads  | \<int\>   | experiment directory                              |
| -s, --seed     | \<int\>   | VCF or .npz matrix                                |
| -k, --kfolds   | \<int\>   | number of cross-validation folds                  |
| -n, --n_runs   | \<int\>   | number of permutations to include in distribution |
| -r, --restart  | \<int\>   | restart permutations from this number             |
| -c, --clean    | \<bool\>  | clean design matrix and permutation files         |

### Visualize experimental results

Required arguments:

| argument       | type          | description          |
|----------------|---------------|----------------------|
| experiment_dir | \<directory\> | experiment directory |

Optional arguments:

| argument       | type          | description                    |
|----------------|---------------|--------------------------------|
| --dpi          | \<int\>       | DPI for output figures         |
| -o, --output   | \<directory\> | location to output figures     |
| -p, --prefix   | \<str\>       | prefix for output file names   |

## Input Requirements

In order to use this pipeline properly, it requires 3 input files:

1. A VCF file containing all cohort variants annotated with gene information, EA scores, and genotype information for
   each cohort sample.
2. A comma-delimited list of samples along with their disease status (0 or 1).
3. A single column list of genes to test.

The VCF file should follow proper formatting as described [here](<https://samtools.github.io/hts-specs/VCFv4.2.pdf>).

Additionally, some extra information is required:

- 'gene' and 'EA' annotations as fields in the INFO column
- Different fields in INFO and FORMAT columns should be defined in the header, with type information
- 'EA' attribute must be typed as a 'String' (Type=String), this is because of the way EA labels variants by transcript,
  and other variant annotation softwares do this on a variant-wise basis instead

## Credits

Tools used in rendering this package:

-  [cookiecutter](https://github.com/audreyr/cookiecutter)
-  [cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
