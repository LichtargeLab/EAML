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
- OpenJDK 8.0.265+
- [Weka 3.8.0+](https://waikato.github.io/weka-wiki/downloading_weka/)
    - Weka must be downloaded separately

### Installation

To clone the pipeline and install the conda environment:
```bash
git clone https://github.com/LichtargeLab/pyEA-ML.git
conda env create -f ./pyEA-ML/environment.yml
pip install -e ./pyEA-ML/
```

EA-ML can also be installed without editable mode, but any changes pulled later won't be automatically included.

## Usage

It's highly recommended to run any pyEA-ML analysis inside a virtual environment, this is created from the install script.

See available commands:
```bash
ea-ml --help
```

EA-ML can be run by calling `ea-ml` and one of its commands:

| command     | description                                               |
|-------------|-----------------------------------------------------------|
| run         | run the EA-ML analysis                                    |
| permute     | permute sample labels for evaluating confidence intervals |

### Main Pipeline

Required arguments (note: don't include the argument name in the command line):

| argument       | type          | description                                              |
|----------------|---------------|----------------------------------------------------------|
| data           | \<file\>      | VCF or HDF file of DataFrames with gene names as keys    |
| targets        | \<file\>      | two-column CSV with sample IDs and disease status        |

Optional arguments:

| argument             | type      | description                                                                         |
|----------------------|-----------|-------------------------------------------------------------------------------------|
| -e, --experiment_dir | \<str\>   | experiment directory                                                                |
| -r, --reference      | \<str\>   | genome reference (hg19, hg38)                                                       |
| --parse-EA           | \<str\>   | how to parse EA scores from different transcripts (max, mean, all, canonical)  |
| --min-af             | \<float\> | sets minimum allele frequency threshold                                             |
| --max-af             | \<float\> | sets maximum allele frequency threshold                                             |
| --af-field           | \<str\>   | field with AF information                                                           |
| -X, --include-X      | \<bool\>  | includes X chromosome in analysis                                                   |
| -k, --kfolds         | \<int\>   | number of cross-validation folds                                                    |
| -s, --seed           | \<int\>   | random seed for cross-validation                                                    |
| --cpus               | \<int\>   | number of CPUs to use                                                               |
| --write-data         | \<bool\>  | keeps design matrix after analysis completes                                        |
| --dpi                | \<int\>   | DPI for output figures                                                              |
| -w, --weka-path      | \<str\>   | location of Weka installation                                                       |


*Note: To specify leave-one-out cross-validation, set the number of folds equal to -1*

### Permutation Analysis (experimental)

Required arguments:

| argument       | type          | description                                              |
|----------------|---------------|----------------------------------------------------------|
| data           | \<file\>      | VCF or HDF file of DataFrames with gene names as keys    |
| targets        | \<file\>      | two-column CSV with sample IDs and disease status        |
| predictions    | \<file\>      | EA-ML results                                            |

Optional arguments:

| argument             | type      | description                                                                         |
|----------------------|-----------|-------------------------------------------------------------------------------------|
| -e, --experiment_dir | \<str\>   | experiment directory                                                                |
| -r, --reference      | \<str\>   | genome reference (hg19, hg38)                                                       |
| --parse-EA           | \<str\>   | how to parse EA scores from different transcripts (max, mean, all, canonical)  |
| --min-af             | \<float\> | sets minimum allele frequency threshold                                             |
| --max-af             | \<float\> | sets maximum allele frequency threshold                                             |
| --af-field           | \<str\>   | field with AF information                                                           |
| -n, --n_runs         | \<int\>   | number of permutations to include in distribution                                   |
| -X, --include-X      | \<bool\>  | includes X chromosome in analysis                                                   |
| -k, --kfolds         | \<int\>   | number of cross-validation folds                                                    |
| -s, --seed           | \<int\>   | random seed for cross-validation                                                    |
| --cpus               | \<int\>   | number of CPUs to use                                                               |
| --restart            | \<int\>   | restart permutations from this number                                               |
| -c, --clean          | \<bool\>  | clean design matrix and permutation files                                           |
| -w, --weka-path      | \<str\>   | location of Weka installation                                                       |


## Input Requirements

In order to use this pipeline properly, it requires 3 input files:

1. A VCF file containing all cohort variants annotated with gene information (from ANNOVAR), EA scores, and genotype
   information for each cohort sample.
2. A comma-delimited list of samples along with their disease status (0 or 1).
3. A RefGene-formatted reference file of genes (hg19 and hg38 are included in package)

The VCF file should follow proper formatting as described [here](<https://samtools.github.io/hts-specs/VCFv4.2.pdf>).

Additionally, some extra information is required:

- 'gene' and 'EA' annotations as fields in the INFO column
- Different fields in INFO and FORMAT columns should be defined in the header, with type information
- 'EA' attribute must be typed as a 'String' (Type=String), this is because of the way EA labels variants by transcript,
  with nonsense, frameshift-indels and STOP loss variants defined as strings in the EA field

## Troubleshooting

- If the program is unable to find the Java installation, set the `JAVA_HOME` environmental variable to wherever Java is
  installed. If this is in the conda environment, use this:
```bash
export JAVA_HOME=${CONDA_PREFIX}
```

## Credits

Tools used in rendering this package:

-  [cookiecutter](https://github.com/audreyr/cookiecutter)
-  [cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
