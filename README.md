# EA-ML

This pipeline uses an ensemble of supervised learning algorithms to score a gene's contribution to disease risk based on
its case/control separation ability. This is all based on a probability score (pEA) computed using the Evolutionary
Action variant impact scoring method.

The output is a ranked list of genes scored by the Matthew's Correlation Coefficient (MCC), effectively using
classification accuracy as a proxy for a gene's relevance in a specified disease context.

## System Requirements

### Hardware Requirements

EA-ML is runnable on a standard desktop computer with 16GB+ of RAM, however certain model training operations can be
prohibitively slow without the use of multiple CPU cores, especially when operating on large datasets.
Thus, it's highly recommended to use a computing cluster with many CPU cores.

### Software Requirements

#### OS Requirements

This package is supported for macOS and Linux. It has been tested on the following systems:
- macOS: Sierra (10.12) or greater
- Ubuntu: 14.04, 18.04, 20.04
- Red Hat: 7.3

#### Dependencies

- [Anaconda3/Miniconda3](https://docs.anaconda.com/anaconda/install/index.html)
- [Weka 3.8.0+](https://waikato.github.io/weka-wiki/downloading_weka/)
    - Weka must be downloaded separately
- Python 3.7+
- OpenJDK 8.0.0+
- Python package dependencies listed in `environment.yml`

## Installation

To clone the pipeline and install the conda environment:
```bash
git clone https://github.com/LichtargeLab/pyEA-ML.git
conda env create -f ./pyEA-ML/environment.yml
pip install -e ./pyEA-ML/
```

EA-ML can also be installed without editable mode, but any changes pulled later won't be automatically included.

## Usage

It is highly recommended to run any EA-ML analysis inside a virtual environment.

See available commands:
```bash
ea-ml --help
```

EA-ML can be run by calling `ea-ml` and one of its commands:

| command     | description                                                                           |
|-------------|---------------------------------------------------------------------------------------|
| run         | run the EA-ML analysis                                                                |
| downsample  | run power analysis by repeatedly sampling cohort and calculating overlap significance |

### Main Pipeline

Required arguments:

| argument        | type          | description                                              |
|-----------------|---------------|----------------------------------------------------------|
| data            | \<file\>      | VCF or directory of precomputed design matrices          |
| targets         | \<file\>      | two-column CSV with sample IDs and disease status        |
| -w, --weka-path | \<str\>       | location of Weka installation                            |

Optional arguments:

| argument             | type      | description                                                                         |
|----------------------|-----------|-------------------------------------------------------------------------------------|
| -e, --experiment_dir | \<str\>   | experiment directory                                                                |
| -r, --reference      | \<str\>   | genome reference version or file (hg19, hg38, GRCh37, GRCh38)                       |
| -a, --annotation     | \<str\>   | Variant annotation pipeline used (ANNOVAR, VEP)                                     |
| --parse-EA           | \<str\>   | how to parse EA scores from different transcripts (max, mean, all, canonical)       |
| --min-af             | \<float\> | sets minimum allele frequency threshold                                             |
| --max-af             | \<float\> | sets maximum allele frequency threshold                                             |
| --af-field           | \<str\>   | field with AF information                                                           |
| -X, --include-X      |           | includes X chromosome in analysis                                                   |
| -k, --kfolds         | \<int\>   | number of cross-validation folds                                                    |
| -s, --seed           | \<int\>   | random seed for cross-validation                                                    |
| --cpus               | \<int\>   | number of CPUs to use                                                               |
| --write-data         |           | keeps design matrix after analysis completes                                        |
| --dpi                | \<int\>   | DPI for output figures                                                              |
| --memory             | \<str\>   | memory argument for Weka JVM                                                        |


*Note: To specify leave-one-out cross-validation, set the number of folds equal to -1*

#### Example Usage:

Example files are in the example_data directory and can be used to test that the pipeline works. Runtime should only be
a few seconds and output all files described in 'Output' section.
```bash
ea-ml run example.vcf.gz example.samples.csv --experiment_dir ./ --reference example.reference.txt --anotation VEP --weka-path ~/weka/ --seed 1 --cpus 1
```

### Downsampling Analysis

This is a crude power analysis that samples the given cohort at various sample sizes, then calculates the significance
of the average overlap between predictions from the sample and predictions from the whole cohort.

Required arguments:

| argument        | type          | description                                              |
|-----------------|---------------|----------------------------------------------------------|
| data            | \<file\>      | VCF or directory of precomputed design matrices          |
| targets         | \<file\>      | two-column CSV with sample IDs and disease status        |
| true_results    | \<file\>      | final results CSV from full cohort experiment            |
| sample_sizes    | \<int\>       | sample sizes to test                                     |
| -w, --weka-path | \<str\>       | location of Weka installation                            |

Optional arguments:

| argument             | type      | description                                                                         |
|----------------------|-----------|-------------------------------------------------------------------------------------|
| -e, --experiment_dir | \<str\>   | experiment directory                                                                |
| -r, --reference      | \<str\>   | genome reference version or file (hg19, hg38, GRCh37, GRCh38)                       |
| -a, --annotation     | \<str\>   | Variant annotation pipeline used (ANNOVAR, VEP)                                     |
| --parse-EA           | \<str\>   | how to parse EA scores from different transcripts (max, mean, all, canonical)       |
| --min-af             | \<float\> | sets minimum allele frequency threshold                                             |
| --max-af             | \<float\> | sets maximum allele frequency threshold                                             |
| --af-field           | \<str\>   | field with AF information                                                           |
| -X, --include-X      |           | includes X chromosome in analysis                                                   |
| -k, --kfolds         | \<int\>   | number of cross-validation folds                                                    |
| -s, --seed           | \<int\>   | random seed for cross-validation                                                    |
| --cpus               | \<int\>   | number of CPUs to use                                                               |
| --write-data         |           | keeps design matrix after analysis completes                                        |
| --dpi                | \<int\>   | DPI for output figures                                                              |
| --memory             | \<str\>   | memory argument for Weka JVM                                                        |
| --nrepeats           | \<int\>   | number of replicates for each sample size                                           |

#### Example Usage:
```bash
ea-ml downsample VCF.gz SamplePhenotypes.csv ./meanMCC-results.csv 250 500 1000 --experiment_dir ./ --reference GRCh38 \
--anotation VEP --weka-path ~/weka/ --seed 1 --cpus 10 --nrepeats 1000
```

## Input Requirements

In order to use this pipeline properly, it requires 3 input files:

1. A custom-annotated VCF using either VEP (ENSEMBL transcripts) or ANNOVAR (RefSeq transcripts)
2. A comma-delimited list of samples along with their disease status (0 or 1)

*Note: Current package reference files are derived from those used in the 2013-05-09 version of ANNOVAR and
version 94 of ENSEMBL-VEP*

The VCF file should follow proper formatting as described [here](<https://samtools.github.io/hts-specs/VCFv4.2.pdf>).

### VCF INFO fields required (with Type and Number requirements for header):

#### With ANNOVAR annotation:
  - 'gene', 'NM' and 'EA' annotations should be annotated as fields in the INFO column
  - 'EA' field must be typed as a 'String' (Type=String); this is because of the way our custom annotation labels
     variants by transcript, with nonsense, frameshift-indels and STOP loss variants defined as strings in the EA field
  - 'gene' (Number=1, Type=String): Annotated gene symbol
  - 'NM' (Number=., Type=String): List of RefSeq transcript IDs
  - 'EA' (Number=., Type=String): List of EA scores corresponding to each transcript


  - Since the custom ANNOVAR annotation does not include a Consequence field, it is inferred from the EA field.
    As a result, other values can appear:
    - 'fs-indel'
    - 'STOP'
    - 'STOP_loss'


#### With VEP annotation:
  - 'Ensembl_proteinid' (Number=., Type=String): List of ENSEMBL transcript IDs
    - Can be annotated by dbNSFP or VEP
  - 'Consequence' (Number=1, Type=String): Variant consequence, as annotated by VEP
  - 'SYMBOL' (Number=1, Type=String): Gene symbol, as annotated by VEP
  - 'ENSP' (Number=1, Type=String): Canonical transcript ID, as annotated by VEP
  - 'EA' (Number=., Type=Float): List of EA scores corresponding to transcripts from 'Ensembl_proteinid' field

#### Common fields:

- If an allele frequency threshold is set, the VCF requires an additional specified field (see command line options for more detail)
  - Allele frequency can be annotated using another tool like `bcftools`
  
## Output

All files are generated in the directory given by the `--experiment_dir` option:
  - classifier-MCC-summary.csv: An unsorted table of the average MCCs across classifiers for each gene
  - meanMCC-results.nonzero-stats.csv: The final ranked set of genes, with average MCC score, p-value,
    and adjusted p-value
  - A series of plots visualizing results:
    - Scatterplots showing gene rank vs. MCC, using either all genes or only those with classification power
    - Histograms showing the distribution of MCC scores, using either all genes or only those with classification power
    - A Manhattan plot showing the positions of all genes tested and which passed the q-value < 0.1 threshold

## Troubleshooting

- If the program is unable to find the Java installation, set the `JAVA_HOME` environmental variable to wherever Java is
  installed. If this is in the conda environment, use this command in Terminal:
```bash
export JAVA_HOME=${CONDA_PREFIX}
```

## Credits

Tools used in rendering this package:

-  [cookiecutter](https://github.com/audreyr/cookiecutter)
-  [cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
