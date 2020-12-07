# 0.10.1 (2020-12-07)

- Allele frequency is now computed manually using AC and AN fields if AF field isn't present

# 0.10.0 (2020-12-03)

- Added an option to use an allele frequency threshold for variants. This requires the input VCF to contain an 'AF'
  info field annotated by bcftools

# 0.9.0 (2020-11-04)

- In order to standardize the results presented from any given experiment (and reduce confusion), we are removing the
  automatic summarizing of max scores for each gene. This lines up with results from AD work, where genes were ranked
  by mean MCC, and makes more sense when considering ensembling principles.
- If you want to still analyze the max MCC scores, you can do it manually using the API.

# 0.8.2 (2020-10-07)

- Small fix to Manhattan plotting

# 0.8.1 (2020-09-26)

- Added Manhattan plot to standard visualizations

# 0.8.0 (2020-09-21)

- Replaced DesignMatrix framework with multi-indexed Pandas DataFrame for containing all genes' features
- Replaced test gene list requirement with standardized reference lists from RefGene
- Added multiprocessing (using joblib) to VCF parsing
- Updated package requirements
- Switched from Oracle JDK to OpenJDK for Java requirement

# 0.7.4 (2020-08-03)

- Added visualizations for meanMCC rankings

# 0.7.3 (2020-06-21)

- Fixed bug in correctly calculating stats for meanMCC results
- Added meanMCC calculations to permutation analysis
- Visualization options added to README

# 0.7.2 (2020-06-19)

- Fixed bug where unsorted input genes weren't mapped correctly to results

# 0.7.1 (2020-06-19)

- Added summary results files for mean MCC scores

# 0.7.0 (2020-05-12)

- Added visualizations for non-zero MCC distribution and log-transformed MCCs
- `bumpversion` now used to control version labeling

# 0.6.5 (2020-04-09)

- Added option to restart permutations from specified number

# 0.6.4 (2020-04-05)

- Renamed `shuffle` command to `permute`
- Added `permute` arguments to README
- Fixed bug in permutation call that referenced top experiment directory instead of run-specific directory

# 0.6.3 (2019-11-14)

- Leave-One-Out is now specified by a `-1` argument to `--kfolds`
- Code also mirrors how KFold worker arguments are generated

# 0.6.2 (2019-10-20)

- Moved writing of intermediate ARFF files to worker processes
- Intermediate ARFF files are now cleaned after fold is evaluated

# 0.6.1 (2019-09-26)

- Partially vectorized pEA calculations on each variant
- All EA/zygosity filtering now performed by numpy masks
- Renamed 'hypotheses' to 'features' to avoid naming confusion

# 0.6.0 (2019-09-26)

- Converted pipeline to pip-installable Python package
- Moved all argument parsing to CLI module under `run`, `visualize`, and `shuffle` commands
- Expanded package to allow for summary visualization and label-shuffling experiment through CLI
- Added conda environment installation script

# 0.5.0 (2019-09-13)

- Added subsetting of samples through pysam before VCF parsing
- Moved Java installation and JAVA_HOME specification before installation of other requirements
- Switched to pathlib for file path parsing
- Added option to change number of folds for cross-validation
- Added option to load precomputed matrix from .npz file

# 0.4.0 (2019-07-18)

- Added intermediate worker files to track Weka results
- Switched to argument parsing through argparse instead of loading from python-dotenv
- `multiprocessing Pool` uses global variables for common arguments

# 0.3.0 (2019-05-21)

- Added label shuffling experiment script to compute random MCC distributions for assigning statistical significance to
  predictions
- Reports if X/Y chromosomes are missing from VCF data
- Weka code moved to separate wrapper module

# 0.2.0 (2019-04-04)

- JAVA_HOME setup in run.sh
- Added multiprocessing to Weka experiment

# 0.1.0 (2019-02-25)

- Initial release
