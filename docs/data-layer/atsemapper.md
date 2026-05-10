# ATSEmapper

ATSEmapper converts per-cell junction BED files (produced by [regtools](https://regtools.readthedocs.io/)) into a [SplicingDataset](splicingdataset.md).

It is developed and maintained in its own repository: **[daklab/ATSEmapper](https://github.com/daklab/ATSEmapper)**

## What it does

1. Reads per-cell junction BED files from regtools junction extraction
2. Builds a splice graph and clusters junctions into ATSEs
3. Outputs a `SplicingDataset.h5ad` with the two layers LeafletFA and SpliceVI require

## Install and usage

See the [ATSEmapper repository](https://github.com/daklab/ATSEmapper) for installation, CLI reference, and example data.
