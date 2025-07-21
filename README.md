# Agglovar toolkit for fast genomic variant transformations and intersects

Agglovar is a fast toolkit based on Polars to perform fast variant transformations and intersections between
callsets. It defines a standard schema for genomic variants based on Apache Arrow, which Polars uses natively. Whenever
possible, Aggolvar uses Parquet files to store data allowing it to preserve the schema and take advantage of both
columnar storage and pushdown optimizations for fast queries and transformations.

Agglovar is designed to replace the SV-Pop library (https://github.com/EichlerLab/svpop).

The name Agglovar is a portmanteau of the latin word "agglomerare" (to gather) and "variant" (genomic variants).

## Alpha release

The "a" in the version number indicates that Agglovar is an alpha version and is still under active development.

The first full release will be version 1.0.0.
