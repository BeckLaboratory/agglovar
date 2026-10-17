# Changelog

<!--
Style guide:
https://common-changelog.org/
-->

<!-- TEMPLATE
## [x.y.z] - YYYY-MM-DD

### Changed
- 

### Added
- 

### Removed
-

### Fixed
-

-->

## [Unreleased] - TBD

### Changed
- Limit the `INFO/SVLEN` vs `INFO/END` cross-check to `DEL`, `DUP`, `CNV`, and `INV`; drop `INFO/SEQ` from it, which is query-side and measures a different span
- Take `end` from `INFO/END` for non-reserved symbolic types (`<CPX>`), never from `SVLEN`
- Treat a null `filter` as passing in sorts, ID versioning, and `match_filter_pass`

### Added
- `VcfSvtypeMismatchError`, raised when `INFO/SVTYPE` contradicts the type REF and ALT describe
- `VcfOpt.INS_ALT_ANCHORED`, `VcfOpt.SVTYPE_MISMATCH_IGNORE`, and `VcfOpt.FILTER_RETAIN_NULL`
- `vcf_percent_encode()` and `vcf_percent_decode()` for reserved characters in INFO and FORMAT values

### Fixed
- Keep the highest-weight pair when a merged variant has more than one candidate; the selection depended on row order and thread count, and ties now go to the lower index
- Read sequence-resolved inversions (svim, svim-asm) from raw record coordinates; prefix/suffix stripping cut into the inverted span
- Rejoin comma-split `Number=1` string values instead of silently truncating them
- Read per-allele INFO on the symbolic path; every allele took allele 1's `SVTYPE`, `SVLEN`, and `SEQ`
- Correct `match_filter_pass`, which matched exactly one side passing instead of both or neither

## [0.0.1.dev18] - 2026-08-27

### Changed
- Tune pairwise intersect, merge, and BED operation performance
- Improve complex merge performance

### Fixed
- Symbolic ALT parsing failures and off-by-one coordinate errors in the VCF reader

## [0.0.1.dev17] - 2026-07-07

### Changed
- Improve join performance

### Added
- Logging facilities

### Fixed
- Chromosome ranking on empty callsets

## [0.0.1.dev16] - 2026-06-24

### Fixed
- Crash when `seqmatch.MatchScoreModel.match_prop()` returned an int where Polars expected a float

## [0.0.1.dev15] - 2026-06-23

### Changed
- **Breaking:** `merge` accepts per-source metadata and records it in the merge; the `mg_src` schema changed to support it
- **Breaking:** `mg_src_lead` is an index into `mg_src` rather than a copy of the lead source

### Added
- `expr.variant.lead_src_expr()` to extract lead source information from `mg_src`

## [0.0.1.dev14] - 2026-05-06

### Changed
- Back BED intersect with IEJoin

### Added
- `vcf` submodule reading VCF into per-type batches, backed by pysam

### Fixed
- Polars indexing bug causing missed intersects
- Intersect performance lost to re-materializing tables
- Missing field bugs in merges and a schema dtype bug

## [0.0.1.dev12] - 2026-02-18

### Fixed
- Schema bug in concat while merging

## [0.0.1.dev11] - 2026-02-09

### Fixed
- BED joining bug

## [0.0.1.dev10] - 2026-02-09

### Fixed
- Index bugs in variant intersects

## [0.0.1.dev9] - 2026-02-08

### Added
- `util` subpackage
- `librl()` to reload libraries for interactive development

### Fixed
- Variant merging and figure interface issues

## [0.0.1.dev8] - 2025-12-05

### Added
- `match_vartype` and `match_filter_pass` match parameters

## [0.0.1.dev7] - 2025-11-27

### Added
- `bed` submodule for BED-like intersect operations

## [0.0.1.dev6] - 2025-11-26

### Fixed
- Handling of additional filters and columns in pairwise merging

## [0.0.1.dev5] - 2025-11-26

### Changed
- Stabilize merging operations

### Added
- Segment overlap intersect (`seg_ro_min`)
- Non-chunking pairwise intersect option

### Fixed
- Polars handling of empty tables

## [0.0.1.dev4] - 2025-11-06

### Fixed
- Parameter bugs in merge

## [0.0.1.dev3] - 2025-11-06

### Changed
- Restructure cumulative merge source and stat columns

### Added
- Cumulative merge strategy, taking a list of callset tables and producing one merged table with merge stats
- Read the Docs configuration and PyPI publishing

## [0.0.1.dev2] - 2025-09-19

First tagged release.

### Changed
- Restructure the package and convert `join.pairwise` to a class
- Move match comparisons to edlib

### Added
- Pairwise intersect with affine-gap scoring, accepting list parameters for multi-affine models
- Sphinx documentation
- First unit tests

[Unreleased]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev18...HEAD
[0.0.1.dev18]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev17...v0.0.1.dev18
[0.0.1.dev17]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev16...v0.0.1.dev17
[0.0.1.dev16]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev15...v0.0.1.dev16
[0.0.1.dev15]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev14...v0.0.1.dev15
[0.0.1.dev14]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev12...v0.0.1.dev14
[0.0.1.dev12]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev11...v0.0.1.dev12
[0.0.1.dev11]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev10...v0.0.1.dev11
[0.0.1.dev10]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev9...v0.0.1.dev10
[0.0.1.dev9]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev8...v0.0.1.dev9
[0.0.1.dev8]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev7...v0.0.1.dev8
[0.0.1.dev7]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev6...v0.0.1.dev7
[0.0.1.dev6]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev5...v0.0.1.dev6
[0.0.1.dev5]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev4...v0.0.1.dev5
[0.0.1.dev4]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev3...v0.0.1.dev4
[0.0.1.dev3]: https://github.com/BeckLaboratory/agglovar/compare/v0.0.1.dev2...v0.0.1.dev3
[0.0.1.dev2]: https://github.com/BeckLaboratory/agglovar/releases/tag/v0.0.1.dev2
