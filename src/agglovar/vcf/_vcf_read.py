"""VCF record iteration, variant classification, and table construction."""

from __future__ import annotations

__vcf_read_all__ = [
    'iter_vcf',
]

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterable, Optional

import polars as pl
from polars.type_aliases import PolarsDataType

from ._vcf_const import VCF_BASE_FIXED_SCHEMA, VCF_SAMPLE_FIXED_SCHEMA
from ._vcf_header import VcfHeader, _number_to_polars_type, read_vcf_header

if TYPE_CHECKING:
    import pysam


# ---------------------------------------------------------------------------
# ALT filtering

# Symbolic base types that produce no output row
_SKIP_SYMBOLIC: frozenset[str] = frozenset({'BND', 'TRA'})


def _is_skip_alt(alt: str) -> bool:
    """Return ``True`` for BND breakend notation and symbolic BND/TRA ALTs."""
    # Breakend notation: A[chr1:12345[ or ]chr1:12345]A
    if '[' in alt or ']' in alt:
        return True
    if alt.startswith('<') and alt.endswith('>'):
        base = alt[1:-1].split(':')[0].upper()
        return base in _SKIP_SYMBOLIC
    return False


# ---------------------------------------------------------------------------
# Variant classification

def _classify_sequence_allele(
    pos0: int,
    vcf_ref: str,
    vcf_alt: str,
) -> tuple[str, Optional[int], Optional[str], Optional[str], Optional[str], int, int]:
    """Classify a sequence-resolved ALT allele.

    Strips common suffix then common prefix (both case-insensitive) to find the
    minimal changed region and derive the variant type.

    :param pos0: pysam 0-based record position.
    :param vcf_ref: REF allele string.
    :param vcf_alt: ALT allele string (one alternate, not symbolic).
    :returns: ``(vartype, varlen, ref_col, alt_col, seq_col, pos, end)`` where *pos*
        and *end* are 0-based half-open BED coordinates.
    """
    ref, alt = vcf_ref, vcf_alt

    # Strip common suffix (right), case-insensitive
    n = min(len(ref), len(alt))
    sfx = 0
    while sfx < n and ref[-(sfx + 1)].upper() == alt[-(sfx + 1)].upper():
        sfx += 1
    if sfx:
        ref = ref[:-sfx]
        alt = alt[:-sfx]

    # Strip common prefix (left), case-insensitive; track position offset
    n = min(len(ref), len(alt))
    pfx = 0
    while pfx < n and ref[pfx].upper() == alt[pfx].upper():
        pfx += 1
    ref = ref[pfx:]
    alt = alt[pfx:]
    pos = pos0 + pfx

    if not ref and alt:                          # INS
        return ('INS', len(alt), None, None, alt, pos, pos + 1)
    if ref and not alt:                          # DEL
        return ('DEL', len(ref), None, None, ref, pos, pos + len(ref))
    if len(ref) == 1 and len(alt) == 1:          # SNV
        return ('SNV', None, ref, alt, None, pos, pos + 1)
    # SUB: ref/alt columns carry the substituted sequences; varlen and seq are null
    return ('SUB', None, ref, alt, None, pos, pos + len(ref))


def _scalar(v: object) -> object:
    """Return the first element if *v* is a tuple, otherwise *v* unchanged."""
    return v[0] if isinstance(v, tuple) and v else (None if isinstance(v, tuple) else v)


def _classify_symbolic_allele(
    pos0: int,
    alt_str: str,
    info_vals: dict[str, object],
    label: str = '',
) -> tuple[str, Optional[int], Optional[str], Optional[str], Optional[str], int, int]:
    """Classify a symbolic ALT such as ``<INS>``, ``<DEL:ME>``, ``<INV>``, ``<DUP>``.

    :param pos0: pysam 0-based record position.
    :param alt_str: The symbolic ALT string including angle brackets.
    :param info_vals: Pre-fetched INFO values for this record keyed by field ID.
    :param label: Optional context string appended to warning messages.
    :returns: ``(vartype, varlen, ref_col, alt_col, seq_col, pos, end)``
    :raises ValueError: If the variant is an INS with no SVLEN or SEQ in INFO.
    """
    # Derive vartype from ALT base type (strip CNV sub-categories)
    alt_base = alt_str[1:-1].split(':')[0].upper()   # '<DEL:ME>' → 'DEL'

    svtype_raw = _scalar(info_vals.get('SVTYPE'))
    svtype = svtype_raw.upper() if isinstance(svtype_raw, str) else None

    if alt_base and svtype and alt_base != svtype:
        warnings.warn(
            f'iter_vcf{label}: symbolic ALT {alt_str!r} disagrees with '
            f'INFO/SVTYPE={svtype_raw!r}; using symbolic ALT type',
            UserWarning, stacklevel=5,
        )
    vartype = alt_base or svtype or 'UNK'

    # Extract numeric INFO helpers
    svlen_raw = _scalar(info_vals.get('SVLEN'))
    end_raw   = _scalar(info_vals.get('END'))
    seq_raw   = _scalar(info_vals.get('SEQ'))

    # SVLEN: always store as absolute value
    svlen: Optional[int] = abs(int(svlen_raw)) if svlen_raw is not None else None
    # INFO/END is 1-based inclusive; in 0-based half-open BED the end equals that integer
    info_end: Optional[int] = int(end_raw) if end_raw is not None else None
    seq: Optional[str] = str(seq_raw) if seq_raw is not None else None

    if vartype == 'INS':
        if svlen is not None:
            varlen: Optional[int] = svlen
            if seq is not None and len(seq) != varlen:
                warnings.warn(
                    f'iter_vcf{label}: INS SVLEN={varlen} conflicts with '
                    f'SEQ length={len(seq)}',
                    UserWarning, stacklevel=5,
                )
        elif seq is not None:
            varlen = len(seq)
        else:
            raise ValueError(
                f'iter_vcf{label}: cannot determine varlen for symbolic INS '
                f'{alt_str!r}: INFO has neither SVLEN nor SEQ'
            )
        return ('INS', varlen, None, None, seq, pos0, pos0 + 1)

    # DEL, INV, DUP, and any other non-INS symbolic type
    sources: dict[str, int] = {}
    if svlen is not None:
        sources['SVLEN'] = svlen
    if info_end is not None:
        sources['END'] = info_end - pos0
    if seq is not None:
        sources['SEQ'] = len(seq)

    if len(sources) > 1 and len(set(sources.values())) > 1:
        warnings.warn(
            f'iter_vcf{label}: conflicting varlen sources for {vartype} '
            f'{alt_str!r}: {sources}; using highest-priority value',
            UserWarning, stacklevel=5,
        )

    if 'SVLEN' in sources:
        varlen = sources['SVLEN']
    elif 'END' in sources:
        varlen = sources['END']
    elif 'SEQ' in sources:
        varlen = sources['SEQ']
    else:
        varlen = None

    # Prefer INFO/END for the end coordinate; fall back to pos + varlen
    if info_end is not None:
        end = info_end
    elif varlen is not None:
        end = pos0 + varlen
    else:
        end = pos0 + 1  # unknown extent; emit a point interval

    return (vartype, varlen, None, None, seq, pos0, end)


# ---------------------------------------------------------------------------
# Schema construction

def _build_schemas(
    header: VcfHeader,
) -> tuple[dict[str, PolarsDataType], dict[str, PolarsDataType]]:
    """Build full base and sample Polars schemas from *header*.

    :raises ValueError: If a FORMAT field name collides with a reserved sample-table column.
    """
    _reserved = frozenset({'vcf_sample', 'vcf_rec'})
    for fmt in header.formats:
        if fmt.id in _reserved:
            raise ValueError(
                f'iter_vcf: FORMAT field {fmt.id!r} collides with reserved '
                f'sample-table column name'
            )

    base_schema: dict[str, PolarsDataType] = dict(VCF_BASE_FIXED_SCHEMA)
    for info in header.info:
        base_schema[f'vcf_info_{info.id}'] = _number_to_polars_type(info.number, info.type)

    sample_schema: dict[str, PolarsDataType] = dict(VCF_SAMPLE_FIXED_SCHEMA)
    for fmt in header.formats:
        # Sample table is long-format (one row per sample), so FORMAT fields are scalar
        # regardless of how many samples the file has.
        sample_schema[fmt.id] = _number_to_polars_type(fmt.number, fmt.type)

    return base_schema, sample_schema


# ---------------------------------------------------------------------------
# GT formatting

def _format_gt(alleles: Optional[tuple], phased: bool) -> Optional[str]:
    """Reconstruct a VCF GT string from pysam's integer-tuple representation."""
    if alleles is None:
        return None
    sep = '|' if phased else '/'
    return sep.join('.' if a is None else str(a) for a in alleles)


# ---------------------------------------------------------------------------
# Public iterator

def iter_vcf(
    path: str | Path,
    *,
    batch_size: int = 10_000,
    chrom: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    vartype: Optional[str | Iterable[str]] = None,
) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
    """
    Iterate over a VCF/BCF file, yielding ``(base, sample)`` DataFrame pairs.

    Each yielded tuple contains a batch of rows controlled by *batch_size*. The generator
    always yields at least one tuple, even for empty files (both DataFrames will be
    empty but have the correct schema). Returned batches may be longer than *batch_size*
    when records have multiple alleles (does not stop mid-record), and the last batch may
    be shorter than *batch_size*.

    **Base table** — one row per alternate allele, columns:

    * Derived agglovar VARIANT fields: ``chrom``, ``pos`` (0-based), ``end`` (0-based
      half-open), ``vartype``, ``varlen``, ``ref``, ``alt``, ``seq``.
    * Raw VCF fields: ``vcf_pos`` (1-based), ``vcf_id``, ``vcf_ref``, ``vcf_alt``,
      ``vcf_qual``, ``filter``, ``vcf_rec``, ``vcf_allele``.
    * Dynamic ``vcf_info_{id}`` columns for every INFO field in the header.

    ``BND`` and ``TRA`` alternate alleles are silently dropped. If all ALTs of a record
    are dropped, the record contributes no rows to either table.

    **Sample table** — one row per ``(vcf_rec, sample_name)`` pair (long format):

    * ``vcf_rec``, ``vcf_sample`` fixed columns.
    * Dynamic FORMAT columns (e.g. ``GT``, ``GQ``) named directly from the FORMAT field IDs.

    The schema of both tables is fixed for the lifetime of the generator and is derived
    from the VCF header on the first call.

    :param path: Path to a ``.vcf``, ``.vcf.gz``, or ``.bcf`` file.
    :param batch_size: Target number of base-table rows per batch. Records are never
        split mid-way. A batch may exceed *batch_size* when the final record has multiple
        ALT alleles.
    :param chrom: Restrict to records on this chromosome. Requires a tabix/CSI index for
        efficient access; falls back to linear scan for unindexed files.
    :param start: 0-based half-open BED start bound. Requires *chrom*.
    :param end: 0-based half-open BED end bound. Requires *chrom*.
    :param vartype: Restrict output to one or more variant types (e.g. ``'INS'``,
        ``('INS', 'DEL')``). Applied after classification, before emitting.

    :raises FileNotFoundError: If *path* does not exist.
    :raises ValueError: If *start*/*end* is given without *chrom*, or *start* > *end*, or
        a FORMAT field name collides with a reserved column name.
    """
    import pysam

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')

    # Check chrom, start, end
    if (start is not None or end is not None) and chrom is None:
        raise ValueError('Start/end require chrom')

    if start is not None and end is not None and start > end:
        raise ValueError(f'Start ({start}) > end ({end})')

    # Check vartype
    vartype_set: Optional[frozenset[str]] = None

    if vartype is not None:
        if isinstance(vartype, str):
            vartype_set = frozenset({vartype.upper()})
        else:
            vartype_set = frozenset(v.upper() for v in vartype)

    vcf_file: pysam.VariantFile = pysam.VariantFile(str(path))
    try:
        header = read_vcf_header(vcf_file)
        base_schema, sample_schema = _build_schemas(header)

        # Pre-categorise INFO fields once so the per-record loop stays simple
        info_ids: list[str] = [info.id for info in header.info]
        # Number='A': one value per ALT allele → indexed per allele
        _info_number_a: frozenset[str] = frozenset(
            i.id for i in header.info if i.number == 'A'
        )
        # Number=0 (Flag) or Number=1 → scalar value
        _info_scalar: frozenset[str] = frozenset(
            i.id for i in header.info if i.number in (0, 1)
        )
        # Everything else → list

        fmt_ids: list[str] = [fmt.id for fmt in header.formats]
        # Number=0 (Flag) or Number=1 → scalar; everything else → list
        _fmt_scalar: frozenset[str] = frozenset(
            f.id for f in header.formats if f.number in (0, 1)
        )

        sample_names: list[str] = header.samples
        has_samples = bool(sample_names)

        # Choose pysam fetch region
        if chrom is not None:
            # pysam.fetch uses 0-based half-open coordinates, same as our BED convention
            fetch_iter = vcf_file.fetch(chrom, start, end)
        else:
            fetch_iter = vcf_file.fetch()

        def _new_base() -> dict[str, list]:
            return {col: [] for col in base_schema}

        def _new_sample() -> dict[str, list]:
            return {col: [] for col in sample_schema}

        base_cols = _new_base()
        sample_cols = _new_sample()
        n_rows = 0
        rec_idx = 0

        for record in fetch_iter:
            alts = record.alts or ()

            # --- Per-record scalars fetched once ---
            vcf_pos_1 = record.pos + 1   # pysam is 0-based; VCF POS is 1-based
            vcf_id_val: Optional[str] = (
                record.id if record.id and record.id != '.' else None
            )
            vcf_qual_val: Optional[float] = (
                float(record.qual) if record.qual is not None else None
            )

            # FILTER column: null = '.', [] = PASS, [ids...] = failed filters
            filter_keys = list(record.filter.keys())
            actual_filters = [f for f in filter_keys if f not in ('.', 'PASS')]
            if not filter_keys or set(filter_keys) <= {'.'}:
                filter_val: Optional[list] = None
            elif actual_filters:
                filter_val = actual_filters
            else:
                filter_val = []   # PASS or only '.'/'PASS' entries

            # Pre-fetch all INFO values once per record
            record_info: dict[str, object] = {}
            for iid in info_ids:
                try:
                    record_info[iid] = record.info[iid]
                except KeyError:
                    record_info[iid] = None

            # --- Per-allele rows ---
            record_emitted = False
            for allele_k, alt_str in enumerate(alts, start=1):
                if _is_skip_alt(alt_str):
                    continue

                label = f' (vcf_rec={rec_idx}, allele={allele_k})'
                try:
                    if alt_str.startswith('<'):
                        vt, vl, rc, ac, sc, pos, end_val = _classify_symbolic_allele(
                            record.pos, alt_str, record_info, label,
                        )
                    else:
                        vt, vl, rc, ac, sc, pos, end_val = _classify_sequence_allele(
                            record.pos, record.ref, alt_str,
                        )
                except (ValueError, TypeError) as exc:
                    warnings.warn(
                        f'iter_vcf: skipping allele{label}: {exc}',
                        UserWarning, stacklevel=2,
                    )
                    continue

                if vartype_set is not None and vt not in vartype_set:
                    continue

                # Append base-table columns
                base_cols['chrom'].append(record.contig)
                base_cols['pos'].append(pos)
                base_cols['end'].append(end_val)
                base_cols['vartype'].append(vt)
                base_cols['varlen'].append(vl)
                base_cols['ref'].append(rc)
                base_cols['alt'].append(ac)
                base_cols['seq'].append(sc)
                base_cols['vcf_pos'].append(vcf_pos_1)
                base_cols['vcf_id'].append(vcf_id_val)
                base_cols['vcf_ref'].append(record.ref)
                base_cols['vcf_alt'].append(alt_str)
                base_cols['vcf_qual'].append(vcf_qual_val)
                base_cols['filter'].append(filter_val)
                base_cols['vcf_rec'].append(rec_idx)
                base_cols['vcf_allele'].append(allele_k)

                # INFO columns
                for iid in info_ids:
                    raw = record_info[iid]
                    if raw is None:
                        val: object = None
                    elif iid in _info_number_a:
                        # One value per ALT allele: index by allele_k - 1 (0-based)
                        if isinstance(raw, tuple):
                            val = raw[allele_k - 1] if allele_k - 1 < len(raw) else None
                        else:
                            val = raw
                    elif iid in _info_scalar:
                        val = _scalar(raw)
                    else:
                        val = list(raw) if isinstance(raw, tuple) else raw
                    base_cols[f'vcf_info_{iid}'].append(val)

                record_emitted = True
                n_rows += 1

            # --- Sample rows (one per sample, keyed to this record) ---
            if record_emitted and has_samples:
                for sname in sample_names:
                    sample_cols['vcf_rec'].append(rec_idx)
                    sample_cols['vcf_sample'].append(sname)

                    try:
                        sdata = record.samples[sname]
                    except KeyError:
                        sdata = None

                    for fid in fmt_ids:
                        if sdata is None:
                            sample_cols[fid].append(None)
                            continue
                        try:
                            raw = sdata[fid]
                        except KeyError:
                            raw = None

                        if fid == 'GT':
                            sample_cols[fid].append(
                                _format_gt(raw, sdata.phased) if raw is not None else None
                            )
                        elif fid in _fmt_scalar:
                            sample_cols[fid].append(_scalar(raw))
                        else:
                            sample_cols[fid].append(
                                list(raw) if isinstance(raw, tuple) else raw
                            )

            rec_idx += 1

            # Emit when the batch target is reached; never split a record
            if n_rows >= batch_size:
                yield (
                    pl.DataFrame(base_cols, schema=base_schema),
                    pl.DataFrame(sample_cols, schema=sample_schema),
                )
                base_cols = _new_base()
                sample_cols = _new_sample()
                n_rows = 0

        # Always emit a final batch (may be empty if the file had no usable records)
        yield (
            pl.DataFrame(base_cols, schema=base_schema),
            pl.DataFrame(sample_cols, schema=sample_schema),
        )

    finally:
        vcf_file.close()
