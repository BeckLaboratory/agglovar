"""VCF record iteration, variant classification, and table construction."""

from __future__ import annotations

__all__ = [
    'VcfBatch',
    'iter_vcf',
]

import warnings
from pathlib import Path
from typing import Generator, Optional, Union

import polars as pl

import agglovar.schema as _agg_schema
from agglovar.expr.variant import id_expr, sort_cols

from ._vcf_const import VCF_SAMPLE_FIXED_SCHEMA
from ._vcf_header import VcfHeader, _number_to_polars_type, read_vcf_header

import pysam

PolarsDataType = Union[pl.DataType, type[pl.DataType]]


# ---------------------------------------------------------------------------
# Module-level constants

_VCF_SUFFIX_SCHEMA: dict[str, PolarsDataType] = {
    'vcf_pos':    pl.Int64,
    'vcf_id':     pl.String,
    'vcf_ref':    pl.String,
    'vcf_alt':    pl.String,
    'vcf_qual':   pl.Float32,
    'vcf_rec':    pl.Int64,
    'vcf_allele': pl.Int32,
}
"""Raw VCF columns appended to every per-type base table (not part of STANDARD_FIELDS)."""

_VARTYPE_TO_TABLE: dict[str, str] = {
    'INS': 'insdel',
    'DEL': 'insdel',
    'SNV': 'snv',
    'SUB': 'sub',
    'DUP': 'dup',
    'INV': 'inv',
    'CPX': 'cpx',
}
"""Maps a classified vartype string to its destination table in :class:`VcfBatch`."""

_KEEP_IGNORED_VARTYPE: frozenset[str] = frozenset({'BND', 'TRA'})
"""Vartype strings preserved verbatim in the ``ignored`` table; all others become ``null``."""


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
    # SUB: varlen is the total span of changed bases (ref + alt); ref/alt cols carry sequences
    return ('SUB', len(ref) + len(alt), ref, alt, None, pos, pos + len(ref))


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

    svlen_raw = _scalar(info_vals.get('SVLEN'))
    end_raw = _scalar(info_vals.get('END'))
    seq_raw = _scalar(info_vals.get('SEQ'))

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

    if info_end is not None:
        end = info_end
    elif varlen is not None:
        end = pos0 + varlen
    else:
        end = pos0 + 1

    return (vartype, varlen, None, None, seq, pos0, end)


# ---------------------------------------------------------------------------
# GT formatting

def _format_gt(alleles: Optional[tuple], phased: bool) -> Optional[str]:
    """Reconstruct a VCF GT string from pysam's integer-tuple representation."""
    if alleles is None:
        return None
    sep = '|' if phased else '/'
    return sep.join('.' if a is None else str(a) for a in alleles)


# ---------------------------------------------------------------------------
# Result class

class VcfBatch:
    """A batch of classified variants from :func:`iter_vcf`.

    Per-type tables are :class:`polars.LazyFrame` objects that can be filtered or
    projected before collecting with ``.collect()``. Within each batch the rows are
    sorted by ``chrom``, ``pos``, ``end``, and ``id``, but sort order is not
    guaranteed across batches from the same file.

    Each base table contains the columns defined in
    :data:`agglovar.schema.STANDARD_FIELDS` for that type, followed by the raw VCF
    columns ``vcf_pos``, ``vcf_id``, ``vcf_ref``, ``vcf_alt``, ``vcf_qual``,
    ``vcf_rec``, and ``vcf_allele``, then any ``vcf_info_*`` columns declared in
    the VCF header.

    :attr header: VCF header parsed from the source file.
    :attr snv: SNV variants (vartype ``SNV``).
    :attr insdel: Insertion and deletion variants (vartype ``INS`` or ``DEL``).
    :attr inv: Inversion variants (vartype ``INV``).
    :attr dup: Duplication variants (vartype ``DUP``).
    :attr sub: Multi-nucleotide substitutions (vartype ``SUB``); ``varlen`` is
        ``len(ref) + len(alt)``.
    :attr cpx: Complex variants (vartype ``CPX``).
    :attr ignored: Records that could not be routed to a type table — BND breakends,
        TRA, unrecognized symbolic types, and classification errors. The
        ``vcf_ignored`` column contains the reason; ``vartype`` is preserved for
        known types (``BND``, ``TRA``) and ``null`` otherwise.
    :attr sample_table: Long-format genotype data — one row per
        ``(vcf_rec, vcf_sample)`` pair, for records that produced at least one
        base-table row.
    """

    __slots__ = (
        'header', 'snv', 'insdel', 'inv', 'dup', 'sub', 'cpx',
        'ignored', 'sample_table', '_counts',
    )

    def __init__(
        self,
        header: VcfHeader,
        snv: pl.LazyFrame,
        insdel: pl.LazyFrame,
        inv: pl.LazyFrame,
        dup: pl.LazyFrame,
        sub: pl.LazyFrame,
        cpx: pl.LazyFrame,
        ignored: pl.LazyFrame,
        sample_table: pl.LazyFrame,
        _counts: dict[str, int],
    ) -> None:
        """Store the per-type frames and the originating header on this batch."""
        self.header = header
        self.snv = snv
        self.insdel = insdel
        self.inv = inv
        self.dup = dup
        self.sub = sub
        self.cpx = cpx
        self.ignored = ignored
        self.sample_table = sample_table
        self._counts = _counts

    def __repr__(self) -> str:
        """Return a debug summary including per-type record counts."""
        c = self._counts
        counts_str = (
            f'snv={c["snv"]}, insdel={c["insdel"]}, inv={c["inv"]}, '
            f'dup={c["dup"]}, sub={c["sub"]}, cpx={c["cpx"]}, '
            f'ignored={c["ignored"]}'
        )
        return f'VcfBatch({counts_str})\n{self.header!r}'


# ---------------------------------------------------------------------------
# Schema construction

def _build_schemas(
    header: VcfHeader,
) -> tuple[
    dict[str, PolarsDataType],   # combined base schema (all types + routing key)
    dict[str, list[str]],        # per-type column lists for .select() projection
    dict[str, PolarsDataType],   # ignored table schema
    dict[str, PolarsDataType],   # sample table schema
]:
    """Build Polars schemas and column lists from *header*.

    :raises ValueError: If a FORMAT field name collides with a reserved sample-table column.
    """
    _reserved = frozenset({'vcf_sample', 'vcf_rec'})
    for fmt in header.formats:
        if fmt.id in _reserved:
            raise ValueError(
                f'iter_vcf: FORMAT field {fmt.id!r} collides with reserved '
                f'sample-table column name'
            )

    info_schema: dict[str, PolarsDataType] = {
        f'vcf_info_{info.id}': _number_to_polars_type(info.number, info.type)
        for info in header.info
    }

    # Per-type column projection: STANDARD_FIELDS cols + vcf_* suffix + info cols
    type_cols: dict[str, list[str]] = {
        tname: list(fields) + list(_VCF_SUFFIX_SCHEMA) + list(info_schema)
        for tname, fields in _agg_schema.STANDARD_FIELDS.items()
    }

    # Combined base schema: union of all VARIANT fields + vcf_* suffix + info + routing key
    combined_schema: dict[str, PolarsDataType] = dict(_agg_schema.VARIANT)
    combined_schema.update(_VCF_SUFFIX_SCHEMA)
    combined_schema.update(info_schema)
    combined_schema['_table_key'] = pl.String

    ignored_schema: dict[str, PolarsDataType] = {
        'chrom':       pl.String,
        'filter':      pl.List(pl.String),
        'vartype':     pl.String,
        'vcf_ignored': pl.String,
        **_VCF_SUFFIX_SCHEMA,
    }

    sample_schema: dict[str, PolarsDataType] = dict(VCF_SAMPLE_FIXED_SCHEMA)
    for fmt in header.formats:
        sample_schema[fmt.id] = _number_to_polars_type(fmt.number, fmt.type)

    return combined_schema, type_cols, ignored_schema, sample_schema


# ---------------------------------------------------------------------------
# Batch helpers

def _append_ignored(
    cols: dict[str, list],
    chrom: str,
    filter_val: Optional[list],
    vartype: Optional[str],
    reason: str,
    vcf_common: dict[str, object],
) -> None:
    """Append one row to the ignored column accumulator."""
    cols['chrom'].append(chrom)
    cols['filter'].append(filter_val)
    cols['vartype'].append(vartype)
    cols['vcf_ignored'].append(reason)
    for col, val in vcf_common.items():
        cols[col].append(val)


def _emit_batch(
    header: VcfHeader,
    base_cols: dict[str, list],
    ignored_cols: dict[str, list],
    sample_cols: dict[str, list],
    combined_schema: dict[str, PolarsDataType],
    type_cols: dict[str, list[str]],
    ignored_schema: dict[str, PolarsDataType],
    sample_schema: dict[str, PolarsDataType],
    counts: dict[str, int],
) -> VcfBatch:
    """Build a :class:`VcfBatch` from accumulated column lists."""
    combined_df = pl.DataFrame(base_cols, schema=combined_schema)

    tables: dict[str, pl.LazyFrame] = {
        tname: (
            combined_df
            .filter(pl.col('_table_key') == tname)
            .select(tcols)
            .lazy()
            .with_columns(id_expr)
            .sort(list(sort_cols))
        )
        for tname, tcols in type_cols.items()
    }

    return VcfBatch(
        header=header,
        snv=tables['snv'],
        insdel=tables['insdel'],
        inv=tables['inv'],
        dup=tables['dup'],
        sub=tables['sub'],
        cpx=tables['cpx'],
        ignored=pl.DataFrame(ignored_cols, schema=ignored_schema).lazy(),
        sample_table=pl.DataFrame(sample_cols, schema=sample_schema).lazy(),
        _counts=dict(counts),
    )


# ---------------------------------------------------------------------------
# Public iterator

def iter_vcf(
    path: str | Path,
    *,
    batch_size: int = 10_000,
    chrom: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Generator[VcfBatch, None, None]:
    """
    Iterate over a VCF/BCF file, yielding :class:`VcfBatch` objects.

    Each batch contains one :class:`polars.LazyFrame` per variant type, an
    ``ignored`` frame, and a ``sample_table`` frame. The caller may filter or
    project any frame before calling ``.collect()``.

    Batches contain at least *batch_size* base-table rows, but records are never
    split: a batch may exceed *batch_size* when the final record has multiple ALT
    alleles. The last batch may be smaller. At least one batch is always yielded,
    even for empty files.

    Rows within each batch are sorted by ``chrom``, ``pos``, ``end``, and ``id``.
    Sort order is not guaranteed across batches.

    BND breakend alleles (containing ``[`` or ``]``) and unrecognized symbolic
    types are silently routed to the ``ignored`` table rather than emitted as
    base-table rows.

    :param path: Path to a ``.vcf``, ``.vcf.gz``, or ``.bcf`` file.
    :param batch_size: Target number of base-table rows per batch.
    :param chrom: Restrict to records on this chromosome. Requires a tabix/CSI
        index for efficient access; falls back to a linear scan for unindexed files.
    :param start: 0-based half-open BED start bound. Requires *chrom*.
    :param end: 0-based half-open BED end bound. Requires *chrom*.

    :raises FileNotFoundError: If *path* does not exist.
    :raises ValueError: If *start*/*end* is given without *chrom*, *start* > *end*,
        or a FORMAT field name collides with a reserved column name.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')

    if (start is not None or end is not None) and chrom is None:
        raise ValueError('start/end require chrom')
    if start is not None and end is not None and start > end:
        raise ValueError(f'start ({start}) > end ({end})')

    vcf_file: pysam.VariantFile = pysam.VariantFile(str(path))
    try:
        header = read_vcf_header(vcf_file)
        combined_schema, type_cols, ignored_schema, sample_schema = _build_schemas(header)

        info_ids: list[str] = [info.id for info in header.info]
        _info_number_a: frozenset[str] = frozenset(
            i.id for i in header.info if i.number == 'A'
        )
        _info_scalar: frozenset[str] = frozenset(
            i.id for i in header.info if i.number in (0, 1)
        )
        fmt_ids: list[str] = [fmt.id for fmt in header.formats]
        _fmt_scalar: frozenset[str] = frozenset(
            f.id for f in header.formats if f.number in (0, 1)
        )
        sample_names: list[str] = header.samples
        has_samples = bool(sample_names)

        fetch_iter = (
            vcf_file.fetch(chrom, start, end) if chrom is not None
            else vcf_file.fetch()
        )

        _table_types = list(_agg_schema.STANDARD_FIELDS)

        def _new_base() -> dict[str, list]:
            return {col: [] for col in combined_schema}

        def _new_ignored() -> dict[str, list]:
            return {col: [] for col in ignored_schema}

        def _new_sample() -> dict[str, list]:
            return {col: [] for col in sample_schema}

        def _new_counts() -> dict[str, int]:
            return {t: 0 for t in _table_types + ['ignored']}

        base_cols = _new_base()
        ignored_cols = _new_ignored()
        sample_cols = _new_sample()
        counts = _new_counts()
        n_base_rows = 0
        rec_idx = 0

        for record in fetch_iter:
            alts = record.alts or ()

            vcf_pos_1 = record.pos + 1
            vcf_id_val: Optional[str] = (
                record.id if record.id and record.id != '.' else None
            )
            vcf_qual_val: Optional[float] = (
                float(record.qual) if record.qual is not None else None
            )

            filter_keys = list(record.filter.keys())
            actual_filters = [f for f in filter_keys if f not in ('.', 'PASS')]
            if not filter_keys or set(filter_keys) <= {'.'}:
                filter_val: Optional[list] = None
            elif actual_filters:
                filter_val = actual_filters
            else:
                filter_val = []

            record_info: dict[str, object] = {}
            for iid in info_ids:
                try:
                    record_info[iid] = record.info[iid]
                except KeyError:
                    record_info[iid] = None

            record_emitted = False

            for allele_k, alt_str in enumerate(alts, start=1):
                vcf_common = {
                    'vcf_pos':    vcf_pos_1,
                    'vcf_id':     vcf_id_val,
                    'vcf_ref':    record.ref,
                    'vcf_alt':    alt_str,
                    'vcf_qual':   vcf_qual_val,
                    'vcf_rec':    rec_idx,
                    'vcf_allele': allele_k,
                }

                # BND breakend notation — route to ignored before classifying
                if '[' in alt_str or ']' in alt_str:
                    _append_ignored(
                        ignored_cols, record.contig, filter_val,
                        'BND', 'BND breakend allele', vcf_common,
                    )
                    counts['ignored'] += 1
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
                    _append_ignored(
                        ignored_cols, record.contig, filter_val,
                        None, str(exc), vcf_common,
                    )
                    counts['ignored'] += 1
                    continue

                table_key = _VARTYPE_TO_TABLE.get(vt)
                if table_key is None:
                    ignored_vt = vt if vt in _KEEP_IGNORED_VARTYPE else None
                    reason = (
                        f'{vt} structural variant' if ignored_vt is not None
                        else f'unrecognized variant type {vt!r}'
                    )
                    _append_ignored(
                        ignored_cols, record.contig, filter_val,
                        ignored_vt, reason, vcf_common,
                    )
                    counts['ignored'] += 1
                    continue

                # Append to combined base accumulator
                base_cols['chrom'].append(record.contig)
                base_cols['pos'].append(pos)
                base_cols['end'].append(end_val)
                base_cols['id'].append(None)   # filled by id_expr in _emit_batch
                base_cols['vartype'].append(vt)
                base_cols['varlen'].append(vl)
                base_cols['ref'].append(rc)
                base_cols['alt'].append(ac)
                base_cols['filter'].append(filter_val)
                base_cols['seq'].append(sc)
                for col, val in vcf_common.items():
                    base_cols[col].append(val)

                for iid in info_ids:
                    raw = record_info[iid]
                    if raw is None:
                        info_val: object = None
                    elif iid in _info_number_a:
                        info_val = (
                            raw[allele_k - 1]
                            if isinstance(raw, tuple) and allele_k - 1 < len(raw)
                            else None
                        )
                    elif iid in _info_scalar:
                        info_val = _scalar(raw)
                    else:
                        info_val = list(raw) if isinstance(raw, tuple) else raw
                    base_cols[f'vcf_info_{iid}'].append(info_val)

                base_cols['_table_key'].append(table_key)
                counts[table_key] += 1
                n_base_rows += 1
                record_emitted = True

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

            if n_base_rows >= batch_size:
                yield _emit_batch(
                    header, base_cols, ignored_cols, sample_cols,
                    combined_schema, type_cols, ignored_schema, sample_schema,
                    counts,
                )
                base_cols = _new_base()
                ignored_cols = _new_ignored()
                sample_cols = _new_sample()
                counts = _new_counts()
                n_base_rows = 0

        yield _emit_batch(
            header, base_cols, ignored_cols, sample_cols,
            combined_schema, type_cols, ignored_schema, sample_schema,
            counts,
        )

    finally:
        vcf_file.close()
