"""VCF record iteration, variant classification, and table construction."""

from __future__ import annotations

__all__ = [
    'VcfBatch',
    'VcfOpt',
    'VcfSvtypeMismatchError',
    'iter_vcf',
]

import logging
from pathlib import Path
import re
from typing import (
    Generator,
    Iterable,
    NamedTuple,
    Optional,
    Self,
    Union,
)
import warnings

import polars as pl
import pysam

from agglovar.expr.variant import (
    id_expr_for_table,
    sort_cols,
)
import agglovar.schema as _agg_schema

from ._vcf_const import VCF_SAMPLE_FIXED_SCHEMA, vcf_percent_decode
from ._vcf_header import (
    VcfHeader,
    _number_to_polars_type,
    read_vcf_header,
)

logger = logging.getLogger(__name__)

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

_SVLEN_END_AGREE: frozenset[str] = frozenset({'DEL', 'DUP', 'CNV', 'INV'})
"""Symbolic types whose ``INFO/SVLEN`` and ``INFO/END`` must measure the same span.

The specification defines ``SVLEN`` only for the reserved ALT types, and for these four it is a
reference-side count -- bases deleted, duplicated, copied or inverted -- so ``|SVLEN|`` and
``END - POS`` describe one quantity and a disagreement is a real conflict.  ``INS`` is reserved as
well but its ``SVLEN`` counts inserted bases, which say nothing about the reference span, and it
returns before this point.  ``SVLEN`` is undefined for a non-reserved type such as ``<CPX>``, where
a caller is free to put a query-side length in it, so there is nothing to compare and nothing to
warn about."""

_SVTYPE_REFINEMENTS: dict[str, frozenset[str]] = {
    'INS': frozenset({'INS', 'SUB', 'SNV'}),
    'DUP': frozenset({'DUP', 'INS', 'SUB', 'SNV'}),
    'DEL': frozenset({'DEL', 'SUB', 'SNV'}),
    'INV': frozenset({'INV'}),
    'CPX': frozenset({'CPX'}),
}
"""Classified types each declared ``INFO/SVTYPE`` accepts, checked by :func:`_svtype_agrees`.

``SVTYPE`` labels an event; REF and ALT spell out what actually changed, and the two disagree
routinely without contradicting each other.  A caller writing ``SVTYPE=DEL`` for a deletion that
replaces the removed bases with a few new ones (sawfish marks these with ``INFO/INSLEN``) is
describing the same event that REF and ALT resolve to ``SUB``, or to ``SNV`` once the shared
bases are stripped.  Added-sequence types accept the insertion shapes and removed-sequence types
the deletion shapes; ``INV`` and ``CPX`` have no such sequence-level reading, so anything other
than an exact match is a real conflict.  A type absent from this table is not checked at all."""

_INFO_END_DTYPE: PolarsDataType = pl.Int64
"""Dtype of the ``vcf_info_END`` column.

``INFO/END`` never reaches the accumulator through ``record.info``: htslib folds it into the record
end coordinate and :func:`_record_info_end` restores it as a scalar integer.  The column therefore
follows that value rather than the ``Number`` and ``Type`` the header declares for ``END``, which
off-spec files may write as a list or a non-integer.
"""


# ---------------------------------------------------------------------------
# Errors

class VcfSvtypeMismatchError(ValueError):
    """An allele's ``INFO/SVTYPE`` names a different event than its REF and ALT describe.

    Raised by :func:`iter_vcf`, which stops rather than choosing between the two: a record that
    declares one event and spells out another cannot be read without guessing which field the
    caller meant.  Set :attr:`VcfOpt.SVTYPE_MISMATCH_IGNORE` to route the allele to the
    ``ignored`` table with a warning instead of stopping.

    This subclasses :class:`ValueError` so that existing handlers still catch it.
    """


# ---------------------------------------------------------------------------
# VCF parsing options

class VcfOpt(NamedTuple):
    """VCF parsing options."""

    SYM_ALT_BRACE: bool = False
    """Correct for missing braces in symbolic alts, such as "CSV" instead of "<CSV>" (SVision-pro)."""

    SYM_ALT_MISMATCH_PREFER_SVTYPE: bool = False
    """Take the type from INFO/SVTYPE when it disagrees with a symbolic ALT instead of warning (SVision-pro)."""

    INS_ALT_ANCHORED: bool = False
    """Read the first ALT base of a placeholder-anchor insertion as a mistyped anchor base.

    An insertion whose ALT shares no base with REF is read as the whole inserted sequence, which
    is what Severus means by it.  longcalld writes the same shape for a different reason: it pads
    ALT like the specification requires but occasionally writes a base that does not match REF,
    and its ``len(ALT) - |SVLEN|`` is 1 for every one of its insertions.  Setting this drops that
    first base, so those records agree with the rest of the file instead of running one base long.
    The two shapes are identical per record and differ only across a whole file, so this cannot be
    detected automatically -- set it per caller.
    """

    SVTYPE_MISMATCH_IGNORE: bool = False
    """Route an allele whose ``INFO/SVTYPE`` contradicts its REF and ALT to the ``ignored`` table.

    Such an allele raises :class:`VcfSvtypeMismatchError` by default, stopping the read, because a
    systematic disagreement means the file is being read under the wrong convention.  Setting this
    keeps the rest of a callset readable when only a few records are affected, at the cost of
    dropping them.  See :data:`_SVTYPE_REFINEMENTS` for what counts as a contradiction.
    """

    FILTER_RETAIN_NULL: bool = False
    """Leave the "filter" column null for records with no FILTER instead of an empty list.

    A record with no FILTER reads as an empty list by default, the same as PASS, because callers
    that write the missing value on every record still mean "nothing failed" (Severus, PAV).
    Setting this keeps the VCF distinction between "filters were not applied" (null) and "passed"
    (empty list), at the cost of a column that reads as null in every filter check.
    """

    def __or__(self, other: Self) -> Self:
        """Combine two VcfOpt objects."""
        return self._make(a | b for a, b in zip(self, other))

    @classmethod
    def only(cls, name: str) -> Self:
        """Create a VcfOpt with only the specified flag set to True."""
        name = name.upper()
        if name not in cls._fields:
            raise ValueError(
                f"{name!r} is not a valid field of {cls.__name__} "
                f"(expected one of {cls._fields})"
            )
        return cls(**{field: field == name for field in cls._fields})


def _get_vcf_opt(
    *args: list[str | VcfOpt]
) -> VcfOpt:
    """Get VCF parsing options from a list of strings or VcfOpt objects.

    :param args: List of strings or VcfOpt objects.
    :return: VCF parsing options.
    """
    vcf_opt = VcfOpt()

    for opt_val in args:
        if isinstance(opt_val, str):
            for opt_val_item in re.split(r',|\|', opt_val):
                opt_val_item = opt_val_item.strip()

                if opt_val_item:
                    vcf_opt |= VcfOpt.only(opt_val_item)

        elif isinstance(opt_val, VcfOpt):
            vcf_opt |= opt_val

        else:
            raise ValueError(f"Invalid vcf_opt: {opt_val}")

    return vcf_opt


# ---------------------------------------------------------------------------
# Variant classification

def _svtype_agrees(svtype: str, vartype: str) -> bool:
    """Check whether a classified *vartype* is compatible with a declared *svtype*.

    :param svtype: ``INFO/SVTYPE`` base type, upper-cased with any subtype removed.
    :param vartype: Type classified from REF and ALT.

    :returns: `True` when the two describe the same event, or when *svtype* is a type this
        reader does not model and so has nothing to contradict.
    """
    if svtype == vartype:
        return True

    accepted = _SVTYPE_REFINEMENTS.get(svtype)

    return True if accepted is None else vartype in accepted


def _classify_sequence_allele(
    pos0: int,
    vcf_ref: str,
    vcf_alt: str,
    svtype: Optional[str] = None,
    info_end: Optional[int] = None,
    ins_alt_anchored: bool = False,
    label: str = '',
) -> tuple[str, Optional[int], Optional[str], Optional[str], Optional[str], int, int]:
    """Classify a sequence-resolved ALT allele.

    Strips common suffix then common prefix (both case-insensitive) to find the
    minimal changed region and derive the variant type.

    The VCF specification requires an indel to repeat the base before the event in both REF
    and ALT, which the prefix strip removes.  Callers such as Severus write REF as ``"N"``
    instead of that base and put the whole insertion in ALT, so no anchor base exists to
    strip and both sides are left holding sequence.  When *svtype* declares INS, those
    alleles are read as the insertion itself: the type comes from ``INFO/SVTYPE`` and the
    length and sequence come from ALT, with POS anchoring the insertion exactly as it does
    for a symbolic ALT.  ``INFO/SVLEN`` is deliberately not consulted -- ALT is the assembled
    consensus, while SVLEN is an independent breakpoint estimate that disagrees with it in
    roughly a third of records in a Severus callset.  Without an ``INFO/SVTYPE`` such alleles
    stay ``SUB``, which keeps a spec-conforming complex substitution at a genuine reference
    ``N`` from being reinterpreted.

    An inversion reaches the same point for a different reason: ALT spells the inverted span
    rather than a replacement, so the strips find no anchor base and the residuals describe
    nothing.  When *svtype* declares INV the event is taken to be the REF span itself, which is
    how svim and svim-asm write it -- no anchor base, ``INFO/END`` on the last REF base.  The
    span is read from the raw record rather than from the residuals: a reverse-complemented ALT
    matches REF at both edges whenever the span ends in its own complement, which strips real
    bases from about one inversion in six.  ``INFO/END`` leads where it disagrees, as it does
    for a symbolic ALT.  Sequence-resolved inversions are not in the VCF specification, which
    describes them only as symbolic ALTs or breakends, so this reads a convention rather than a
    documented encoding; it is gated on ``INFO/SVTYPE`` for that reason.

    The mirrored deletion shape, a sequence REF against a placeholder ALT, is not recognised:
    no known caller writes it, so there is no data to pin its coordinates against.

    :param pos0: 0-based position of the REF allele (``record.start``).
    :param vcf_ref: REF allele string.
    :param vcf_alt: ALT allele string (one alternate, not symbolic).
    :param svtype: ``INFO/SVTYPE`` for this allele, upper-cased with any subtype removed.
    :param info_end: ``INFO/END`` for this record, or `None` if it declares none.  Supplied by
        :func:`_record_info_end`, which reports only an END that reaches past the REF span.
    :param ins_alt_anchored: Read the first ALT base of a placeholder-anchor insertion as a
        mistyped anchor base and drop it, rather than as the first inserted base.  See
        :attr:`VcfOpt.INS_ALT_ANCHORED`.
    :param label: Optional context string appended to warning messages.
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

    # Placeholder anchor: no base could be stripped, so the insertion sits past the span the
    # record declares, the same offset _classify_symbolic_allele applies for a symbolic ALT
    if ref and alt and svtype == 'INS':
        # The caller does pad ALT, it just wrote a base REF disagrees with, so the padding is
        # dropped here exactly as the common-prefix strip would have dropped a matching one
        if ins_alt_anchored and len(vcf_ref) == 1:
            return ('INS', len(alt) - 1, None, None, alt[1:], pos0 + 1, pos0 + 2)

        if len(ref) > 1:
            warnings.warn(
                f'iter_vcf{label}: INFO/SVTYPE=INS discards a {len(ref)} bp REF allele '
                f'{ref!r}; reading ALT as the inserted sequence',
                UserWarning, stacklevel=5,
            )
        ins_pos = pos0 + len(vcf_ref)
        return ('INS', len(alt), None, None, alt, ins_pos, ins_pos + 1)

    # Inversion: the REF span is the inverted span, read from the raw record because the strips
    # above cut into it whenever the reverse-complemented ALT happens to match REF at an edge
    if ref and alt and svtype == 'INV':
        ref_end = pos0 + len(vcf_ref)
        inv_end = info_end if info_end is not None else ref_end

        if inv_end != ref_end:
            warnings.warn(
                f'iter_vcf{label}: INV INFO/END ends the inversion at {inv_end} but the REF '
                f'allele spans to {ref_end}; using INFO/END',
                UserWarning, stacklevel=5,
            )

        # REF holds the inverted bases only when it spans the whole event
        inv_seq = vcf_ref if inv_end == ref_end else None
        return ('INV', inv_end - pos0, None, None, inv_seq, pos0, inv_end)

    # SUB: varlen is the total span of changed bases (ref + alt); ref/alt cols carry sequences
    return ('SUB', len(ref) + len(alt), ref, alt, None, pos, pos + len(ref))


def _scalar(v: object) -> object:
    """Return the first element if *v* is a tuple, otherwise *v* unchanged."""
    return v[0] if isinstance(v, tuple) and v else (None if isinstance(v, tuple) else v)


def _allele_info(
    info_vals: dict[str, object],
    key: str,
    allele_k: int,
    number_a: frozenset[str],
) -> object:
    """Read INFO *key* for one allele of a record.

    ``Number=A`` fields carry one value per ALT allele and are indexed by *allele_k*; every
    other shape reduces to a scalar.  Unlike :func:`_scalar` alone, this does not hand
    allele 1's value to every allele of a multiallelic record.

    :param info_vals: INFO values for the record.
    :param key: INFO key to read.
    :param allele_k: 1-based ALT allele index.
    :param number_a: INFO keys the header declares as ``Number=A``.

    :returns: The value for this allele, or `None` if the record does not carry one.
    """
    raw = info_vals.get(key)

    if key in number_a and isinstance(raw, tuple):
        return raw[allele_k - 1] if allele_k - 1 < len(raw) else None

    return _scalar(raw)


def _normalize_field_value(
    kind: str,
    key: str,
    raw: object,
    is_string: bool,
    is_scalar: bool,
    warned: set[str],
    n_alt: Optional[int] = None,
) -> object:
    """Percent-decode an INFO or FORMAT value and repair one htslib splits against its ``Number``.

    htslib splits a value on commas before ``Number`` is consulted, so a field declared ``Number=1``
    holding ``DEL[1,234]:INV[5,678]`` arrives as a three-element tuple and :func:`_scalar` would
    keep only ``'DEL[1'``.  The specification requires a comma inside a value to be written
    ``%2C``, so this only happens for a file that does not encode, but it happens silently: the
    truncated value is a perfectly good string and nothing downstream can tell it was cut.
    Rejoining the parts restores the field text exactly, because the split consumed nothing but the
    delimiter itself.

    That recovery is only available for text.  A numeric ``Number=1`` field carrying several values
    is a genuine disagreement between the header and the data rather than a delimiter artifact, so
    it keeps the first value and is only reported.  A ``Number=A`` field is likewise unrecoverable:
    which commas delimit alleles and which were meant literally cannot be told apart, and the
    per-allele indexing in :func:`_allele_info` is already reading the wrong element.

    Each field is reported once per file.  A writer that escapes nothing produces the same problem
    on every record it writes, and one message per field is enough to identify it.

    :param kind: ``'INFO'`` or ``'FORMAT'``, naming the field in warnings.
    :param key: Field ID.
    :param raw: Value as pysam returned it, either a scalar or a tuple.
    :param is_string: Whether the header declares ``Type=String`` or ``Type=Character``.
    :param is_scalar: Whether the header declares ``Number=1``.
    :param warned: Field names already reported, added to in place.
    :param n_alt: Number of ALT alleles when the header declares ``Number=A``, else `None`.

    :returns: The decoded value, with a split ``Number=1`` text value rejoined.
    """
    if is_string:
        if isinstance(raw, tuple):
            raw = tuple(
                vcf_percent_decode(v) if isinstance(v, str) else v for v in raw
            )
        elif isinstance(raw, str):
            raw = vcf_percent_decode(raw)

    if not isinstance(raw, tuple):
        return raw

    if is_scalar and len(raw) > 1:
        if f'{kind}/{key}' not in warned:
            warned.add(f'{kind}/{key}')
            cause = (
                'which happens when a comma inside it was not written as %2C; rejoining them'
                if is_string
                else 'so the header and the data disagree; keeping the first value and '
                     'discarding the rest'
            )
            warnings.warn(
                f'iter_vcf: {kind}/{key} declares Number=1 but a value arrived as {len(raw)} '
                f'values, {cause}.  Further records with this field are not reported',
                UserWarning, stacklevel=3,
            )

        if is_string:
            return ','.join(raw)

    elif n_alt is not None and len(raw) != n_alt:
        if f'{kind}/{key}' not in warned:
            warned.add(f'{kind}/{key}')
            warnings.warn(
                f'iter_vcf: {kind}/{key} declares Number=A but a value arrived as {len(raw)} '
                f'values for {n_alt} ALT alleles, which happens when a comma inside it was not '
                f'written as %2C; per-allele values for this field are unreliable.  Further '
                f'records with this field are not reported',
                UserWarning, stacklevel=3,
            )

    return raw


def _record_info_end(record: pysam.VariantRecord) -> Optional[int]:
    """Recover ``INFO/END`` from a pysam record.

    htslib folds ``INFO/END`` into the record end coordinate and pysam never exposes it through
    ``record.info``, so it has to be read back from ``record.stop``.  That attribute falls back to
    ``start + len(REF)`` for records declaring no END, so a record counts as declaring one only when
    ``record.stop`` reaches past the REF span.  An END written redundantly (equal to the implicit
    REF end) therefore reads as absent, which costs nothing: it carries no information the implicit
    end does not.

    :param record: Record to read.

    :returns: The ``INFO/END`` value, or `None` if the record declares no END.
    """
    return record.stop if record.stop != record.start + len(record.ref) else None


def _classify_symbolic_allele(
    pos0: int,
    alt_str: str,
    info_vals: dict[str, object],
    info_end: Optional[int] = None,
    label: str = '',
    alt_prefer_svtype: Optional[bool] = None,
    allele_k: int = 1,
    number_a: frozenset[str] = frozenset(),
) -> tuple[str, Optional[int], Optional[str], Optional[str], Optional[str], int, int]:
    """Classify a symbolic ALT such as ``<INS>``, ``<DEL:ME>``, ``<INV>``, ``<DUP>``.

    :param pos0: 0-based position of the REF allele (``record.start``).  A symbolic ALT pads REF
        with the base preceding the event, so the event itself starts at ``pos0 + 1``.
    :param alt_str: The symbolic ALT string including angle brackets.
    :param info_vals: Pre-fetched INFO values for this record keyed by field ID.
    :param info_end: ``INFO/END`` for this record, or `None` if it declares none.  Supplied by
        :func:`_record_info_end` rather than read from *info_vals*, which never carries it.
    :param label: Optional context string appended to warning messages.
    :param alt_prefer_svtype: If a symbolic ALT and INFO/SVTYPE do not agree, prefer the INFO/SVTYPE
        value if True, prefer the symbolic ALT value if False. If `None`, use the symbolic ALT value
        and emit a warning.
    :param allele_k: 1-based ALT allele index, used to read per-allele INFO fields.
    :param number_a: INFO keys the header declares as ``Number=A``.

    :returns: ``(vartype, varlen, ref_col, alt_col, seq_col, pos, end)``
    :raises ValueError: If the variant is an INS with no SVLEN or SEQ in INFO.
    """
    alt_base = alt_str[1:-1].split(':')[0].upper()   # '<DEL:ME>' → 'DEL'

    svtype_raw = _allele_info(info_vals, 'SVTYPE', allele_k, number_a)
    # SVTYPE carries subtypes as well ('DUP:TANDEM'), so both sides compare as base types
    svtype = svtype_raw.upper().split(':')[0] if isinstance(svtype_raw, str) else None

    vartype = alt_base or svtype or 'UNK'

    if alt_base and svtype and alt_base != svtype:
        # True and False resolve the disagreement silently, None warns and keeps the ALT type
        if alt_prefer_svtype is None:
            warnings.warn(
                f'iter_vcf{label}: symbolic ALT {alt_str!r} disagrees with '
                f'INFO/SVTYPE={svtype_raw!r}; using symbolic ALT type',
                UserWarning, stacklevel=5,
            )
        elif alt_prefer_svtype:
            vartype = svtype

    pos = pos0 + 1   # Step over the REF padding base to the first affected base

    svlen_raw = _allele_info(info_vals, 'SVLEN', allele_k, number_a)
    seq_raw = _allele_info(info_vals, 'SEQ', allele_k, number_a)

    svlen: Optional[int] = abs(int(svlen_raw)) if svlen_raw is not None else None
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
        return ('INS', varlen, None, None, seq, pos, pos + 1)

    # DEL, INV, DUP, and any other non-INS symbolic type.  SEQ is deliberately absent: it is the
    # query-side allele, so it measures the same span as SVLEN and END only for a clean deletion,
    # and an inversion carrying internal indels disagrees with both without contradicting either
    sources: dict[str, int] = {}
    if svlen is not None:
        sources['SVLEN'] = svlen
    if info_end is not None:
        # INFO/END is 1-based inclusive, which is also the 0-based half-open end
        sources['END'] = info_end - pos

    # Callers disagree over whether POS names the padding base or the first affected base, so a
    # 1 bp spread is convention noise rather than a real conflict
    if (
        vartype in _SVLEN_END_AGREE
        and len(sources) > 1
        and max(sources.values()) - min(sources.values()) > 1
    ):
        warnings.warn(
            f'iter_vcf{label}: conflicting varlen sources for {vartype} '
            f'{alt_str!r}: {sources}; using highest-priority value',
            UserWarning, stacklevel=5,
        )

    if 'SVLEN' in sources:
        varlen = sources['SVLEN']
    elif 'END' in sources:
        varlen = sources['END']
    elif seq is not None:
        # Last resort: no length was declared, so the allele itself has to supply one
        varlen = len(seq)
    else:
        varlen = None

    if vartype in _SVLEN_END_AGREE and varlen is not None:
        # varlen measures the reference here, so it holds "end - pos == varlen"
        end = pos + varlen
    elif info_end is not None:
        # A non-reserved type may carry a query-side SVLEN, which says nothing about where the
        # event ends on the reference; END is the only source that does
        end = info_end
    else:
        # Nothing states where the event ends on the reference, so claim only the anchored base
        # rather than stretching the span by a length that may not measure the reference at all
        end = pos + 1

    return (vartype, varlen, None, None, seq, pos, end)


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
        base-table row.  A FORMAT column is scalar only for ``Number=1``; every other
        ``Number``, ``'A'`` included, is a list.  Sample rows carry no allele index, so a
        ``Number='A'`` list holds one value per ALT in ALT order and is indexed with
        ``vcf_allele - 1`` from the base table.
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

    def get_sample_callset(
        self,
        sample: str,
        vartype: str,
    ) -> pl.LazyFrame:
        """Get the full callset of one variant type for one sample.

        The per-type table named by *vartype* is joined to :attr:`sample_table` on ``vcf_rec``,
        keeping every variant row of that type and attaching the FORMAT columns *sample* carries
        for its record.  Variant columns come first and sample columns second.

        A FORMAT field declared ``Number='A'`` is one list per record in :attr:`sample_table`,
        ordered by ALT and carrying no allele index of its own.  Each is indexed with the variant
        row's ``vcf_allele`` and collapsed to a scalar, so the column holds the value for that
        row's allele alone (a ``list[f64]`` column becomes ``f64``).

        :param sample: Sample name declared in the VCF header.
        :param vartype: Variant-type table to read (``'insdel'``, ``'snv'``, ``'cpx'``, ...).

        :returns: The variant table for *sample*.  Nothing is collected.

        :raises ValueError: If *vartype* is not a variant-type table, *sample* is not declared in
            the VCF header, or a FORMAT field collides with a variant table column name.
        """
        if vartype not in _agg_schema.STANDARD_FIELDS:
            raise ValueError(
                f'VcfBatch.get_sample_callset: unknown variant type {vartype!r}: expected one '
                f'of {", ".join(_agg_schema.STANDARD_FIELDS)}'
            )

        if sample not in self.header.samples:
            raise ValueError(
                f'VcfBatch.get_sample_callset: unknown sample {sample!r}: the VCF header '
                f'declares {len(self.header.samples)} sample(s)'
            )

        var_lf: pl.LazyFrame = getattr(self, vartype)
        var_cols: list[str] = var_lf.collect_schema().names()
        sample_schema: pl.Schema = self.sample_table.collect_schema()

        # vcf_rec is the join key and is already carried by the variant table
        sample_cols: list[str] = [col for col in sample_schema.names() if col != 'vcf_rec']

        collision = sorted(set(sample_cols) & set(var_cols))
        if collision:
            raise ValueError(
                f'VcfBatch.get_sample_callset: FORMAT field(s) {", ".join(collision)} collide '
                f'with {vartype} table column names'
            )

        # Right join: the variant table drives the row set, so a sample row for a record whose
        # alleles all went to other type tables drops out
        callset = (
            self.sample_table
            .filter(pl.col('vcf_sample') == sample)
            .join(var_lf, on='vcf_rec', how='right')
            .select(var_cols + sample_cols)
        )

        # Collapse Number='A' FORMAT lists to this row's allele.  A Flag column is Boolean
        # whatever its Number, so the columns are matched on dtype rather than on Number alone
        allele_cols = [
            fmt.id for fmt in self.header.formats
            if fmt.number == 'A' and isinstance(sample_schema.get(fmt.id), pl.List)
        ]

        if allele_cols:
            callset = callset.with_columns(
                pl.col(col).list.get(pl.col('vcf_allele') - 1, null_on_oob=True)
                for col in allele_cols
            )

        return callset


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

    # The base table is expanded one row per ALT allele, so Number=A INFO values are scalar there
    info_schema: dict[str, PolarsDataType] = {
        f'vcf_info_{info.id}': _number_to_polars_type(info.number, info.type, per_allele=True)
        for info in header.info
    }

    # END is reconstructed as a scalar int, so its column ignores the declared Number and Type
    if 'vcf_info_END' in info_schema:
        info_schema['vcf_info_END'] = _INFO_END_DTYPE

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

    # The sample table is one row per (vcf_rec, vcf_sample) and carries no allele index, so
    # Number=A FORMAT values stay a list ordered by ALT; index it with the base table vcf_allele
    sample_schema: dict[str, PolarsDataType] = dict(VCF_SAMPLE_FIXED_SCHEMA)
    for fmt in header.formats:
        sample_schema[fmt.id] = _number_to_polars_type(fmt.number, fmt.type)

    return combined_schema, type_cols, ignored_schema, sample_schema


# ---------------------------------------------------------------------------
# Header checks

def _warn_header_types(header: VcfHeader) -> None:
    """Warn about INFO and FORMAT declarations that will not be read as declared.

    Called once per file before any record is parsed so that header problems surface ahead of the
    column-construction failures they would otherwise cause in :func:`_emit_batch`, where the
    offending field is much harder to identify.

    :param header: Parsed VCF header.
    """
    for kind, records in (('INFO', header.info), ('FORMAT', header.formats)):
        for rec in records:
            for err in rec.errors:
                warnings.warn(
                    f'iter_vcf: {kind}/{rec.id} header is invalid: {err}; reading it as '
                    f'Number={rec.number!r} Type={rec.type!r}, which may disagree with the data',
                    UserWarning, stacklevel=3,
                )

            if rec.type == 'Flag' and rec.number != 0:
                warnings.warn(
                    f'iter_vcf: {kind}/{rec.id} declares Type=Flag with Number={rec.number!r} '
                    f'(Flag requires Number=0); reading it as Boolean',
                    UserWarning, stacklevel=3,
                )

            elif rec.number == 0 and rec.type != 'Flag':
                warnings.warn(
                    f'iter_vcf: {kind}/{rec.id} declares Number=0 with Type={rec.type!r} '
                    f'(Number=0 requires Type=Flag); the field carries no readable value and '
                    f'its column will be null for every record',
                    UserWarning, stacklevel=3,
                )

    for info in header.info:
        if info.id == 'END' and (info.number != 1 or info.type != 'Integer'):
            warnings.warn(
                f'iter_vcf: INFO/END declares Number={info.number!r} Type={info.type!r} '
                f'(expected Number=1 Type=Integer); vcf_info_END is recovered from the record '
                f'end coordinate and typed {_INFO_END_DTYPE}',
                UserWarning, stacklevel=3,
            )

        if info.number == 'G':
            warnings.warn(
                f'iter_vcf: INFO/{info.id} declares Number=G, which htslib exposes only for '
                f'FORMAT fields; vcf_info_{info.id} will be null for every record',
                UserWarning, stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Batch helpers

def _frame_from_cols(
    cols: dict[str, list],
    schema: dict[str, PolarsDataType],
    table: str,
) -> pl.DataFrame:
    """Build a DataFrame from column accumulators, naming the offending column on failure.

    Polars reports a value that does not match its column dtype as a bare :class:`TypeError`
    naming neither the column nor the record, which is not enough to find the VCF field at fault.
    On failure the columns are rebuilt one at a time to identify it.  The happy path is a single
    :class:`polars.DataFrame` construction and pays nothing for this.

    :param cols: Column accumulators keyed by column name.
    :param schema: Target dtype for each column.
    :param table: Table name used in the error message.

    :raises TypeError: If a column value does not match its schema dtype.
    """
    try:
        return pl.DataFrame(cols, schema=schema)

    except (TypeError, OverflowError) as exc:
        for col, dtype in schema.items():
            values = cols.get(col, [])

            try:
                pl.Series(col, values, dtype=dtype)
                continue
            except (TypeError, OverflowError):
                pass

            # Narrow to the first value the dtype rejects
            detail = ''
            vcf_rec = cols.get('vcf_rec', [])
            for row, value in enumerate(values):
                try:
                    pl.Series(col, [value], dtype=dtype)
                except (TypeError, OverflowError):
                    loc = (
                        f'vcf_rec={vcf_rec[row]}' if row < len(vcf_rec)
                        else f'batch row {row}'
                    )
                    detail = (
                        f' at {loc}: got {type(value).__name__} {value!r}'
                    )
                    break

            source = (
                f' (INFO/{col[len("vcf_info_"):]} as declared in the VCF header)'
                if col.startswith('vcf_info_') else ''
            )

            raise TypeError(
                f'iter_vcf: {table} column {col!r}{source} does not match its declared type '
                f'{dtype}{detail}'
            ) from exc

        raise


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
    """Build a :class:`VcfBatch` from accumulated column lists.

    :raises TypeError: If an accumulated value does not match its column type, which points to a
        VCF field whose data disagrees with the type declared for it in the header.
    """
    combined_df = _frame_from_cols(base_cols, combined_schema, 'base')

    tables: dict[str, pl.LazyFrame] = {
        tname: (
            combined_df
            .filter(pl.col('_table_key') == tname)
            .select(tcols)
            .lazy()
            .with_columns(id_expr_for_table(tname))
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
        ignored=_frame_from_cols(ignored_cols, ignored_schema, 'ignored').lazy(),
        sample_table=_frame_from_cols(sample_cols, sample_schema, 'sample').lazy(),
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
    vcf_opt: Iterable[str | VcfOpt] | str | VcfOpt = ()
) -> Generator[VcfBatch, None, None]:
    """Iterate over a VCF/BCF file, yielding :class:`VcfBatch` objects.

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

    INFO and FORMAT fields that cannot be read as the header declares them — an invalid
    ``Number`` or ``Type``, a ``Flag`` with a non-zero ``Number``, ``Number=0`` on a
    non-``Flag``, or an INFO field with ``Number=G`` — warn once per file before the
    first record is read.

    :param path: Path to a ``.vcf``, ``.vcf.gz``, or ``.bcf`` file.
    :param batch_size: Target number of base-table rows per batch.
    :param chrom: Restrict to records on this chromosome. Requires a tabix/CSI
        index for efficient access; falls back to a linear scan for unindexed files.
    :param start: 0-based half-open BED start bound. Requires *chrom*.
    :param end: 0-based half-open BED end bound. Requires *chrom*.
    :param vcf_opt: VCF parsing options.

    :raises FileNotFoundError: If *path* does not exist.
    :raises ValueError: If *start*/*end* is given without *chrom*, *start* > *end*,
        or a FORMAT field name collides with a reserved column name.
    """
    if isinstance(vcf_opt, (str, VcfOpt)):
        vcf_opt = _get_vcf_opt(*(vcf_opt,))
    elif vcf_opt is None:
        vcf_opt = _get_vcf_opt()
    else:
        vcf_opt = _get_vcf_opt(*vcf_opt)

    # None leaves ALT/SVTYPE disagreements warning and resolving to the ALT type
    alt_prefer_svtype: Optional[bool] = True if vcf_opt.SYM_ALT_MISMATCH_PREFER_SVTYPE else None

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
        _warn_header_types(header)
        combined_schema, type_cols, ignored_schema, sample_schema = _build_schemas(header)

        alt_set = {s.id.upper() for s in header.alts}

        info_ids: list[str] = [info.id for info in header.info]
        _has_info_end: bool = 'END' in info_ids
        _info_number_a: frozenset[str] = frozenset(
            i.id for i in header.info if i.number == 'A'
        )
        _info_scalar: frozenset[str] = frozenset(
            i.id for i in header.info if i.number in (0, 1)
        )
        # pysam refuses Number=G INFO reads ('genotype is only valid as a format field');
        # _warn_header_types() reports these once per file
        _info_genotype: frozenset[str] = frozenset(
            i.id for i in header.info if i.number == 'G'
        )
        _info_string: frozenset[str] = frozenset(
            i.id for i in header.info if i.type in ('String', 'Character')
        )
        fmt_ids: list[str] = [fmt.id for fmt in header.formats]
        _fmt_scalar: frozenset[str] = frozenset(
            f.id for f in header.formats if f.number in (0, 1)
        )
        _fmt_number_a: frozenset[str] = frozenset(
            f.id for f in header.formats if f.number == 'A'
        )
        _fmt_string: frozenset[str] = frozenset(
            f.id for f in header.formats if f.type in ('String', 'Character')
        )
        # Fields needing percent-decoding or a Number check; every other field skips the call
        _info_normalize: frozenset[str] = _info_string | _info_scalar | _info_number_a
        _fmt_normalize: frozenset[str] = (_fmt_string | _fmt_scalar | _fmt_number_a) - {'GT'}
        # Field names already reported by _normalize_field_value, one warning each per file
        _warned_fields: set[str] = set()

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

            vcf_pos_1 = record.pos   # pysam record.pos is the 1-based VCF POS
            vcf_id_val: Optional[str] = (
                record.id if record.id and record.id != '.' else None
            )
            vcf_qual_val: Optional[float] = (
                float(record.qual) if record.qual is not None else None
            )

            # htslib folds a missing FILTER into an empty filter set, so '.' never reaches
            # record.filter and only PASS has to be dropped to leave the codes that failed
            filter_keys = list(record.filter.keys())
            failed_filters = [f for f in filter_keys if f != 'PASS']
            if failed_filters:
                filter_val: Optional[list] = failed_filters
            elif not filter_keys and vcf_opt.FILTER_RETAIN_NULL:
                filter_val = None
            else:
                filter_val = []

            record_info: dict[str, object] = {}
            for iid in info_ids:
                if iid in _info_genotype:
                    record_info[iid] = None
                    continue
                try:
                    raw_info = record.info[iid]
                except (KeyError, ValueError):
                    record_info[iid] = None
                    continue

                if raw_info is not None and iid in _info_normalize:
                    raw_info = _normalize_field_value(
                        'INFO', iid, raw_info,
                        iid in _info_string, iid in _info_scalar, _warned_fields,
                        len(alts) if iid in _info_number_a else None,
                    )

                record_info[iid] = raw_info

            # pysam drops INFO/END from record.info; restore it for the vcf_info_END column
            record_end = _record_info_end(record)

            if _has_info_end:
                record_info['END'] = record_end

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

                # Correct symbolic ALTs with missing braces
                if vcf_opt.SYM_ALT_BRACE and alt_str.upper() in alt_set:
                    alt_str = f'<{alt_str.upper()}>'

                label = f' (vcf_rec={rec_idx}, allele={allele_k})'
                symbolic = alt_str.startswith('<')

                # SVTYPE carries subtypes as well ('DEL:ME'), so compare base types.  A symbolic
                # ALT states its own type and settles a disagreement through alt_prefer_svtype,
                # so only the sequence path reads SVTYPE here
                svtype_raw = (
                    None if symbolic
                    else _allele_info(record_info, 'SVTYPE', allele_k, _info_number_a)
                )
                svtype = (
                    str(svtype_raw).upper().split(':')[0]
                    if svtype_raw is not None else None
                )

                try:
                    if symbolic:
                        vt, vl, rc, ac, sc, pos, end_val = _classify_symbolic_allele(
                            record.start, alt_str, record_info, record_end, label,
                            alt_prefer_svtype=alt_prefer_svtype,
                            allele_k=allele_k, number_a=_info_number_a,
                        )
                    else:
                        vt, vl, rc, ac, sc, pos, end_val = _classify_sequence_allele(
                            record.start, record.ref, alt_str,
                            svtype=svtype,
                            info_end=record_end,
                            ins_alt_anchored=vcf_opt.INS_ALT_ANCHORED,
                            label=label,
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

                # A declared type that names a different event than REF and ALT spell out is a
                # contradiction, not a coarser label for the same change
                if svtype is not None and not _svtype_agrees(svtype, vt):
                    reason = (
                        f'INFO/SVTYPE={svtype_raw!r} conflicts with the {vt} that REF and ALT '
                        f'describe'
                    )
                    if not vcf_opt.SVTYPE_MISMATCH_IGNORE:
                        raise VcfSvtypeMismatchError(f'iter_vcf{label}: {reason}')

                    warnings.warn(f'iter_vcf{label}: {reason}', UserWarning, stacklevel=2)
                    _append_ignored(
                        ignored_cols, record.contig, filter_val, vt, reason, vcf_common,
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
                base_cols['id'].append(None)   # filled by id_expr_for_table() in _emit_batch
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
                            continue

                        if raw is not None and fid in _fmt_normalize:
                            raw = _normalize_field_value(
                                'FORMAT', fid, raw,
                                fid in _fmt_string, fid in _fmt_scalar, _warned_fields,
                                len(alts) if fid in _fmt_number_a else None,
                            )

                        if fid in _fmt_scalar:
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
