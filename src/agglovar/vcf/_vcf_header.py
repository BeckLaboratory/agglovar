"""VCF header representation and parsing."""

from __future__ import annotations

__all__ = [
    'ContigHeader',
    'InfoHeader',
    'FilterHeader',
    'FormatHeader',
    'AltHeader',
    'VcfHeader',
    'read_vcf_header',
]

import datetime
import reprlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

import polars as pl

from ._vcf_const import VCF_SOURCE, VCF_TO_POLARS_TYPE, VCF_VERSION

if TYPE_CHECKING:
    import pysam

PolarsDataType = Union[pl.DataType, type[pl.DataType]]

# --- Module-level constants ---

_VALID_TYPES: frozenset[str] = frozenset(VCF_TO_POLARS_TYPE)
_VALID_NUMBER_STRS: frozenset[str] = frozenset({'A', 'R', 'G', '.'})
_VCF_FIXED_COLS: frozenset[str] = frozenset({
    '#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT',
})

# Keys handled individually in read_vcf_header; not stored as generic meta
_META_KNOWN_KEYS: frozenset[str] = frozenset({
    'fileformat', 'fileDate', 'source', 'reference',
})


# --- Internal helpers ---

def _parse_number(number: int | str, field_id: str = '') -> int | str:
    """Validate and normalize a VCF Number value.

    :param number: Raw value from a header or user input.
    :param field_id: Field ID used in error messages.

    :returns: Non-negative :class:`int` or one of ``'A'``, ``'R'``, ``'G'``, ``'.'``.

    :raises ValueError: If the value is not a valid VCF Number.
    """
    ctx = f' for {field_id!r}' if field_id else ''

    if isinstance(number, int):
        if number < 0:
            raise ValueError(f'VCF Number{ctx} must be non-negative, got {number}')
        return number

    s = str(number).strip()

    if s in _VALID_NUMBER_STRS:
        return s

    try:
        n = int(s)
        if n < 0:
            raise ValueError(f'VCF Number{ctx} must be non-negative, got {s!r}')

        return n

    except ValueError:
        raise ValueError(
            f'VCF Number{ctx} invalid value {s!r}; '
            f'expected a non-negative integer or one of {sorted(_VALID_NUMBER_STRS)}'
        )


def _parse_type(type_str: str, field_id: str = '') -> str:
    """Validate a VCF Type string.

    :raises ValueError: If ``type_str`` is not a valid VCF Type.
    """
    if type_str not in _VALID_TYPES:
        ctx = f' for {field_id!r}' if field_id else ''
        raise ValueError(
            f'VCF Type{ctx} invalid value {type_str!r}; '
            f'expected one of {sorted(_VALID_TYPES)}'
        )
    return type_str


def _number_to_polars_type(number: int | str, type_str: str) -> PolarsDataType:
    """Map a VCF (Number, Type) pair to a Polars data type.

    * ``Flag`` / ``Number=0`` → :class:`polars.Boolean`
    * ``Number=1`` or ``Number='A'`` → scalar base type (``'A'`` = one value per alt allele;
      in the per-allele expanded table this is already scalar)
    * ``Number='R'``, ``'G'``, ``'.'``, or integer > 1 → :class:`polars.List` of base type
    """
    if type_str == 'Flag':
        return pl.Boolean

    base: PolarsDataType = VCF_TO_POLARS_TYPE[type_str]

    if number == 1 or number == 'A':
        return base

    return pl.List(base)


def _escape_desc(value: str) -> str:
    """Escape backslashes and double-quotes in a VCF Description or similar field."""
    return value.replace('\\', '\\\\').replace('"', '\\"')


def _fmt_number(number: int | str) -> str:
    """Format a validated VCF Number value for output."""
    return str(number)


def _get_attrs(rec: object) -> dict[str, str]:
    """Extract pysam header record ``attrs`` as a ``{key: value}`` string dict.

    pysam returns ``attrs`` as a tuple of ``(key, value)`` string pairs. This helper
    converts it safely regardless of whether ``attrs`` is a tuple, a mapping, or absent.
    """
    raw = getattr(rec, 'attrs', None)

    if not raw:
        return {}

    try:
        pairs = raw.items() if hasattr(raw, 'items') else raw
        return {str(k): str(v) for k, v in pairs}

    except (TypeError, ValueError):
        return {}


# --- Header record dataclasses ---

@dataclass
class _VcfRecord(ABC):
    """Abstract base for all VCF header record dataclasses.

    :param id: Record identifier.
    :param extra: Any additional key/value attributes not covered by the subclass fields.
    :param errors: Validation error messages collected when the record was constructed.
        An empty tuple means the record is valid. Excluded from equality and repr.
    """

    id: str
    extra: dict[str, str] = field(default_factory=dict, kw_only=True)
    errors: tuple[str, ...] = field(default=(), repr=False, compare=False, kw_only=True)

    @property
    def is_valid(self) -> bool:
        """``True`` if this record has no validation errors."""
        return not self.errors

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __str__(self) -> str: ...


@dataclass
class ContigHeader(_VcfRecord):
    """A single ``##contig`` header record.

    :param id: Contig name.
    :param length: Sequence length in bases, or ``None`` if not specified.
    :param url: URL referencing the sequence, or ``None``.
    :param md5: MD5 checksum of the sequence, or ``None``.
    :param assembly: Assembly identifier, or ``None``.
    """

    length: Optional[int] = None
    url: Optional[str] = None
    md5: Optional[str] = None
    assembly: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate fields after dataclass initialization."""
        errs: list[str] = []
        if self.length is not None and self.length < 0:
            errs.append(f'length must be non-negative, got {self.length}')
            self.length = None
        self.errors = tuple(errs)

    def __repr__(self) -> str:
        """Return a debug representation of this contig header."""
        parts = [f'id={self.id!r}']
        if self.length is not None:
            parts.append(f'length={self.length!r}')
        return f'ContigHeader({", ".join(parts)})'

    def __str__(self) -> str:
        """Return the VCF-formatted ``##contig`` header line."""
        parts = [f'ID={self.id}']
        if self.length is not None:
            parts.append(f'length={self.length}')
        if self.url is not None:
            parts.append(f'URL={self.url}')
        if self.md5 is not None:
            parts.append(f'md5={self.md5}')
        if self.assembly is not None:
            parts.append(f'assembly={self.assembly}')
        for k, v in self.extra.items():
            parts.append(f'{k}={v}')
        return f'##contig=<{",".join(parts)}>'


@dataclass
class InfoHeader(_VcfRecord):
    """A single ``##INFO`` header record.

    :param id: Field identifier.
    :param number: Number of values: non-negative ``int``, or ``'A'``, ``'R'``, ``'G'``,
        or ``'.'`` for per-allele, per-allele-including-ref, per-genotype, and variable.
    :param type: VCF type string: one of ``'Integer'``, ``'Float'``, ``'Flag'``,
        ``'Character'``, ``'String'``.
    :param description: Human-readable description.
    :param source: Optional source annotation (VCF 4.2 §1.4.1).
    :param version: Optional version annotation (VCF 4.2 §1.4.1).
    """

    number: int | str
    type: str
    description: str
    source: Optional[str] = None
    version: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate ``number`` and ``type`` fields after dataclass initialization."""
        errs: list[str] = []
        try:
            self.number = _parse_number(self.number, self.id)
        except ValueError as exc:
            errs.append(str(exc))
            self.number = '.'
        try:
            self.type = _parse_type(self.type, self.id)
        except ValueError as exc:
            errs.append(str(exc))
            self.type = 'String'
        self.errors = tuple(errs)

    def __repr__(self) -> str:
        """Return a debug representation of this INFO header."""
        return (
            f'InfoHeader(id={self.id!r}, number={self.number!r}, '
            f'type={self.type!r}, description={reprlib.repr(self.description)})'
        )

    def __str__(self) -> str:
        """Return the VCF-formatted ``##INFO`` header line."""
        parts = [
            f'ID={self.id}',
            f'Number={_fmt_number(self.number)}',
            f'Type={self.type}',
            f'Description="{_escape_desc(self.description)}"',
        ]
        if self.source is not None:
            parts.append(f'Source="{_escape_desc(self.source)}"')
        if self.version is not None:
            parts.append(f'Version="{_escape_desc(self.version)}"')
        for k, v in self.extra.items():
            parts.append(f'{k}={v}')
        return f'##INFO=<{",".join(parts)}>'


@dataclass
class FilterHeader(_VcfRecord):
    """A single ``##FILTER`` header record.

    :param id: Filter identifier (e.g. ``'PASS'``, ``'LowQual'``).
    :param description: Human-readable description.
    """

    description: str

    def __repr__(self) -> str:
        """Return a debug representation of this FILTER header."""
        return f'FilterHeader(id={self.id!r}, description={reprlib.repr(self.description)})'

    def __str__(self) -> str:
        """Return the VCF-formatted ``##FILTER`` header line."""
        parts = [
            f'ID={self.id}',
            f'Description="{_escape_desc(self.description)}"',
        ]
        for k, v in self.extra.items():
            parts.append(f'{k}={v}')
        return f'##FILTER=<{",".join(parts)}>'


@dataclass
class FormatHeader(_VcfRecord):
    """A single ``##FORMAT`` header record.

    :param id: Field identifier (e.g. ``'GT'``, ``'DP'``).
    :param number: Number of values (same encoding as :class:`InfoHeader`).
    :param type: VCF type string (same encoding as :class:`InfoHeader`).
    :param description: Human-readable description.
    """

    number: int | str
    type: str
    description: str

    def __post_init__(self) -> None:
        """Validate ``number`` and ``type`` fields after dataclass initialization."""
        errs: list[str] = []
        try:
            self.number = _parse_number(self.number, self.id)
        except ValueError as exc:
            errs.append(str(exc))
            self.number = '.'
        try:
            self.type = _parse_type(self.type, self.id)
        except ValueError as exc:
            errs.append(str(exc))
            self.type = 'String'
        self.errors = tuple(errs)

    def __repr__(self) -> str:
        """Return a debug representation of this FORMAT header."""
        return (
            f'FormatHeader(id={self.id!r}, number={self.number!r}, '
            f'type={self.type!r}, description={reprlib.repr(self.description)})'
        )

    def __str__(self) -> str:
        """Return the VCF-formatted ``##FORMAT`` header line."""
        parts = [
            f'ID={self.id}',
            f'Number={_fmt_number(self.number)}',
            f'Type={self.type}',
            f'Description="{_escape_desc(self.description)}"',
        ]
        for k, v in self.extra.items():
            parts.append(f'{k}={v}')
        return f'##FORMAT=<{",".join(parts)}>'


@dataclass
class AltHeader(_VcfRecord):
    """A single ``##ALT`` header record.

    :param id: Symbolic allele identifier without angle brackets (e.g. ``'INS'``, ``'DEL'``).
    :param description: Human-readable description.
    """

    description: str

    def __repr__(self) -> str:
        """Return a debug representation of this ALT header."""
        return f'AltHeader(id={self.id!r}, description={reprlib.repr(self.description)})'

    def __str__(self) -> str:
        """Return the VCF-formatted ``##ALT`` header line."""
        parts = [
            f'ID={self.id}',
            f'Description="{_escape_desc(self.description)}"',
        ]
        for k, v in self.extra.items():
            parts.append(f'{k}={v}')
        return f'##ALT=<{",".join(parts)}>'


# --- Main header class ---

class VcfHeader:
    """
    Mutable, validated VCF file header.

    All mutations validate their input immediately. Invalid field values (bad Number or
    Type strings) are stored with fallback values rather than raising, so broken VCF files
    can be read without crashing. Each stored record tracks its own errors via the
    ``errors`` tuple and ``is_valid`` property. The header-level :attr:`is_valid` and
    :attr:`validation_errors` aggregate across all records, making it easy to gate writes
    on validity.

    The default constructor produces a minimal valid header with no fields::

        hdr = VcfHeader()
        hdr.add_contig('chr1', length=248_956_422)
        hdr.add_info('SVTYPE', number=1, type='String', description='Variant type')
        hdr.add_sample('NA12878')
        print(hdr)              # VCF header text
        hdr.is_valid            # True / False
        hdr.validation_errors   # list of error strings

    :param vcf_version: VCF spec version (default ``'4.2'``).
    :param file_date: Date string ``YYYYMMDD``. Defaults to today when ``None``.
    :param source: Value for ``##source``. Pass an empty string or ``None`` to suppress.
    :param reference: Value for ``##reference``. Pass ``None`` to suppress.
    :param warn_invalid: When ``True`` (default), invalid field values emit a
        :class:`UserWarning`. Set to ``False`` to suppress these warnings (e.g. when
        bulk-reading known-broken files).
    """

    def __init__(
        self,
        vcf_version: str = VCF_VERSION,
        file_date: Optional[str] = None,
        source: Optional[str] = VCF_SOURCE,
        reference: Optional[str] = None,
        warn_invalid: bool = True,
    ) -> None:
        """Create an empty header initialized with the given metadata fields."""
        vcf_version = vcf_version.strip()

        if not vcf_version:
            raise ValueError('VcfHeader: vcf_version may not be empty')

        self._vcf_version: str = vcf_version
        self._file_date: Optional[str] = (
            file_date if file_date is not None
            else datetime.date.today().strftime('%Y%m%d')
        )
        self._source: Optional[str] = source
        self._reference: Optional[str] = reference
        self._warn_invalid: bool = warn_invalid

        self._contigs: dict[str, ContigHeader] = {}
        self._info: dict[str, InfoHeader] = {}
        self._filters: dict[str, FilterHeader] = {}
        self._formats: dict[str, FormatHeader] = {}
        self._alts: dict[str, AltHeader] = {}
        self._samples: list[str] = []
        self._meta: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ properties

    @property
    def vcf_version(self) -> str:
        """VCF format version string (e.g. ``'4.2'``)."""
        return self._vcf_version

    @vcf_version.setter
    def vcf_version(self, value: str) -> None:
        value = value.strip()
        if not value:
            raise ValueError('VcfHeader.vcf_version may not be empty')
        self._vcf_version = value

    @property
    def file_date(self) -> Optional[str]:
        """File date string (``YYYYMMDD``) or ``None``."""
        return self._file_date

    @file_date.setter
    def file_date(self, value: Optional[str]) -> None:
        self._file_date = value

    @property
    def source(self) -> Optional[str]:
        """``##source`` value, or ``None`` to suppress the line."""
        return self._source

    @source.setter
    def source(self, value: Optional[str]) -> None:
        self._source = value

    @property
    def reference(self) -> Optional[str]:
        """``##reference`` value, or ``None`` to suppress the line."""
        return self._reference

    @reference.setter
    def reference(self, value: Optional[str]) -> None:
        self._reference = value

    @property
    def contigs(self) -> list[ContigHeader]:
        """Ordered list of contig records (read-only view)."""
        return list(self._contigs.values())

    @property
    def info(self) -> list[InfoHeader]:
        """Ordered list of INFO field records (read-only view)."""
        return list(self._info.values())

    @property
    def filters(self) -> list[FilterHeader]:
        """Ordered list of FILTER records (read-only view)."""
        return list(self._filters.values())

    @property
    def formats(self) -> list[FormatHeader]:
        """Ordered list of FORMAT field records (read-only view)."""
        return list(self._formats.values())

    @property
    def alts(self) -> list[AltHeader]:
        """Ordered list of ALT records (read-only view)."""
        return list(self._alts.values())

    @property
    def samples(self) -> list[str]:
        """Ordered list of sample names (read-only view)."""
        return list(self._samples)

    @property
    def meta(self) -> list[tuple[str, str]]:
        """Ordered ``(key, value)`` pairs for arbitrary ``##key=value`` metadata lines."""
        return list(self._meta)

    @property
    def warn_invalid(self) -> bool:
        """When ``True``, adding a field with invalid values emits a :class:`UserWarning`."""
        return self._warn_invalid

    @warn_invalid.setter
    def warn_invalid(self, value: bool) -> None:
        self._warn_invalid = value

    @property
    def is_valid(self) -> bool:
        """``True`` if every stored record passes its own validation."""
        return all(
            rec.is_valid
            for collection in (
                self._contigs, self._info, self._filters, self._formats, self._alts
            )
            for rec in collection.values()
        )

    @property
    def validation_errors(self) -> list[str]:
        """
        Flat list of validation error strings for all invalid records.

        Each entry is prefixed with the field type and ID for easy identification, e.g.
        ``"INFO 'SVTYPE': invalid Number value '-1'"``
        """
        errors: list[str] = []
        type_map = [
            ('INFO', self._info),
            ('FORMAT', self._formats),
            ('FILTER', self._filters),
            ('ALT', self._alts),
            ('contig', self._contigs),
        ]
        for type_name, collection in type_map:
            for fid, rec in collection.items():
                for msg in rec.errors:
                    errors.append(f"{type_name} {fid!r}: {msg}")
        return errors

    # ------------------------------------------------------------------ mutations

    def add_contig(
        self,
        id: str,
        *,
        length: Optional[int] = None,
        url: Optional[str] = None,
        md5: Optional[str] = None,
        assembly: Optional[str] = None,
        extra: Optional[dict[str, str]] = None,
    ) -> None:
        """Add a ``##contig`` record.

        :raises ValueError: If a contig with this *id* already exists.
        """
        if id in self._contigs:
            raise ValueError(f'VcfHeader.add_contig: duplicate contig ID {id!r}')
        rec = ContigHeader(id=id, length=length, url=url, md5=md5, assembly=assembly,
                           extra=extra or {})
        self._contigs[id] = rec
        self._warn(rec, 'contig')

    def remove_contig(self, id: str) -> None:
        """Remove a contig record by *id*.

        :raises KeyError: If *id* is not present.
        """
        del self._contigs[id]

    def add_info(
        self,
        id: str,
        *,
        number: int | str,
        type: str,
        description: str,
        source: Optional[str] = None,
        version: Optional[str] = None,
        extra: Optional[dict[str, str]] = None,
    ) -> None:
        """Add an ``##INFO`` record.

        Invalid *number* or *type* values are stored with fallback values (``'.'`` and
        ``'String'`` respectively) and a :class:`UserWarning` is emitted unless
        ``warn_invalid`` is ``False``.

        :raises ValueError: If an INFO field with this *id* already exists.
        """
        if id in self._info:
            raise ValueError(f'VcfHeader.add_info: duplicate INFO ID {id!r}')
        rec = InfoHeader(id=id, number=number, type=type, description=description,
                         source=source, version=version, extra=extra or {})
        self._info[id] = rec

        self._warn(rec, 'INFO')

    def remove_info(self, id: str) -> None:
        """Remove an INFO record by *id*.

        :raises KeyError: If *id* is not present.
        """
        del self._info[id]

    def add_filter(
        self,
        id: str,
        *,
        description: str,
        extra: Optional[dict[str, str]] = None,
    ) -> None:
        """Add a ``##FILTER`` record.

        :raises ValueError: If a FILTER with this *id* already exists.
        """
        if id in self._filters:
            raise ValueError(f'VcfHeader.add_filter: duplicate FILTER ID {id!r}')
        self._filters[id] = FilterHeader(id=id, description=description, extra=extra or {})

    def remove_filter(self, id: str) -> None:
        """Remove a FILTER record by *id*.

        :raises KeyError: If *id* is not present.
        """
        del self._filters[id]

    def add_format(
        self,
        id: str,
        *,
        number: int | str,
        type: str,
        description: str,
        extra: Optional[dict[str, str]] = None,
    ) -> None:
        """Add a ``##FORMAT`` record.

        Invalid *number* or *type* values are stored with fallback values (``'.'`` and
        ``'String'`` respectively) and a :class:`UserWarning` is emitted unless
        ``warn_invalid`` is ``False``.

        :raises ValueError: If a FORMAT field with this *id* already exists.
        """
        if id in self._formats:
            raise ValueError(f'VcfHeader.add_format: duplicate FORMAT ID {id!r}')
        rec = FormatHeader(id=id, number=number, type=type, description=description,
                           extra=extra or {})
        self._formats[id] = rec

        self._warn(rec, 'FORMAT')

    def remove_format(self, id: str) -> None:
        """Remove a FORMAT record by *id*.

        :raises KeyError: If *id* is not present.
        """
        del self._formats[id]

    def add_alt(
        self,
        id: str,
        *,
        description: str,
        extra: Optional[dict[str, str]] = None,
    ) -> None:
        """Add an ``##ALT`` record.

        :raises ValueError: If an ALT with this *id* already exists.
        """
        if id in self._alts:
            raise ValueError(f'VcfHeader.add_alt: duplicate ALT ID {id!r}')
        self._alts[id] = AltHeader(id=id, description=description, extra=extra or {})

    def remove_alt(self, id: str) -> None:
        """Remove an ALT record by *id*.

        :raises KeyError: If *id* is not present.
        """
        del self._alts[id]

    def add_sample(self, name: str) -> None:
        """Append a sample name.

        :raises ValueError: If *name* duplicates an existing sample or conflicts with a
            VCF fixed column name (``#CHROM``, ``POS``, etc.).
        """
        if name in _VCF_FIXED_COLS:
            raise ValueError(
                f'VcfHeader.add_sample: {name!r} conflicts with a VCF column header name'
            )
        if name in self._samples:
            raise ValueError(f'VcfHeader.add_sample: duplicate sample name {name!r}')
        self._samples.append(name)

    def remove_sample(self, name: str) -> None:
        """Remove a sample by name.

        :raises ValueError: If *name* is not present.
        """
        try:
            self._samples.remove(name)
        except ValueError:
            raise ValueError(f'VcfHeader.remove_sample: sample {name!r} not found')

    def add_meta(self, key: str, value: str) -> None:
        """Append an arbitrary ``##key=value`` metadata line.

        Structured fields (INFO, FORMAT, FILTER, contig, ALT) should be added via their
        dedicated methods. Duplicate keys are allowed since multiple lines with the same
        key are legal in VCF (they are written in insertion order).
        """
        self._meta.append((key, value))

    # ------------------------------------------------------------------ representations

    def __str__(self) -> str:
        r"""Return the complete VCF header as a string.

        Each line ends with ``\n``, including the column-header line, so the result can
        be written directly to a file with ``file.write(str(hdr))``.
        """
        lines: list[str] = []

        lines.append(f'##fileformat=VCFv{self._vcf_version}')

        if self._file_date:
            lines.append(f'##fileDate={self._file_date}')
        if self._source:
            lines.append(f'##source={self._source}')
        if self._reference:
            lines.append(f'##reference={self._reference}')

        for c in self._contigs.values():
            lines.append(str(c))
        for key, value in self._meta:
            lines.append(f'##{key}={value}')
        for i in self._info.values():
            lines.append(str(i))
        for f in self._filters.values():
            lines.append(str(f))
        for fmt in self._formats.values():
            lines.append(str(fmt))
        for a in self._alts.values():
            lines.append(str(a))

        col_line = '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO'
        if self._samples:
            col_line += '\tFORMAT'
            for s in self._samples:
                col_line += f'\t{s}'
        lines.append(col_line)

        return '\n'.join(lines) + '\n'

    def __repr__(self) -> str:
        """Return a compact summary of the header contents.

        Shows the VCF version, counts and IDs of key fields::

            VcfHeader(version='4.2', contigs=25, info=['SVTYPE', 'SVLEN', 'END'],
                      filters=['PASS'], format=['GT', 'GQ'], samples=['NA12878'])
        """
        return (
            f'VcfHeader('
            f'version={self._vcf_version!r}, '
            f'contigs={len(self._contigs)}, '
            f'info={reprlib.repr(list(self._info))}, '
            f'filters={reprlib.repr(list(self._filters))}, '
            f'format={reprlib.repr(list(self._formats))}, '
            f'samples={reprlib.repr(self._samples)}'
            f')'
        )

    # ------------------------------------------------------------------ internal

    def _warn(self, rec: object, type_name: str) -> None:
        """Emit a :class:`UserWarning` for each error on *rec* if ``warn_invalid`` is True."""
        if self._warn_invalid:
            for msg in getattr(rec, 'errors', ()):
                warnings.warn(
                    f"VcfHeader: {type_name} field {getattr(rec, 'id', '?')!r} stored "
                    f"with fallback values: {msg}",
                    UserWarning,
                    stacklevel=3,
                )


# --- Header reader ---


def read_vcf_header(vcf_file: 'pysam.VariantFile') -> VcfHeader:
    """
    Build a :class:`VcfHeader` from an open pysam :class:`~pysam.VariantFile`.

    Iterates ``vcf_file.header.records`` to preserve the original field order and to
    obtain raw string values (avoiding pysam's numeric conversions for ``Number``).
    Sample names are taken from ``vcf_file.header.samples``.

    :param vcf_file: An open :class:`pysam.VariantFile`.
    :returns: A :class:`VcfHeader` populated from the file's header.
    :raises ValueError: If the VCF version is missing or malformed.
    """
    ph = vcf_file.header

    # VCF version — pysam exposes this as e.g. "VCFv4.2"
    raw_ver = (getattr(ph, 'version', None) or '').strip()
    vcf_ver = raw_ver.lstrip('VCFv').lstrip('v').strip() or VCF_VERSION

    hdr = VcfHeader(vcf_version=vcf_ver, file_date=None, source=None, reference=None)

    for rec in ph.records:
        rec_type = str(getattr(rec, 'type', 'GENERIC')).upper()
        attrs = _get_attrs(rec)

        if rec_type == 'GENERIC':
            key = str(getattr(rec, 'key', '') or '')
            value = str(getattr(rec, 'value', '') or '')
            if key == 'fileformat':
                pass  # handled above via ph.version
            elif key == 'fileDate':
                hdr._file_date = value
            elif key == 'source':
                hdr._source = value
            elif key == 'reference':
                hdr._reference = value
            elif key:
                hdr.add_meta(key, value)

        elif rec_type == 'CONTIG':
            fid = attrs.get('ID', '')
            if not fid or fid in hdr._contigs:
                continue
            try:
                length = int(attrs['length']) if 'length' in attrs else None
            except ValueError:
                length = None
            extra = {
                k: v for k, v in attrs.items()
                if k not in {'ID', 'length', 'URL', 'md5', 'assembly'}
            }
            hdr.add_contig(
                fid,
                length=length,
                url=attrs.get('URL'),
                md5=attrs.get('md5'),
                assembly=attrs.get('assembly'),
                extra=extra,
            )

        elif rec_type == 'INFO':
            fid = attrs.get('ID', '')
            if not fid or fid in hdr._info:
                continue
            desc = attrs.get('Description', '').strip('"')
            extra = {
                k: v for k, v in attrs.items()
                if k not in {'ID', 'Number', 'Type', 'Description', 'Source', 'Version'}
            }
            hdr.add_info(
                fid,
                number=attrs.get('Number', '.'),
                type=attrs.get('Type', 'String'),
                description=desc,
                source=attrs.get('Source', '').strip('"') or None,
                version=attrs.get('Version', '').strip('"') or None,
                extra=extra,
            )

        elif rec_type == 'FILTER':
            fid = attrs.get('ID', '')
            if not fid or fid in hdr._filters:
                continue
            desc = attrs.get('Description', '').strip('"')
            extra = {k: v for k, v in attrs.items() if k not in {'ID', 'Description'}}
            hdr.add_filter(fid, description=desc, extra=extra)

        elif rec_type == 'FORMAT':
            fid = attrs.get('ID', '')
            if not fid or fid in hdr._formats:
                continue
            desc = attrs.get('Description', '').strip('"')
            extra = {k: v for k, v in attrs.items() if k not in {'ID', 'Number', 'Type', 'Description'}}
            hdr.add_format(
                fid,
                number=attrs.get('Number', '.'),
                type=attrs.get('Type', 'String'),
                description=desc,
                extra=extra,
            )

        elif rec_type == 'ALT':
            fid = attrs.get('ID', '')
            if not fid or fid in hdr._alts:
                continue
            desc = attrs.get('Description', '').strip('"')
            extra = {k: v for k, v in attrs.items() if k not in {'ID', 'Description'}}
            hdr.add_alt(fid, description=desc, extra=extra)

    for name in ph.samples:
        hdr.add_sample(name)

    return hdr
