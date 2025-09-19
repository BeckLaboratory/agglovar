"""Reference and FASTA processing utilities."""

__all__ = [
    'read_fai',
    'fa_info',
]

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import Bio.SeqIO
import polars as pl

from . import io


def read_fai(
        fai_file_name: str | Path,
        cols: Optional[Iterable[str]] = ('chrom', 'len'),
        name: str = None
) -> pl.DataFrame:
    """Read an FAI File name.

    By default, return a Series of chromosome lengths keyed by the chromosome (or contig) name.

    Available columns are: chrom, len, pos, line_bp, line_bytes

    :param fai_file_name: File to read.
    :param cols: Select these columns. None to retain all columns.
    :param name: Name of the chromosome column (typically "qry_id" to match query alignment tables).


    :returns: A table of the FAI file.
    """
    fai_cols = ['chrom', 'len', 'pos', 'line_bp', 'line_bytes']

    schema = {
        'chrom': pl.String,
        'len': pl.Int64,
        'pos': pl.Int64,
        'line_bp': pl.Int64,
        'line_bytes': pl.Int64
    }

    if cols is None:
        cols = fai_cols

    return (
        pl.scan_csv(
            fai_file_name,
            separator='\t',
            has_header=False,
            new_columns=fai_cols,
            schema_overrides=schema
        )
        .select(list(cols))
        .rename({'chrom': name} if 'chrom' in cols and name is not None else {})
        .collect()
    )


def fa_info(
        ref_fa: str | Path
) -> pl.DataFrame:
    """Get a table of reference information from a FASTA file.

    FASTA must have a ".fai".

    Table columns:
        chrom: Chromosome name.
        md5: MD5 of the sequence.
        order: Order of the sequence in the FASTA file.
        pos: Start position of the sequence in the FASTA file.
        line_bp: Number of base pairs per line.
        line_bytes: Number of bytes per line.

    :param ref_fa: Reference FASTA.

    :returns: A table with sequence information.
    """
    # Read FAI
    record_list = list()
    record_count = 0

    # Read sequneces and get MD5
    with io.PlainOrGzReader(ref_fa, 'rt') as in_file:
        for record in Bio.SeqIO.parse(in_file, 'fasta'):
            record_list.append(
                [
                    record.id,
                    hashlib.md5(str(record.seq).upper().encode()).hexdigest(),
                    record_count
                ]
            )

            record_count += 1

    return (
        read_fai(str(ref_fa) + '.fai', cols=None)
        .join(
            pl.DataFrame(
                record_list,
                orient='row',
                schema=(('chrom', pl.String), ('md5', pl.String), ('order', pl.Int32))
            ),
            on='chrom', how='left'
        )
    )
