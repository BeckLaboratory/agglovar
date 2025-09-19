"""Constants and functions for working with alignment operations.

Definitions in this subpackage are borrowed from
`PAV 3+ <https://github.com/BeckLaboratory/pav>`__.
"""

__all__ = [
    # Sets of valid characters/codes
    'INT_STR_SET',
    'CIGAR_OP_SET',

    # CIGAR operation codes
    'M',
    'I',
    'D',
    'N',
    'S',
    'H',
    'P',
    'EQ',
    'X',

    # Operation code sets
    'CLIP_SET',
    'ALIGN_SET',
    'EQX_SET',

    # Mappings
    'OP_CHAR',
    'OP_CHAR_FUNC',
    'OP_CODE',

    # Arrays
    'CONSUMES_QRY_ARR',
    'CONSUMES_REF_ARR',
    'ADV_REF_ARR',
    'ADV_QRY_ARR',
    'VAR_ARR',

    # Functions
    'cigar_to_arr',
]


import numpy as np
from types import MappingProxyType
from typing import Mapping

INT_STR_SET: frozenset[str] = frozenset({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', })
"""Set of valid integer character strings representing operation codes."""

CIGAR_OP_SET: frozenset[str] = frozenset({'M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X', })
"""Set of valid operation characters."""

M: int = 0
"""Match or mismatch operation code."""

I: int = 1  # noqa: E741
"""Insertion operation code."""

D: int = 2
"""Deletion operation code."""

N: int = 3
"""Skipped region operation code."""

S: int = 4
"""Soft clipping operation code."""

H: int = 5
"""Hard clipping operation code."""

P: int = 6
"""Padding operation code."""

EQ: int = 7
"""Sequence match operation code."""

X: int = 8
"""Sequence mismatch operation code."""

CLIP_SET: frozenset[int] = frozenset({S, H, })
"""Set of clipping operation codes (soft and hard clipping)."""

ALIGN_SET: frozenset[int] = frozenset({M, EQ, X, })
"""Set of alignment operation codes (match, sequence match, mismatch)."""

EQX_SET: frozenset[int] = frozenset({EQ, X, })
"""Set of exact match/mismatch operation codes."""

OP_CHAR: Mapping[int, str] = MappingProxyType({
    M: 'M',
    I: 'I',
    D: 'D',
    N: 'N',
    S: 'S',
    H: 'H',
    P: 'P',
    EQ: '=',
    X: 'X'
})
"""Mapping from operation codes to their character representations."""

OP_CHAR_FUNC = np.vectorize(lambda val: OP_CHAR.get(val, '?'))
"""Vectorized function to convert operation codes to characters."""

OP_CODE: Mapping[str, int] = MappingProxyType({
    'M': M,
    'I': I,
    'D': D,
    'N': N,
    'S': S,
    'H': H,
    'P': P,
    '=': EQ,
    'X': X
})
"""Mapping from CIGAR characters to operation codes."""

CONSUMES_QRY_ARR: np.typing.NDArray[np.integer] = np.array([M, I, S, EQ, X])
"""Array of operation codes that consume query bases."""

CONSUMES_REF_ARR: np.typing.NDArray[np.integer] = np.array([M, D, N, EQ, X])
"""Array of operation codes that consume reference bases."""

ADV_REF_ARR: np.typing.NDArray[np.integer] = np.array([M, EQ, X, D])
"""Array of operation codes that advance the reference position."""

ADV_QRY_ARR: np.typing.NDArray[np.integer] = np.array([M, EQ, X, I, S, H])
"""Array of operation codes that advance the query position."""

VAR_ARR: np.typing.NDArray[np.integer] = np.array([X, I, D])
"""Array of operation codes that introduce variation."""


def cigar_to_arr(
        cigar_str: str
) -> np.ndarray:
    """Get a numpy array with two dimensions (dtype int).

    The first column is the operation codes, the second column is the operation lengths.

    :param cigar_str: CIGAR string.

    :returns: Array of operation codes and lengths (dtype int).

    :raises ValueError: If the CIGAR string is not properly formatted.
    """
    pos = 0
    max_pos = len(cigar_str)

    op_tuples = list()

    while pos < max_pos:

        len_pos = pos

        while cigar_str[len_pos] in INT_STR_SET:
            len_pos += 1

        if len_pos == pos:
            raise ValueError(f'Missing length in CIGAR string at index {pos}')

        if cigar_str[len_pos] not in CIGAR_OP_SET:
            raise ValueError(f'Unknown CIGAR operation {cigar_str[len_pos]}')

        op_tuples.append(
            (OP_CODE[cigar_str[len_pos]], int(cigar_str[pos:len_pos]))
        )

        pos = len_pos + 1

    return np.array(op_tuples)
