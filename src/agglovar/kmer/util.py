"""
Basic k-mer manipulation utilities.
"""

import math
from typing import Self, Optional, Iterable, Iterator

import numpy as np

# Integer to base
INT_TO_BASE: list[str] = ['A', 'C', 'G', 'T']
"""Maps 2-bit integer to base string."""

# Base to integer
BASE_TO_INT: dict[str, int] = {
    'A': 0x0,
    'C': 0x1,
    'G': 0x2,
    'T': 0x3,
    'a': 0x0,
    'c': 0x1,
    'g': 0x2,
    't': 0x3
}
"""Maps base string to 2-bit integer."""

BYTE_SIZE_TO_NUMPY_UINT: dict[int, np.integer] = {
    1: np.uint8,
    2: np.uint16,
    3: np.uint16,
    4: np.uint32,
    5: np.uint32,
    6: np.uint32,
    7: np.uint32,
    8: np.uint64,
}
"""
Maps k-mer byte size the smallest numpy integer type that can store it. Only defined for sizes up to the maximum
numpy unsigned numpy size (i.e. 8: np.uint64).
"""

NP_MAX_BYTE_SIZE = max(BYTE_SIZE_TO_NUMPY_UINT.keys())
"""Maximum k-mer byte size for numpy arrays."""

NP_MAX_BIT_SIZE = NP_MAX_BYTE_SIZE * 8
"""Maximum k-mer bit size for numpy arrays."""

NP_MAX_KMER_SIZE = NP_MAX_BIT_SIZE // 2
"""Maximum k-mer size for numpy arrays."""

class KmerUtil:
    k_size: int
    k_bit_size: int
    k_byte_size: int
    k_min_size: int
    k_min_mask: int
    k_mask: int
    min_kmer_util: Optional[Self]
    minimizer_mask: int
    sub_per_kmer: Optional[int]
    np_int_type: Optional[np.integer]
    """
    Manages basic k-mer functions, such as converting formats, appending bases, and reverse-complementing.
    Contains a set of constants specific to the k-mer size and minimizer (if defined).
    
    Attributes:
        k_size: K-mer size.
        k_bit_size: Number of bits in a k-mer. Minimum size of unsigned integers storing k-mers.
        k_min_size: Minimizer size or <code>0</code> if a minimizer is not used.
        k_min_mask: Minimizer mask if set and <code>kMinSize</code> is not <code>0</code>.
        k_mask: Mask for k-mer part of integer. Masks out unused bits if an integer has more bits than `k_bit_size`.
        k_bytes: Bytes per k-mer.
        min_kmer_util: K-mer util for minimizer. `None` if a minimizer is not defined.
        minimizer_mask: Mask for extracting minimizers from k-mers (minimizer-mask). `0` if a minimizer is not defined.
        sub_per_kmer: Number of sub-kmers per k-mer (sub_per_kmer). `None` if a minimizer is not defined.
        np_int_type: Smallest numpy unsigned integer type that can store k-mers of k_size. `None` if k_size is is
            larger than the maximum numpy unsigned integer size (i.e. 8: np.uint64).
    """

    def __init__(
            self,
            k_size: int,
            k_min_size: int = 0,
            k_min_mask: int = 0
    ) -> None:

        if k_min_mask != 0:
            raise NotImplementedError('Non-zero minimizer mask is not yet supported')

        if k_size < 0:
            raise ValueError(f'K-mer size must be positive: {k_size}')

        # Size of k-mers this utility works with.
        self.k_size = int(k_size)

        # Size of k-mers in bits
        self.k_bit_size = self.k_size * 2

        # Bytes per k-mer
        self.k_byte_size = math.ceil(self.k_bit_size / 8)

        # Minimizer size or <code>0</code> if a minimizer is not used.
        self.k_min_size = k_min_size

        # Minimizer mask if set and <code>kMinSize</code> is not <code>0</code>.
        self.k_min_mask = k_min_mask

        # Mask for k-mer part of integer.
        self.k_mask = ~(~0 << self.k_bit_size)

        # Mask for extracting minimizers from k-mers (minimizer-mask)
        # Number of sub-kmers per k-mer (sub_per_kmer)
        if self.k_min_size > 0:
            self.min_kmer_util = KmerUtil(k_min_size, 0)
            self.sub_per_kmer = self.k_size - self.k_min_size + 1
        else:
            self.min_kmer_util = None
            self.minimizer_mask = 0
            self.sub_per_kmer = None

        # Numpy integer type
        self.np_int_type = BYTE_SIZE_TO_NUMPY_UINT.get(self.k_byte_size, None)

    def append(self,
        kmer: int,
        base: str
    ) -> int:
        """
        Shift k-mer one base and append a new base.

        :param kmer: Old k-mer.
        :param base: Base to be appended.
        :return: New k-mer with appended base.
        """
        return ((kmer << 2) | BASE_TO_INT[base]) & self.k_mask

    def to_string(self,
        kmer: int
    ) -> str:
        """
        Translate integer k-mer to a string.

        :param kmer: Integer k-mer.

        :return: String representation of `kmer`.
        """

        mask = self.k_mask
        shift = self.k_bit_size - 2

        kmer_string = ['X'] * self.k_size

        for index in range(self.k_size):
            kmer_string[index] = INT_TO_BASE[(kmer & mask) >> shift]
            shift -= 2
            mask >>= 2

        return ''.join(kmer_string)

    def to_kmer(self,
        k_str: str
    ) -> int:
        """
        Convert a string to a k-mer.

        :param k_str: K-mer string.

        :return: K-mer integer.
        """

        # Check arguments
        if len(k_str) != self.k_size:
            raise RuntimeError(
                'Cannot convert string to k-mer ({}-mer): String length does not match k-mer size: '.format(
                    self.k_size, len(k_str)
                )
            )

        # Convert
        kmer = 0

        for index in range(self.k_size):
            kmer |= BASE_TO_INT[k_str[index]]
            kmer <<= 2

        return kmer >> 2

    def rev_complement(self,
        kmer: int
    ) -> int:
        """
        Reverse-complement k-mer.

        :param kmer: K-mer.

        :return: Reverse-complement of `kmer`.
        """
        rev_kmer = 0

        for index in range(self.k_size):
            rev_kmer |= (kmer & 0x3) ^ 0x3
            rev_kmer <<= 2
            kmer >>= 2

        return rev_kmer >> 2

    def rev_complement_array(
            self,
            kmer_arr: np.ndarray
    ) -> np.ndarray:
        """
        Reverse-complement k-mers in an array.

        Args:
            kmer_arr: K-mer array.

        Returns:
            Reverse-complemented k-mers.
        """

        kmer_arr = kmer_arr.copy()
        rev_kmer_arr = np.empty_like(kmer_arr)

        for index in range(self.k_size):
            rev_kmer_arr = rev_kmer_arr << 2 | (kmer_arr & 0x3) ^ 0x3
            kmer_arr >>= 2

        return rev_kmer_arr & self.k_mask

    def canonical_complement(self,
        kmer: int
    ) -> int:
        """
        Get the canonical k-mer of a k-mer. The canonical k-mer is the lesser of the k-mer and its reverse-complement.

        :param kmer: K-mer.

        :return: `kmer` if it is less than the reverse-complement, and the reverse-complement otherwise.
        """
        kmer_rev = self.rev_complement(kmer)

        if kmer_rev < kmer:
            return kmer_rev

        return kmer

    def minimizer(self,
        kmer: int
    ) -> int:
        """
        Get the minimizer of a k-mer. This function can only be used if a minimizer size is defined.

        The minimizer of a k-mer is the lesser of all sub-k-mers and their reverse-complements.

        :param kmer: K-mer.

        :return: Minimizer.
        """

        if self.k_min_size == 0:
            raise RuntimeError('Cannot get minimizer: No minimizer size set')

        min_kmer = kmer & self.min_kmer_util.k_mask

        next_rev_kmer = self.min_kmer_util.rev_complement(min_kmer)

        if next_rev_kmer < min_kmer:
            min_kmer = next_rev_kmer

        for index in range(self.sub_per_kmer - 1):
            kmer >>= 2

            next_kmer = kmer & self.min_kmer_util.k_mask
            next_rev_kmer = self.min_kmer_util.rev_complement(next_kmer)

            if next_kmer < min_kmer:
                min_kmer = next_kmer

            if next_rev_kmer < min_kmer:
                min_kmer = next_rev_kmer

        return min_kmer


def stream(
    seq: str,
    kutil: KmerUtil
) -> Iterator[int]:
    """
    Get an iterator for k-mers in a sequence.

    Args:
    :param seq: String sequence of bases.
    :param kutil: K-mer util describing parameters (e.g. k-mer size).

    :return: K-mer iterator.
    """

    # Prepare sequence
    kmer = 0
    load = 1
    k_size = kutil.k_size

    # Iterate and yield
    for base in seq:

        if base in {'A', 'C', 'G', 'T', 'a', 'c', 'g', 't'}:

            kmer = kutil.append(kmer, base)

            if load == k_size:
                yield kmer
            else:
                load += 1

        else:
            load = 1

def stream_index(
    seq: str,
    kutil: KmerUtil,
) -> Iterator[tuple[int, int]]:
    """
    Get an iterator for k-mers in a sequence with their index.

    :param seq: String sequence of bases.
    :param kutil: K-mer util describing parameters (e.g. k-mer size).

    :return: Iterator of tuples (kmer, index). Index starts at 0 and increments for each k-mer.
    """

    # Prepare sequence
    kmer = 0
    load = 1
    kmer_index = -kutil.k_size
    k_size = kutil.k_size

    # Iterate and yield
    for base in seq:

        kmer_index += 1

        if base in {'A', 'C', 'G', 'T', 'a', 'c', 'g', 't'}:

            kmer = kutil.append(kmer, base)

            if load == k_size:
                yield kmer, kmer_index
            else:
                load += 1

        else:
            load = 1
