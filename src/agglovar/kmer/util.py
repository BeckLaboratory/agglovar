"""
Basic k-mer manipulation utilities.
"""

import typing

# Integer to base
INT_TO_BASE = ['A', 'C', 'G', 'T']

# Base to integer
BASE_TO_INT = {
    'A': 0x0,
    'C': 0x1,
    'G': 0x2,
    'T': 0x3,
    'a': 0x0,
    'c': 0x1,
    'g': 0x2,
    't': 0x3
}

class KmerUtil:
    k_size: int
    k_bit_size: int
    k_min_size: int
    k_min_mask: int
    k_mask: int
    word_size: int
    word_size_bytes: int
    min_kmer_util: typing.Self
    minimizer_mask: int

    """
    Manages basic k-mer functions, such as converting formats, appending bases, and reverse-complementing.
    Contains a set of constants specific to the k-mer size and minimizer (if defined).
    """

    def __init__(self, k_size, k_min_size=0, k_min_mask=0):

        if k_min_mask != 0:
            raise RuntimeError('Non-zero minimizer mask is not yet supported')

        # Size of k-mers this utility works with.
        self.k_size = int(k_size)

        # Size of k-mers in bits
        self.k_bit_size = self.k_size * 2

        # Minimizer size or <code>0</code> if a minimizer is not used.
        self.k_min_size = k_min_size

        # Minimizer mask if set and <code>kMinSize</code> is not <code>0</code>.
        self.k_min_mask = k_min_mask

        # Mask for k-mer part of integer.
        self.k_mask = ~(~0 << (self.k_size * 2))

        # 32-bit word size of k-mers
        self.word_size = ((self.k_size - 1) // 16) + 1

        # Bytes per k-mer
        self.word_size_bytes = (self.k_size - 1) // 4 + 1

        # Mask for extracting minimizers from k-mers (minimizer-mask)
        # Number of sub-kmers per k-mer (sub_per_kmer)
        if self.k_min_size > 0:
            self.min_kmer_util = KmerUtil(k_min_size, 0)
            self.sub_per_kmer = self.k_size - self.k_min_size + 1
        else:
            self.min_kmer_util = None
            self.minimizer_mask = 0

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
) -> typing.Iterator[int]:
    """
    Get an iterator for k-mers in a sequence.

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
) -> typing.Iterator[typing.Tuple[int, int]]:
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
