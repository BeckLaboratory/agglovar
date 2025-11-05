"""Fast Smith-waterman implementation."""

__all__ = [
    'MatchScoreModel',
]

import collections
from dataclasses import dataclass
import numpy as np
from typing import Optional

import edlib

from .align.score import (
    AffineScoreModel,
    AFFINE_SCORE_MATCH,
    AFFINE_SCORE_MISMATCH,
    AFFINE_SCORE_GAP,
)

from .align import op

# Op codes for trace matrix (will be removed in a future version)
_TRACE_OP_NONE = 0
_TRACE_OP_MATCH = 1
_TRACE_OP_MISMATCH = 2
_TRACE_OP_GAP_SUB = 3  # Gap subject, insertion
_TRACE_OP_GAP_QRY = 4  # Gap query, deletion

_TRACE_OP_CODE = ['*', '=', 'X', 'I', 'D']  # Indexed by OP_ constants


@dataclass(frozen=True)
class MatchScoreModel:
    """An engine for scoring alignments.

    Intended for determining sequence similarity through an alignment where the alignment itself is
    not important.

    :param match: Base match score (> 0.0).
    :param mismatch: Base mismatch score (< 0.0).
    :param gap_open: Gap open cost (<= 0.0).
    :param gap_extend: Gap extend cost (<= 0.0).
    :param map_limit: Maximum sequence size before falling back to Jaccard index in :func:`match_prop()`.
    :param jaccard_kmer: Jaccard k-mer size for comparisons falling back to Jaccard index in
        :func:`match_prop()`.
    """

    match: float = AFFINE_SCORE_MATCH
    mismatch: float = AFFINE_SCORE_MISMATCH
    affine_gap: float = AFFINE_SCORE_GAP
    map_limit: Optional[int] = None
    jaccard_kmer: int = 9
    rotate_min: int = 3
    score_model: AffineScoreModel = None

    def __post_init__(self) -> None:
        """Post-initialization."""
        object.__setattr__(
            self,
            'score_model',
            AffineScoreModel(
                match=self.match,
                mismatch=self.mismatch,
                affine_gap=self.affine_gap
            )
        )

        if self.map_limit is not None and self.map_limit < 0:
            raise ValueError(f'Map must be >= 0.0 or None (no limit): {self.map_limit}')

        if self.jaccard_kmer <= 0:
            raise ValueError(f'Jaccard k-mer size must be > 0: {self.jaccard_kmer}')

        if self.rotate_min <= 0:
            raise ValueError(f'Minimum rotation size must be > 0: {self.rotate_min}')

        return

    def score_align(
            self,
            seq_a: str,
            seq_b: str
    ) -> float:
        """Get max score aligning two sequences.

        Only returns the score, not the alignment.

        This is a legacy native Python implementation used by SV-Pop. It is accurate, but extremely slow.

        :param seq_a: Subject sequence.
        :param seq_b: Query sequence.


        :returns: Maximum alignment score.
        """
        align = edlib.align(seq_a.upper(), seq_b.upper(), mode='HW', task='path')

        return self.score_model.score_operations(
            op.cigar_to_arr(align['cigar'])
        )

    def score_align_native(
            self,
            seq_a: str,
            seq_b: str
    ) -> float:
        """Get max score aligning two sequences.

        Only returns the score, not the alignment.commands = [["pytest", "{posargs}"]]

        This is a legacy native Python implementation used by SV-Pop. It is validated, but extremely slow.

        :param seq_a: Subject sequence.
        :param seq_b: Query sequence.

        :returns: Maximum alignment score.
        """
        if len(self.score_model.score_affine_gap) != 1:
            raise RuntimeError(
                f'Legacy align only support single-affine gap models, found '
                f'{len(self.score_model.score_affine_gap)} affine segments'
            )

        score_match = self.score_model.score_match
        score_mismatch = self.score_model.score_mismatch
        score_gap_open = self.score_model.score_affine_gap[0][0]
        score_gap_extend = self.score_model.score_affine_gap[0][1]

        # Compute for convenience
        gap_1bp = score_gap_open + score_gap_extend

        # Scrub sequences
        seq_a = seq_a.upper().strip()
        seq_b = seq_b.upper().strip()

        # Get length
        len_a = len(seq_a) + 1
        len_b = len(seq_b) + 1

        trace_matrix = [_ScoreTraceNode() for _ in range(len_a)]
        trace_matrix_last = [_ScoreTraceNode() for _ in range(len_a)]

        # Max values
        global_max_score = 0.0  # Max score

        # Iterate bases in seq_b
        for j in range(1, len_b):

            # Swap trace arrays
            trace_matrix, trace_matrix_last = trace_matrix_last, trace_matrix

            # Iterate bases in seq_a
            for i in range(1, len_a):

                # Aligned bases
                if seq_a[i - 1] == seq_b[j - 1]:
                    score_max = trace_matrix_last[i - 1].score + score_match
                    op_code = _TRACE_OP_MATCH

                else:
                    score_max = trace_matrix_last[i - 1].score + score_mismatch
                    op_code = _TRACE_OP_MISMATCH

                # Gap subject (insertion)
                score_gap_sub = trace_matrix_last[i].score + (
                    score_gap_extend if trace_matrix_last[i].op_code == _TRACE_OP_GAP_SUB else gap_1bp
                )

                if score_gap_sub > score_max:
                    score_max = score_gap_sub
                    op_code = _TRACE_OP_GAP_SUB

                # Gap query (deletion)
                score_gap_qry = trace_matrix[i - 1].score + (
                    score_gap_extend if trace_matrix[i - 1].op_code == _TRACE_OP_GAP_QRY else gap_1bp
                )

                if score_gap_qry > score_max:
                    score_max = score_gap_qry
                    op_code = _TRACE_OP_GAP_SUB

                # Update trace matrix
                if score_max > 0:
                    trace_matrix[i].score = score_max
                    trace_matrix[i].op_code = op_code

                    if op_code == _TRACE_OP_MATCH:

                        # Check for new global max
                        if score_max >= global_max_score:
                            global_max_score = score_max

                else:
                    trace_matrix[i].op_code = _TRACE_OP_NONE
                    trace_matrix[i].score = 0

        return global_max_score

    def match_prop(
            self,
            seq_a: str,
            seq_b: str
    ) -> float:
        """Get the alignment score proportion over the max possible score between two sequences.

        To follow tandem duplications to map correctly, seq_b is duplicated head-to-tail
        (seq_b + seq_b) and seq_a is aligned to it.

        The max possible score is achieved if seq_a and seq_b are the same size and seq_a aligns to
        seq_b + seq_b with all seq_a bases matching (function returns 1.0).

        The numerator is the alignment score and the denominator is the size of the larger sequence times the match

        score::

            min(
                score_align(seq_a, seq_b + seq_b), min_len(seq_a, seq_b) * match
            ) / (
                max_len(seq_a, seq_b) * match
            )

        :param seq_a: Subject sequence.
        :param seq_b: Query sequence.

        :returns: Alignment proportion with seq_b duplicated head to tail. If either sequence is
            None or empty, returns 0.0
        """
        if seq_a is None or seq_b is None:
            return 0.0

        seq_a = seq_a.upper().strip()
        seq_b = seq_b.upper().strip()

        if len(seq_a) == 0 or len(seq_b) == 0:
            return 0.0

        if len(seq_a) > len(seq_b):
            seq_a, seq_b = seq_b, seq_a

        max_len = np.max([len(seq_a), len(seq_b)])
        min_len = np.min([len(seq_a), len(seq_b)])

        if min_len == 0:
            return 0.0

        if self.map_limit is None or max_len <= self.map_limit:

            if min_len >= self.rotate_min:
                # Align with rotation
                return np.clip(
                    np.min([
                        self.score_align(seq_a, seq_b + seq_b), min_len * self.score_model.score_match
                    ]) / (max_len * self.score_model.score_match),
                    0.0, 1.0
                )

            else:
                # Align without rotation
                return np.clip(
                    self.score_align(seq_a, seq_b) / (max_len * self.score_model.score_match),
                    0.0, 1.0
                )

        elif min_len > self.jaccard_kmer:
            return _jaccard_distance(seq_a, seq_b, self.jaccard_kmer)

        else:
            return 1 if seq_a.upper() == seq_b.upper() else 0


def _get_kmer_count(
    seq: str,
    k_size: int,
) -> collections.Counter:
    """Get a counter for all k-mers in a sequence.

    :param seq: Sequence.
    :param k_size: K-mer size.


    :returrs: Counter with k-mers as keys and counts as values.
    """
    counter = collections.Counter()

    seq = seq.strip().upper()

    for index in range(len(seq) - k_size + 1):
        counter[seq[index:index + k_size]] += 1

    return counter


def _jaccard_distance(
    seq_a: str,
    seq_b: str,
    k_size: int,
) -> float:
    """Get the Jaccard distance between k-merized sequences.

    This Jaccard distance is computed on the total number of k-mers including multiplicity
    (the same kmer may appear more than once). For example, if a k-mer is in A twice
    and B once, one instance of the k-mer matches and one does not match.

    :param seq_a: Sequence A (string).
    :param seq_b: Sequence B (string).
    :param k_size: K-mer size.

    :returns: Jaccard distance account for multiplicity.
    """
    count1 = _get_kmer_count(seq_a, k_size)
    count2 = _get_kmer_count(seq_b, k_size)

    key_set = set(count1.keys()) | set(count2.keys())

    return np.sum(
        [np.min([count1[key], count2[key]]) for key in key_set]  # Matching k-mers
    ) / np.sum(
        [np.max([count1[key], count2[key]]) for key in key_set]  # All k-mers
    ) if len(count1) > 0 and len(count2) > 0 else 0.0


@dataclass(frozen=True)
class _ScoreTraceNode:
    """Trace node for sequence alignment tracking."""

    op_code: int = _TRACE_OP_NONE
    score: float = 0.0
