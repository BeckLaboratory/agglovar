"""Tests for the VCF percent-encoding codec shared by the read and write paths."""

from __future__ import annotations

import pytest

from agglovar.vcf import (
    VCF_PERCENT_ESCAPES,
    vcf_percent_decode,
    vcf_percent_encode,
)


@pytest.mark.parametrize('escape,char', list(VCF_PERCENT_ESCAPES.items()))
def test_every_escape_round_trips(escape: str, char: str) -> None:
    """Each escape the specification defines decodes to its character and back."""
    assert vcf_percent_decode(f'a{escape}b') == f'a{char}b'
    assert vcf_percent_encode(f'a{char}b') == f'a{escape}b'


def test_decode_accepts_lower_case_hex() -> None:
    """A writer emitting lower-case hex is read the same as the canonical form."""
    assert vcf_percent_decode('%3b') == ';'


def test_decode_leaves_an_undefined_escape_alone() -> None:
    """Only the eight defined sequences decode, unlike a general percent-decoder."""
    assert vcf_percent_decode('5%AF') == '5%AF'
    assert vcf_percent_decode('100%') == '100%'
    assert vcf_percent_decode('%') == '%'


def test_decode_does_not_rescan_its_own_output() -> None:
    """An encoded escape survives one round trip rather than decoding twice."""
    assert vcf_percent_decode('%253B') == '%3B'
    assert vcf_percent_encode('%3B') == '%253B'


def test_encode_then_decode_restores_the_value() -> None:
    """Every reserved character survives a write followed by a read."""
    value = 'DEL[1,234]:INV[5,678] a;b=c%d\te\nf\rg'

    assert vcf_percent_decode(vcf_percent_encode(value)) == value


def test_a_value_without_a_percent_is_returned_unchanged() -> None:
    """The fast path returns the same object rather than rebuilding it."""
    value = 'ACGTACGT'

    assert vcf_percent_decode(value) is value
