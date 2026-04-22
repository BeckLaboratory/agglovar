"""Routines for converting between VCF format and variant tables."""

from . import _vcf_const
from . import _vcf_header
from . import _vcf_read
from . import _vcf_write

__all__ = (
    getattr(_vcf_const, '__vcf_const_all__', [])
    + getattr(_vcf_header, '__vcf_header_all__', [])
    + getattr(_vcf_read, '__vcf_read_all__', [])
    + getattr(_vcf_write, '__vcf_write_all__', [])
)

from ._vcf_const import *
from ._vcf_header import *
from ._vcf_read import *
from ._vcf_write import *
