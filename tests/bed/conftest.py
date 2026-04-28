"""Shared fixtures for the bed test suite.

Tests are parametrized over every dataset discovered under
``local/test_data/bed/``. The auto-generated ``synthetic1`` dataset is
materialized on first run; additional datasets (real-data extracts, etc.)
can be added by dropping ``a.parquet`` and ``b.parquet`` into a new
subdirectory.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.bed import datasets as bed_datasets
from tests.bed.datasets import BedDataset


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used in the bed suite."""
    config.addinivalue_line(
        'markers',
        'requires_test_data: skip if local/test_data/bed/<dataset> is unavailable',
    )


def _all_datasets() -> list[BedDataset]:
    """Discover all bed datasets, with auto-generated ones built on demand."""
    return bed_datasets.discover()


@pytest.fixture(scope='session')
def bed_data_root() -> Path:
    """Filesystem root for bed test data."""
    return bed_datasets.BED_DATA_ROOT


@pytest.fixture(
    scope='session',
    params=_all_datasets(),
    ids=lambda ds: ds.name,
)
def bed_dataset(request: pytest.FixtureRequest) -> BedDataset:
    """One bed dataset (parametrized over every dataset on disk)."""
    return request.param


@pytest.fixture(scope='session')
def df_a(bed_dataset: BedDataset) -> pl.DataFrame:
    """Table A for the current dataset."""
    return bed_dataset.load_a()


@pytest.fixture(scope='session')
def df_b(bed_dataset: BedDataset) -> pl.DataFrame:
    """Table B for the current dataset."""
    return bed_dataset.load_b()


@pytest.fixture(scope='session')
def synthetic1() -> BedDataset:
    """Direct accessor for the synthetic1 dataset (auto-built if missing)."""
    ds = bed_datasets.get('synthetic1')
    if ds is None:
        pytest.skip('synthetic1 dataset could not be created')
    return ds
