"""Discover and load BED test datasets.

Datasets live under ``local/test_data/bed/<name>/`` and consist of at least
``a.parquet`` and ``b.parquet`` written with columns ``chrom``, ``pos``,
``end`` and (optionally) ``id``. The ``synthetic1`` dataset is generated
deterministically by :mod:`tests.bed._make_synthetic1`; additional datasets
(real-data extracts, etc.) can be dropped into the same root and will be
picked up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BED_DATA_ROOT = PROJECT_ROOT / 'local' / 'test_data' / 'bed'

# Datasets known to be auto-generated. The harness will create them on demand
# when missing.
AUTO_GENERATED = {
    'synthetic1': 'tests.bed._make_synthetic1:write_parquet',
}


@dataclass(frozen=True)
class BedDataset:
    """One named dataset of A and B BED tables.

    :ivar name: Dataset name (matches the directory name).
    :ivar root: Filesystem root.
    :ivar a_path: Path to ``a.parquet``.
    :ivar b_path: Path to ``b.parquet``.
    """

    name: str
    root: Path
    a_path: Path
    b_path: Path

    def load_a(self) -> pl.DataFrame:
        """Load table A."""
        return pl.read_parquet(self.a_path)

    def load_b(self) -> pl.DataFrame:
        """Load table B."""
        return pl.read_parquet(self.b_path)

    def load(self, side: str) -> pl.DataFrame:
        """Load by side ``"a"`` or ``"b"``."""
        if side == 'a':
            return self.load_a()
        if side == 'b':
            return self.load_b()
        raise ValueError(f'side must be "a" or "b", got {side!r}')


def _ensure_auto_generated(name: str) -> None:
    """If ``name`` is auto-generated and missing, build it now."""
    if name not in AUTO_GENERATED:
        return

    target = BED_DATA_ROOT / name
    if (target / 'a.parquet').is_file() and (target / 'b.parquet').is_file():
        return

    module_path, func_name = AUTO_GENERATED[name].split(':')

    import importlib

    module = importlib.import_module(module_path)
    getattr(module, func_name)(out_dir=target)


def discover() -> list[BedDataset]:
    """Discover all BED datasets under ``local/test_data/bed/``.

    Auto-generated datasets are built on demand.
    """
    # Materialize auto-generated datasets first so they show up in discovery.
    for name in AUTO_GENERATED:
        try:
            _ensure_auto_generated(name)
        except Exception:
            # Defer error reporting to the test that actually consumes the
            # dataset; here we just skip silently so unrelated tests still run.
            pass

    if not BED_DATA_ROOT.is_dir():
        return []

    datasets: list[BedDataset] = []
    for entry in sorted(BED_DATA_ROOT.iterdir()):
        if not entry.is_dir():
            continue

        a_path = entry / 'a.parquet'
        b_path = entry / 'b.parquet'

        if not (a_path.is_file() and b_path.is_file()):
            continue

        datasets.append(
            BedDataset(name=entry.name, root=entry, a_path=a_path, b_path=b_path)
        )

    return datasets


def get(name: str) -> Optional[BedDataset]:
    """Return the named dataset or ``None`` if absent."""
    _ensure_auto_generated(name)

    root = BED_DATA_ROOT / name
    a_path = root / 'a.parquet'
    b_path = root / 'b.parquet'

    if not (a_path.is_file() and b_path.is_file()):
        return None

    return BedDataset(name=name, root=root, a_path=a_path, b_path=b_path)
