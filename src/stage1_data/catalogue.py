"""Data catalogue scanning for raw CSV files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .io_utils import CSV_ENCODING, count_rows, read_header, sha256_file
from .schema_rules import EXPECTED_FILES


@dataclass
class CatalogueEntry:
    file_name: str
    exists: bool
    row_count: int
    column_count: int
    size_bytes: int
    size_mb: float
    encoding: str
    modified_time: str
    sha256: str
    first_columns: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def scan_catalogue(raw_dir: Path, expected_files: Sequence[str] = EXPECTED_FILES) -> List[CatalogueEntry]:
    entries: List[CatalogueEntry] = []
    for file_name in expected_files:
        path = raw_dir / file_name
        if not path.exists():
            entries.append(
                CatalogueEntry(
                    file_name=file_name,
                    exists=False,
                    row_count=0,
                    column_count=0,
                    size_bytes=0,
                    size_mb=0.0,
                    encoding=CSV_ENCODING,
                    modified_time="",
                    sha256="",
                    first_columns="",
                )
            )
            continue

        header = read_header(path)
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        entries.append(
            CatalogueEntry(
                file_name=file_name,
                exists=True,
                row_count=count_rows(path),
                column_count=len(header),
                size_bytes=stat.st_size,
                size_mb=round(stat.st_size / 1024 / 1024, 2),
                encoding=CSV_ENCODING,
                modified_time=modified,
                sha256=sha256_file(path),
                first_columns=";".join(header[:10]),
            )
        )
    return entries


def catalogue_totals(entries: Iterable[CatalogueEntry]) -> Dict[str, object]:
    entry_list = list(entries)
    return {
        "file_count": len(entry_list),
        "existing_file_count": sum(1 for entry in entry_list if entry.exists),
        "total_rows": sum(entry.row_count for entry in entry_list),
        "total_size_bytes": sum(entry.size_bytes for entry in entry_list),
        "total_size_mb": round(sum(entry.size_bytes for entry in entry_list) / 1024 / 1024, 2),
    }

