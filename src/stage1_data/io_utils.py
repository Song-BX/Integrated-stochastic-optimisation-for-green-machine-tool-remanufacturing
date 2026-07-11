"""Low-level CSV and filesystem helpers for Stage 1 validation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


CSV_ENCODING = "utf-8-sig"


def csv_path(raw_dir: Path, file_name: str) -> Path:
    return raw_dir / file_name


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def iter_dict_rows(path: Path, columns: Optional[Sequence[str]] = None) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        reader = csv.DictReader(handle)
        if columns is None:
            yield from reader
            return
        for row in reader:
            yield {column: row.get(column, "") for column in columns}


def count_rows(path: Path) -> int:
    with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def try_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_key_set(path: Path, key_columns: Sequence[str]) -> Tuple[set[Tuple[str, ...]], int, int]:
    keys: set[Tuple[str, ...]] = set()
    blank_rows = 0
    total_rows = 0
    for row in iter_dict_rows(path, key_columns):
        total_rows += 1
        key = tuple(row.get(column, "") for column in key_columns)
        if any(value == "" for value in key):
            blank_rows += 1
        keys.add(key)
    return keys, total_rows, blank_rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

