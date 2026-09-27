"""Record ingestion: streaming parse, schema validation, ordinal mapping,
and compact parquet record tables with raw-preserving views."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from ber.normalize import fold_view, is_indic, normalize_text

EXPECTED_HEADER = ["entity_id", "business_name", "business_address", "country"]
TRUTH_HEADER = ["source1_entity_id", "matched_entity_ids"]

SCHEMA = pa.schema([
    ("ordinal", pa.int64()),
    ("entity_id", pa.string()),
    ("name_raw", pa.string()),
    ("addr_raw", pa.string()),
    ("name_norm", pa.string()),
    ("addr_norm", pa.string()),
    ("name_fold", pa.string()),
    ("addr_fold", pa.string()),
    ("country", pa.string()),
    ("name_indic", pa.bool_()),
    ("addr_indic", pa.bool_()),
    ("addr_missing", pa.bool_()),
])


@dataclass(frozen=True)
class Record:
    entity_id: str
    business_name: str
    business_address: str
    country: str


@dataclass(frozen=True)
class RecordViews:
    entity_id: str
    name_raw: str
    addr_raw: str
    name_norm: str
    addr_norm: str
    name_fold: str
    addr_fold: str
    country: str
    name_indic: bool
    addr_indic: bool
    addr_missing: bool


def sha256_file(path: Path | str, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stream_records(path: Path | str) -> Iterator[Record]:
    """Yield validated Record rows from a source TSV (explicit tab separator)."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty file")
        if [c.strip().lower() for c in header] != EXPECTED_HEADER:
            if len(header) == 1 and "," in header[0]:
                raise ValueError(f"{path}: header has no TAB but contains commas — file looks comma-separated")
            raise ValueError(f"{path}: unexpected header {header!r}; expected {EXPECTED_HEADER}")
        for lineno, row in enumerate(reader, start=2):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != 4:
                raise ValueError(f"{path}:{lineno}: expected 4 columns, got {len(row)}: {row!r}")
            rid, name, addr, country = row
            yield Record(rid, name, addr, country)


def build_views(entity_id: str, name: str, addr: str, country: str) -> RecordViews:
    """Raw-preserving normalized views + script/missingness flags."""
    name_norm = normalize_text(name)
    addr_norm = normalize_text(addr)
    return RecordViews(
        entity_id=entity_id,
        name_raw=name,
        addr_raw=addr,
        name_norm=name_norm,
        addr_norm=addr_norm,
        name_fold=fold_view(name_norm),
        addr_fold=fold_view(addr_norm),
        country=country,
        name_indic=is_indic(name),
        addr_indic=is_indic(addr),
        addr_missing=not addr.strip(),
    )


def build_source_parquet(src: Path | str, dst: Path | str, source_tag: str) -> tuple[int, int]:
    """Stream a source TSV into the canonical parquet record table.

    Returns (n_rows, n_duplicate_ids). Duplicate entity_ids are fatal.
    Ordinals are assigned in first-appearance order.
    """
    seen: set[str] = set()
    batch: dict[str, list] = {f.name: [] for f in SCHEMA}
    n = 0
    writer = pq.ParquetWriter(dst, SCHEMA, compression="zstd")
    try:
        for rec in stream_records(src):
            if not rec.entity_id.startswith(f"{source_tag}-"):
                raise ValueError(f"{src}: entity_id {rec.entity_id!r} does not match source tag {source_tag!r}")
            if rec.entity_id in seen:
                raise ValueError(f"{src}: duplicate entity_id {rec.entity_id!r}")
            seen.add(rec.entity_id)
            v = build_views(rec.entity_id, rec.business_name, rec.business_address, rec.country)
            batch["ordinal"].append(n)
            batch["entity_id"].append(v.entity_id)
            batch["name_raw"].append(v.name_raw)
            batch["addr_raw"].append(v.addr_raw)
            batch["name_norm"].append(v.name_norm)
            batch["addr_norm"].append(v.addr_norm)
            batch["name_fold"].append(v.name_fold)
            batch["addr_fold"].append(v.addr_fold)
            batch["country"].append(v.country)
            batch["name_indic"].append(v.name_indic)
            batch["addr_indic"].append(v.addr_indic)
            batch["addr_missing"].append(v.addr_missing)
            n += 1
            if n % 200_000 == 0:
                writer.write_table(pa.table(batch, schema=SCHEMA))
                batch = {f.name: [] for f in SCHEMA}
        if batch["ordinal"]:
            writer.write_table(pa.table(batch, schema=SCHEMA))
    finally:
        writer.close()
    return n, 0


def stream_truth(path: Path | str) -> Iterator[tuple[str, list[str]]]:
    """Yield (source1_entity_id, [matched ids]) rows; empty cell -> empty list.

    Duplicate S1 rows and duplicate IDs inside one list are fatal.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None or [c.strip().lower() for c in header] != TRUTH_HEADER:
            raise ValueError(f"{path}: unexpected ground-truth header {header!r}")
        seen: set[str] = set()
        for lineno, row in enumerate(reader, start=2):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != 2:
                raise ValueError(f"{path}:{lineno}: expected 2 columns, got {len(row)}")
            sid, ids_cell = row
            if sid in seen:
                raise ValueError(f"{path}:{lineno}: duplicate truth row for {sid!r}")
            seen.add(sid)
            ids = ids_cell.split(",") if ids_cell.strip() else []
            if len(ids) != len(set(ids)):
                raise ValueError(f"{path}:{lineno}: duplicate IDs inside truth list for {sid!r}")
            yield sid, ids


def build_truth_parquet(src: Path | str, dst: Path | str) -> int:
    """Write the immutable truth relation; returns the row count."""
    sids: list[str] = []
    lists: list[str] = []
    for sid, ids in stream_truth(src):
        sids.append(sid)
        lists.append(",".join(ids))
    schema = pa.schema([("source1_entity_id", pa.string()), ("matched_entity_ids", pa.string())])
    pq.write_table(pa.table({"source1_entity_id": sids, "matched_entity_ids": lists}, schema=schema), dst)
    return len(sids)


def iter_truth_parquet(path: Path | str, batch_size: int = 200_000):
    """Yield (source1_entity_id, [matched ids]) rows from the truth parquet.

    Empty cell -> empty list; preserves file order; no pandas.
    """
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["source1_entity_id", "matched_entity_ids"]):
        for sid, ids in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            yield sid, (ids.split(",") if ids else [])
