"""Strict release audit: every contract condition, even ones the official
helper only warns about (master.md 17.3)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def _read_rows(path: Path, expected_header: tuple[str, str]) -> list[tuple[str, list[str]]] | None:
    if not path.is_file():
        return None
    rows: list[tuple[str, list[str]]] = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        cols = tuple(c.strip().lower() for c in header.split("\t"))
        if cols != expected_header:
            return []
        for lineno, line in enumerate(f, start=2):
            if not line.strip():
                continue
            left, _, rest = line.rstrip("\n").partition("\t")
            ids = rest.split(",") if rest.strip() else []
            rows.append((left, ids))
    return rows


def audit_outputs(
    matching_path: Path | str,
    candidate_path: Path | str,
    required: Iterable[str],
    valid_target_ids: Iterable[str] | None,
) -> list[str]:
    """Return a list of contract failures; an empty list means PASS."""
    failures: list[str] = []
    matching_path = Path(matching_path)
    candidate_path = Path(candidate_path)
    required_list = list(required)
    required_set = set(required_list)

    m_rows = _read_rows(matching_path, MATCHING_HEADER)
    if m_rows is None:
        failures.append(f"missing file: {matching_path}")
        return failures
    if m_rows == []:
        failures.append(f"bad header in {matching_path}")
        return failures
    if not candidate_path.is_file():
        failures.append(f"missing file: {candidate_path}")
        return failures
    c_rows = _read_rows(candidate_path, CANDIDATE_HEADER)
    if c_rows is None or c_rows == []:
        failures.append(f"bad header or unreadable: {candidate_path}")
        return failures

    valid: set[str] | None = set(valid_target_ids) if valid_target_ids is not None else None

    for label, rows in (("matching", m_rows), ("candidate", c_rows)):
        seen: set[str] = set()
        dup_rows = [r for r, _ in rows if r in seen or seen.add(r)]
        if dup_rows:
            failures.append(f"{label}: duplicate S1 rows: {sorted(set(dup_rows))[:5]}")
        missing = required_set - seen
        if missing:
            failures.append(f"{label}: missing required S1 rows: {sorted(missing)[:5]}")
        extra = seen - required_set
        if extra:
            failures.append(f"{label}: rows for unknown S1 ids: {sorted(extra)[:5]}")
        for rid, ids in rows:
            counts = Counter(ids)
            dups = [i for i, n in counts.items() if n > 1]
            if dups:
                failures.append(f"{label}: duplicate ids inside list of {rid}: {sorted(dups)[:5]}")
            for i in ids:
                if i.startswith("S1-"):
                    failures.append(f"{label}: self-match S1 id in list of {rid}: {i}")
                elif not i.startswith(("S2-", "S3-")):
                    failures.append(f"{label}: id without S2-/S3- prefix in list of {rid}: {i}")
                elif valid is not None and i not in valid:
                    failures.append(f"{label}: id not present in test set in list of {rid}: {i}")

    m_map = {r: set(ids) for r, ids in m_rows}
    c_map = {r: set(ids) for r, ids in c_rows}
    for rid in required_set & set(m_map):
        unmatched = m_map[rid] - c_map.get(rid, set())
        if unmatched:
            failures.append(
                f"matching: selected ids absent from candidate list of {rid}: {sorted(unmatched)[:5]}"
            )
    return failures


def audit_support(
    candidate_path: Path | str,
    ledger_edges: Iterable[tuple[str, str]],
) -> list[str]:
    """Fail if candidate_pairs.tsv does not equal the support ledger."""
    candidate_path = Path(candidate_path)
    if not candidate_path.is_file():
        return [f"missing file: {candidate_path}"]
    c_rows = _read_rows(candidate_path, CANDIDATE_HEADER)
    if c_rows is None or c_rows == []:
        return [f"bad header or unreadable: {candidate_path}"]
    file_edges = sorted((r, t) for r, ids in c_rows for t in ids)
    ledger = sorted(set(ledger_edges))
    if file_edges != ledger:
        only_file = [e for e in file_edges if e not in set(ledger)][:5]
        only_ledger = [e for e in ledger if e not in set(file_edges)][:5]
        return [
            "candidate file does not equal the support ledger",
            f"edges only in file: {only_file}",
            f"edges only in ledger: {only_ledger}",
        ]
    return []


def audit_scored_support(
    candidate_path: Path | str,
    features_dir: Path | str,
    ref_ids: Sequence[str],
    s2_ids: Sequence[str],
    s3_ids: Sequence[str],
) -> list[str]:
    """Compare every exported candidate with the actual scored feature edge.

    Uses a compact uint64 key per pair rather than materializing tens of
    millions of Python (reference ID, target ID) tuples. IDs are mapped back
    to their ordinal only for this audit; their suffix is never interpreted.
    """
    paths = sorted(Path(features_dir).glob("*/src*.parquet"))
    if not paths:
        return ["no scored feature shards found"]
    count = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    scored = np.empty(count, dtype=np.uint64)
    offset = 0
    for path in paths:
        cols = pq.read_table(path, columns=["ref_ord", "source", "tgt_ord"])
        ref = cols.column("ref_ord").to_numpy().astype(np.uint64)
        src = cols.column("source").to_numpy().astype(np.uint64)
        tgt = cols.column("tgt_ord").to_numpy().astype(np.uint64)
        if (ref >= (1 << 38)).any() or (tgt >= (1 << 24)).any() or not np.isin(src, (2, 3)).all():
            return [f"scored feature shard has an invalid ordinal/source: {path}"]
        n = len(ref)
        scored[offset:offset + n] = (ref << 26) | (src << 24) | tgt
        offset += n
        del cols, ref, src, tgt
    scored.sort()

    s2_ord = {rid: ordinal for ordinal, rid in enumerate(s2_ids)}
    s3_ord = {rid: ordinal for ordinal, rid in enumerate(s3_ids)}
    exported = np.empty(count, dtype=np.uint64)
    offset = 0
    with Path(candidate_path).open(encoding="utf-8") as f:
        if f.readline().rstrip("\n") != "source1_entity_id\tcandidate_entity_ids":
            return ["bad candidate header"]
        for ref_ord, line in enumerate(f):
            if ref_ord >= len(ref_ids):
                return ["candidate file contains extra reference rows"]
            rid, tab, cell = line.rstrip("\r\n").partition("\t")
            if not tab or rid != ref_ids[ref_ord]:
                return [f"candidate reference row {ref_ord} is missing or out of order"]
            for tid in (cell.split(",") if cell else ()):
                if offset == count:
                    return ["candidate file contains more pairs than were scored"]
                source = 2 if tid.startswith("S2-") else 3
                ordinal = (s2_ord if source == 2 else s3_ord).get(tid)
                if ordinal is None:
                    return [f"candidate has an unknown target ID: {tid}"]
                exported[offset] = (ref_ord << 26) | (source << 24) | ordinal
                offset += 1
        if ref_ord + 1 != len(ref_ids):
            return ["candidate file is missing reference rows"]
    if offset != count:
        return [f"candidate file has {offset} pairs, but {count} pairs were scored"]
    exported.sort()
    if not np.array_equal(scored, exported):
        mismatch = int(np.flatnonzero(scored != exported)[0])
        return [f"candidate support differs from scored edges at sorted position {mismatch}"]
    return []
