"""Writers for the two required TSV outputs (exact contract)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids"


def _list_cell(ids: Iterable[str]) -> str:
    return ",".join(sorted(set(ids)))


def write_matching(
    path: Path | str,
    required: Sequence[str],
    preds: Mapping[str, Iterable[str]],
) -> None:
    lines = [MATCHING_HEADER]
    for rid in required:
        lines.append(f"{rid}\t{_list_cell(preds.get(rid, ()))}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_candidates(
    path: Path | str,
    required: Sequence[str],
    cands: Mapping[str, Iterable[str]],
) -> None:
    lines = [CANDIDATE_HEADER]
    for rid in required:
        lines.append(f"{rid}\t{_list_cell(cands.get(rid, ()))}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    matching_path: Path | str,
    candidate_path: Path | str,
    required: Sequence[str],
    preds: Mapping[str, Iterable[str]],
    cands: Mapping[str, Iterable[str]],
) -> None:
    """Write both files: every required row once, sorted IDs, empty allowed."""
    write_matching(matching_path, required, preds)
    write_candidates(candidate_path, required, cands)


def write_scored_outputs(
    output_dir: Path | str,
    ref_ids: Sequence[str],
    s2_ids: Sequence[str],
    s3_ids: Sequence[str],
    ref_ord: np.ndarray,
    tgt_ord: np.ndarray,
    source: np.ndarray,
    selected: np.ndarray,
) -> int:
    """Write scored edges, preserving the target identity through ref sorting.

    The candidate output is the complete set of scored edges, not just the
    selected matches. Returns the number of references with a match.
    """
    if not (len(ref_ord) == len(tgt_ord) == len(source) == len(selected)):
        raise ValueError("scored edge arrays must be aligned")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(ref_ord, kind="stable")
    refs = ref_ord[order]
    tgts = tgt_ord[order]
    sources = source[order]
    kept = selected[order]
    starts = np.flatnonzero(np.r_[True, refs[1:] != refs[:-1]]) if len(refs) else np.array([], dtype=int)
    ends = np.r_[starts[1:], len(refs)]

    match_lines = 0
    group = 0
    with (output_dir / "matching_results.tsv").open("w", encoding="utf-8") as fm, \
         (output_dir / "candidate_pairs.tsv").open("w", encoding="utf-8") as fc:
        fm.write(MATCHING_HEADER + "\n")
        fc.write(CANDIDATE_HEADER + "\n")
        for r, rid in enumerate(ref_ids):
            cand: set[str] = set()
            matches: set[str] = set()
            if group < len(starts) and refs[starts[group]] == r:
                for j in range(starts[group], ends[group]):
                    source_id = int(sources[j])
                    if source_id not in (2, 3):
                        raise ValueError(f"unexpected target source {source_id}")
                    target = (s2_ids if source_id == 2 else s3_ids)[int(tgts[j])]
                    cand.add(target)
                    if kept[j]:
                        matches.add(target)
                group += 1
            fc.write(f"{rid}\t{','.join(sorted(cand))}\n")
            fm.write(f"{rid}\t{','.join(sorted(matches))}\n")
            match_lines += bool(matches)
    if group != len(starts):
        raise ValueError("scored edge has an unknown reference ordinal")
    return match_lines
