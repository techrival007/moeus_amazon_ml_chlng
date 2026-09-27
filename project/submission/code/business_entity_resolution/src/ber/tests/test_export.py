"""Output writers and the strict release audit.

Contract from master.md sections 3.3, 17.2-17.3:
- Exact headers, tab-separated, comma-joined ID lists, no quoting; every
  required S1 row present exactly once, empty string for empty sets; IDs
  within lists sorted deterministically.
- The strict audit fails on: missing candidate file, missing/duplicate/extra
  S1 rows, unknown or wrong-prefix or repeated target IDs, matches not a
  subset of candidates, and candidate support inconsistent with the ledger.
- Parity: outputs that pass our strict audit must also pass the official
  utils/validate_submission.py (with --check-ids).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber.audit import audit_outputs, audit_support
from ber.export import write_outputs

REPO_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_VALIDATOR = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"


def _required():
    return ["S1-00001", "S1-00002", "S1-00003"]


def _valid_targets():
    return {"S2-00047", "S2-00193", "S3-00812", "S3-00999", "S3-00004"}


def _good_preds():
    return {
        "S1-00001": ["S2-00047", "S3-00812"],  # unsorted on purpose
        "S1-00002": ["S3-00004"],
        "S1-00003": [],
    }


def _good_cands():
    return {
        "S1-00001": ["S2-00193", "S2-00047", "S3-00812", "S3-00999"],
        "S1-00002": ["S3-00004"],
        "S1-00003": [],
    }


@pytest.fixture
def written(tmp_path):
    m = tmp_path / "matching_results.tsv"
    c = tmp_path / "candidate_pairs.tsv"
    write_outputs(m, c, _required(), _good_preds(), _good_cands())
    return m, c


class TestWriteOutputs:
    def test_exact_bytes(self, written):
        m, c = written
        want_m = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-00001\tS2-00047,S3-00812\n"
            "S1-00002\tS3-00004\n"
            "S1-00003\t\n"
        )
        want_c = (
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-00001\tS2-00047,S2-00193,S3-00812,S3-00999\n"
            "S1-00002\tS3-00004\n"
            "S1-00003\t\n"
        )
        assert m.read_text(encoding="utf-8") == want_m
        assert c.read_text(encoding="utf-8") == want_c

    def test_ids_sorted_within_lists(self, written):
        m, _ = written
        line = m.read_text(encoding="utf-8").splitlines()[1]
        assert line == "S1-00001\tS2-00047,S3-00812"

    def test_every_required_row_present_once(self, written):
        m, c = written
        for path in (m, c):
            rows = path.read_text(encoding="utf-8").splitlines()[1:]
            ids = [r.split("\t")[0] for r in rows if r.strip()]
            assert sorted(ids) == sorted(_required())

    def test_missing_pred_row_still_written(self, tmp_path):
        m = tmp_path / "m.tsv"
        c = tmp_path / "c.tsv"
        write_outputs(m, c, _required(), {"S1-00001": ["S2-00047"]}, _good_cands())
        text = m.read_text(encoding="utf-8")
        assert "S1-00002\t\n" in text and "S1-00003\t\n" in text

    def test_scored_edges_keep_target_ids_attached_when_references_are_sorted(self, tmp_path):
        from ber.export import write_scored_outputs

        write_scored_outputs(
            tmp_path,
            ["S1-zero", "S1-one", "S1-two", "S1-empty"],
            ["S2-a", "S2-b", "S2-c"], ["S3-a", "S3-b"],
            np.array([2, 0, 2, 1, 0], dtype=np.int64),
            np.array([1, 0, 0, 0, 2], dtype=np.int64),
            np.array([3, 2, 2, 3, 2], dtype=np.uint8),
            np.array([True, True, False, False, True]),
        )
        assert (tmp_path / "candidate_pairs.tsv").read_text() == (
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-zero\tS2-a,S2-c\n"
            "S1-one\tS3-a\n"
            "S1-two\tS2-a,S3-b\n"
            "S1-empty\t\n"
        )
        assert (tmp_path / "matching_results.tsv").read_text() == (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-zero\tS2-a,S2-c\n"
            "S1-one\t\n"
            "S1-two\tS3-b\n"
            "S1-empty\t\n"
        )


class TestAuditOutputs:
    def test_valid_outputs_pass(self, written):
        m, c = written
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert failures == []

    def test_missing_candidate_file_fails(self, written, tmp_path):
        m, c = written
        c.unlink()
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert any("candidate" in f.lower() for f in failures)

    def test_missing_s1_row_fails(self, written):
        m, c = written
        lines = m.read_text(encoding="utf-8").splitlines()
        m.write_text("\n".join(lines[:2] + lines[3:]) + "\n", encoding="utf-8")
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert failures

    def test_duplicate_s1_row_fails(self, written):
        m, c = written
        text = m.read_text(encoding="utf-8")
        m.write_text(text + "S1-00001\tS2-00047\n", encoding="utf-8")
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert any("duplicate" in f.lower() for f in failures)

    def test_extra_s1_row_fails(self, written):
        m, c = written
        text = m.read_text(encoding="utf-8")
        m.write_text(text + "S1-99999\t\n", encoding="utf-8")
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert any("unknown" in f.lower() or "not in" in f.lower() for f in failures)

    def test_unknown_target_id_fails(self, written):
        m, c = written
        text = m.read_text(encoding="utf-8")
        m.write_text(text.replace("S2-00047,S3-00812", "S2-00047,S2-88888"), encoding="utf-8")
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert any("S2-88888" in f for f in failures)

    def test_self_match_fails(self, written):
        m, c = written
        text = m.read_text(encoding="utf-8")
        m.write_text(text.replace("S3-00004", "S1-00003"), encoding="utf-8")
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert any("s1" in f.lower() or "self" in f.lower() or "prefix" in f.lower() for f in failures)

    def test_duplicate_id_in_list_fails(self, written):
        m, c = written
        text = m.read_text(encoding="utf-8")
        m.write_text(text.replace("S3-00004", "S3-00004,S3-00004"), encoding="utf-8")
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert any("duplicate" in f.lower() for f in failures)

    def test_match_not_in_candidates_fails(self, written):
        m, c = written
        text = m.read_text(encoding="utf-8")
        # add a matched ID that is not in the candidate list
        m.write_text(text.replace("S3-00004", "S3-00004,S2-00193"), encoding="utf-8")
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert any("subset" in f.lower() or "candidate" in f.lower() for f in failures)

    def test_candidate_file_with_repeated_id_fails(self, written):
        m, c = written
        text = c.read_text(encoding="utf-8")
        c.write_text(text.replace("S2-00047,S2-00193", "S2-00047,S2-00047,S2-00193"), encoding="utf-8")
        failures = audit_outputs(m, c, set(_required()), _valid_targets())
        assert any("duplicate" in f.lower() for f in failures)


class TestAuditSupport:
    def test_support_ledger_digest_match(self, written):
        _, c = written
        # ledger as (ref, tgt) edge multiset derived from the same file
        edges = []
        for line in c.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            ref, rest = line.split("\t")
            for t in (rest.split(",") if rest.strip() else []):
                edges.append((ref, t))
        assert audit_support(c, edges) == []

    def test_support_ledger_mismatch_fails(self, written):
        _, c = written
        assert audit_support(c, [("S1-00001", "S2-00047")])  # missing edges

    def test_exact_scored_edge_ledger_rejects_plausible_but_swapped_targets(self, tmp_path):
        from ber.audit import audit_scored_support

        features = tmp_path / "features" / "US"
        features.mkdir(parents=True)
        pq.write_table(pa.table({"ref_ord": [0, 1], "source": [2, 3],
                                 "tgt_ord": [0, 0]}), features / "src2_0000.parquet")
        c = tmp_path / "candidate_pairs.tsv"
        c.write_text("source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-a\nS1-b\tS3-a\n")
        assert audit_scored_support(c, features.parent, ["S1-a", "S1-b"],
                                    ["S2-a", "S2-b"], ["S3-a", "S3-b"]) == []
        c.write_text("source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-b\nS1-b\tS3-a\n")
        assert audit_scored_support(c, features.parent, ["S1-a", "S1-b"],
                                    ["S2-a", "S2-b"], ["S3-a", "S3-b"])


class TestOfficialValidatorParity:
    def test_official_validator_passes_our_outputs(self, written, tmp_path):
        if not OFFICIAL_VALIDATOR.is_file():
            pytest.skip("official validator not present")
        m, c = written
        test_dir = tmp_path / "dataset" / "test"
        test_dir.mkdir(parents=True)
        (test_dir / "test_source1.tsv").write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            + "".join(f"{rid}\tName {i}\tAddr {i}\tUS\n" for i, rid in enumerate(_required())),
            encoding="utf-8",
        )
        for source, ids in (
            ("test_source2.tsv", ["S2-00047", "S2-00193"]),
            ("test_source3.tsv", ["S3-00812", "S3-00999", "S3-00004"]),
        ):
            (test_dir / source).write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                + "".join(f"{tid}\tTName {i}\tTAddr {i}\tUS\n" for i, tid in enumerate(ids)),
                encoding="utf-8",
            )
        proc = subprocess.run(
            [sys.executable, str(OFFICIAL_VALIDATOR),
             "--matching", str(m), "--candidate", str(c),
             "--test-dir", str(test_dir), "--check-ids"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "PASS" in proc.stdout
