"""Record ingestion: streaming TSV parse, schema validation, ordinal mapping,
and raw-preserving views written to compact parquet.

Contract from master.md sections 3.2, 8, 16.2, 21.3:
- Parse with explicit tab separator; preserve literal strings; empty values
  stay empty (never NaN); blank/whitespace-only address is allowed and
  flagged, not dropped.
- Exactly four columns per source row; wrong width is a fatal data error.
- Duplicate entity_id within one source file is a fatal data error.
- Ordinals are assigned in first-appearance order per source; the mapping to
  exact original IDs is stored and immutable.
- Ground truth: comma-separated list; empty cell -> empty list; duplicate IDs
  within a list are a fatal data error.
- File fingerprints: SHA-256 of raw bytes.
"""

import csv
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ber.records import (
    RecordViews,
    build_truth_parquet,
    build_views,
    sha256_file,
    stream_records,
    stream_truth,
)


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class TestStreamRecords:
    HDR = "entity_id\tbusiness_name\tbusiness_address\tcountry"

    def test_streams_all_rows_in_order(self, tmp_path):
        p = _write(tmp_path, "s.tsv", [
            self.HDR,
            "S1-1\tOrelee's Barbershop\t1795 Westchester Drive, High Point, NC\tUS",
            "S2-2\tराम मार्केटिंग प्राइवेट लिमिटेड\tKH NO. -570/13, NEW DELHI, Delhi\tIndia",
            "S3-3\tEmpty Addr\t\tUS",
        ])
        rows = list(stream_records(p))
        assert [r.entity_id for r in rows] == ["S1-1", "S2-2", "S3-3"]
        assert rows[1].business_name == "राम मार्केटिंग प्राइवेट लिमिटेड"
        assert rows[2].business_address == ""

    def test_wrong_column_count_is_fatal(self, tmp_path):
        p = _write(tmp_path, "bad.tsv", [self.HDR, "S1-1\tname\tUS"])
        with pytest.raises(ValueError, match="column"):
            list(stream_records(p))

    def test_missing_header_tab_is_fatal(self, tmp_path):
        p = _write(tmp_path, "csv.tsv", ["entity_id,business_name,business_address,country", "S1-1,a,b,US"])
        with pytest.raises(ValueError, match="header"):
            list(stream_records(p))

    def test_blank_line_is_skipped(self, tmp_path):
        p = _write(tmp_path, "blank.tsv", [self.HDR, "S1-1\tA\tB\tUS", "", "S1-2\tC\tD\tUS"])
        assert len(list(stream_records(p))) == 2


class TestRecordViews:
    def test_views_preserve_raw_and_add_flags(self):
        v = build_views("S1-9", "Béque", "189 Connmre Dr, Nashville, Tennessee", "US")
        assert isinstance(v, RecordViews)
        assert v.name_raw == "Béque"
        assert v.name_norm == "béque"  # base view KEEPS accents
        assert v.name_fold == "beque"  # fold view strips Latin accents
        assert v.addr_norm == "189 connmre dr nashville tennessee"
        assert not v.name_indic
        assert not v.addr_indic
        assert not v.addr_missing

    def test_missing_address_flag(self):
        v = build_views("S1-1", "Nion", "   ", "US")
        assert v.addr_missing
        assert v.addr_norm == ""

    def test_indic_flags(self):
        v = build_views("S2-2", "राम मार्केटिंग", "KH NO. 570 दिल्ली", "India")
        assert v.name_indic
        assert v.addr_indic
        assert v.name_fold == v.name_norm  # fold leaves Devanagari untouched

    def test_fold_differs_from_norm_for_latin_accents(self):
        v = build_views("S1-x", "Animal Welfare Nétwork", "260 Cadillac Dr", "US")
        assert v.name_fold == "animal welfare network"
        assert v.name_norm == "animal welfare " + "nétwork"
        assert v.name_fold != v.name_norm


class TestBuildTruthParquet:
    def test_round_trip_truth(self, tmp_path):
        src = _write(tmp_path, "truth.tsv", [
            "source1_entity_id\tmatched_entity_ids",
            "S1-1\tS2-10,S3-20",
            "S1-2\t",
            "S1-3\tS2-11",
        ])
        out = tmp_path / "truth.parquet"
        n = build_truth_parquet(src, out)
        assert n == 3
        t = pq.read_table(out)
        assert t.column_names == ["source1_entity_id", "matched_entity_ids"]
        rows = t.to_pydict()
        assert rows["matched_entity_ids"][1] == ""
        assert rows["matched_entity_ids"][0] == "S2-10,S3-20"

    def test_stream_truth_empty_is_empty_list(self, tmp_path):
        src = _write(tmp_path, "truth.tsv", [
            "source1_entity_id\tmatched_entity_ids",
            "S1-2\t",
            "S1-1\tS2-10, S3-20",
        ])
        rows = {k: v for k, v in stream_truth(src)}
        assert rows["S1-2"] == []
        assert rows["S1-1"] == ["S2-10", " S3-20"]

    def test_stream_truth_duplicate_s1_row_is_fatal(self, tmp_path):
        src = _write(tmp_path, "truth.tsv", [
            "source1_entity_id\tmatched_entity_ids",
            "S1-1\tS2-10",
            "S1-1\t",
        ])
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            list(stream_truth(src))

    def test_stream_truth_duplicate_id_in_list_is_fatal(self, tmp_path):
        src = _write(tmp_path, "truth.tsv", [
            "source1_entity_id\tmatched_entity_ids",
            "S1-1\tS2-10,S2-10",
        ])
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            list(stream_truth(src))


class TestFingerprint:
    def test_sha256_file_matches_hashlib(self, tmp_path):
        import hashlib as _hl

        p = _write(tmp_path, "f.bin", ["hello"])
        assert sha256_file(p) == _hl.sha256(b"hello\n").hexdigest()

    def test_sha256_file_is_stable_across_calls(self, tmp_path):
        p = _write(tmp_path, "f2.bin", ["abc", "def"])
        assert sha256_file(p) == sha256_file(p)


class TestTruthParquet:
    def test_iter_truth_parquet_round_trip(self, tmp_path):
        from ber.records import iter_truth_parquet

        src = _write(tmp_path, "truth.tsv", [
            "source1_entity_id\tmatched_entity_ids",
            "S1-1\tS2-10,S3-20",
            "S1-2\t",
            "S1-3\tS2-11",
        ])
        out = tmp_path / "truth.parquet"
        build_truth_parquet(src, out)
        rows = dict(iter_truth_parquet(out))
        assert rows["S1-1"] == ["S2-10", "S3-20"]
        assert rows["S1-2"] == []
        assert rows["S1-3"] == ["S2-11"]

    def test_iter_truth_parquet_preserves_order(self, tmp_path):
        from ber.records import iter_truth_parquet

        src = _write(tmp_path, "truth.tsv", [
            "source1_entity_id\tmatched_entity_ids",
            "S1-b\t",
            "S1-a\tS2-1",
        ])
        out = tmp_path / "truth.parquet"
        build_truth_parquet(src, out)
        keys = [k for k, _ in iter_truth_parquet(out)]
        assert keys == ["S1-b", "S1-a"]


class TestSourceTableParquet:
    def test_build_writes_ordinals_views_flags(self, tmp_path):
        from ber.records import build_source_parquet

        src = _write(tmp_path, "s1.tsv", [
            "entity_id\tbusiness_name\tbusiness_address\tcountry",
            "S1-1\tBéque\t189 Connmre Dr, Nashville\tUS",
            "S1-2\tराम प्राइवेट\t\tIndia",
        ])
        out = tmp_path / "s1.parquet"
        n, dup = build_source_parquet(src, out, "S1")
        assert (n, dup) == (2, 0)
        t = pq.read_table(out)
        assert t.column_names == [
            "ordinal", "entity_id", "name_raw", "addr_raw",
            "name_norm", "addr_norm", "name_fold", "addr_fold",
            "country", "name_indic", "addr_indic", "addr_missing",
        ]
        d = t.to_pydict()
        assert d["ordinal"] == [0, 1]
        assert d["name_fold"][0] == "beque"
        assert d["addr_missing"][1] is True
        assert d["name_indic"][1] is True

    def test_duplicate_entity_id_is_fatal(self, tmp_path):
        from ber.records import build_source_parquet

        src = _write(tmp_path, "s1.tsv", [
            "entity_id\tbusiness_name\tbusiness_address\tcountry",
            "S1-1\tA\tB\tUS",
            "S1-1\tC\tD\tUS",
        ])
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            build_source_parquet(src, tmp_path / "o.parquet", "S1")
