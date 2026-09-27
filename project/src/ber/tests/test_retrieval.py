"""Country-partitioned tantivy reference indexes and two-lane retrieval.

Contract from master.md sections 9.1-9.3:
- Reference docs carry the ordinal plus six text fields: name/address
  trigram, word, and accent-folded-trigram views (precomputed term streams).
- Two lanes (name, address), each fusing three sub-searches (trigram field,
  word field, fold field) by within-lane RRF (offset 60) over sub-ranks,
  tie-broken by reference ordinal; each lane returns at most lane_k hits
  with 1-based ranks.
- A lane with no usable terms is inactive and returns no hits. Blank target
  address deactivates the address lane.
- Retrieval is deterministic for a fixed index and query.
- Hits carry within-lane fused scores and provenance; compaction happens in
  candidates.compact_strict, never here.
"""

from pathlib import Path

import pytest

from ber.retrieval import IndexDoc, QueryViews, Retriever, build_index, index_doc_from_views


def make_refs():
    return [
        index_doc_from_views(0, "orelee s barbershop", "1795 westchester drive high point nc",
                             "orelee s barbershop", "1795 westchester drive high point nc"),
        index_doc_from_views(1, "animal welfare network", "260 cadillac dr sacramento ca",
                             "animal welfare network", "260 cadillac dr sacramento ca"),
        index_doc_from_views(2, "animal welfare néetwork", "260 cadillac dr sacramento ca",
                             "animal welfare network", "260 cadillac dr sacramento ca"),
        index_doc_from_views(3, "राम मार्केटिंग", "kh no 570 new delhi",
                             "राम मार्केटिंग", "kh no 570 new delhi"),
        index_doc_from_views(4, "prime money", "17560 ellis road tahlequah ok",
                             "prime money", "17560 ellis road tahlequah ok"),
    ]


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    d = tmp_path_factory.mktemp("idx")
    build_index(d, make_refs())
    return Retriever(d)


class TestBuildAndSearch:
    def test_exact_name_match_ranks_first(self, retriever):
        q = QueryViews("orelee s barbershop", "1795 westchester drive high point nc",
                       "orelee s barbershop", "1795 westchester drive high point nc")
        name, addr = retriever.search(q)
        assert name and name[0].ref_id == 0
        assert name[0].rank == 1
        assert addr and addr[0].ref_id == 0

    def test_blank_address_deactivates_address_lane(self, retriever):
        q = QueryViews("prime money", "", "prime money", "")
        name, addr = retriever.search(q)
        assert addr == []
        assert name and name[0].ref_id == 4

    def test_blank_name_deactivates_name_lane(self, retriever):
        q = QueryViews("", "17560 ellis road tahlequah ok", "", "17560 ellis road tahlequah ok")
        name, addr = retriever.search(q)
        assert name == []
        assert addr and addr[0].ref_id == 4

    def test_accent_mismatch_recovers_via_fold_view(self, retriever):
        # target "animal welfare network" must find ref 2 whose base norm has é
        q = QueryViews("animal welfare network", "", "animal welfare network", "")
        name, _ = retriever.search(q)
        refs = [h.ref_id for h in name]
        assert 2 in refs and 1 in refs  # fold view matches the accented record

    def test_accented_target_finds_plain_reference(self, retriever):
        # target with é (base norm) has fold view matching ref 1
        q = QueryViews("animal welfare néetwork", "", "animal welfare network", "")
        name, _ = retriever.search(q)
        refs = [h.ref_id for h in name]
        assert 1 in refs and 2 in refs

    def test_devanagari_name_is_retrievable(self, retriever):
        q = QueryViews("राम मार्केटिंग", "kh no 570 new delhi", "राम मार्केटिंग", "kh no 570 new delhi")
        name, addr = retriever.search(q)
        assert name and name[0].ref_id == 3
        assert addr and addr[0].ref_id == 3

    def test_no_matching_terms_gives_empty_lanes(self, retriever):
        q = QueryViews("zzz qq vv xx", "kkk jjj hhh ggg", "zzz qq vv xx", "kkk jjj hhh ggg")
        name, addr = retriever.search(q)
        assert name == [] and addr == []

    def test_ranks_are_one_based_and_sequential(self, retriever):
        q = QueryViews("animal welfare network", "260 cadillac dr sacramento ca",
                       "animal welfare network", "260 cadillac dr sacramento ca")
        name, addr = retriever.search(q)
        for lane in (name, addr):
            if lane:
                assert [h.rank for h in lane] == list(range(1, len(lane) + 1))

    def test_lane_k_respected(self, retriever):
        q = QueryViews("animal welfare network", "260 cadillac dr sacramento",
                       "animal welfare network", "260 cadillac dr sacramento")
        name, addr = retriever.search(q, lane_k=1)
        assert len(name) <= 1 and len(addr) <= 1

    def test_merged_terms_dedupe_and_sort(self):
        from ber.retrieval import merged_terms

        terms = merged_terms("global services", "global services")
        assert terms == sorted(set(terms))
        # fold == norm here: union is just the tri/word terms
        assert "glo" in terms and "global" in terms and "ser" in terms

    def test_merged_terms_include_fold_variant(self):
        from ber.retrieval import merged_terms

        terms = merged_terms("animal welfare néetwork", "animal welfare network")
        # fold trigrams of the accented text are present alongside base terms
        assert "ork" in terms or "net" in terms or "etw" in terms

    def test_deterministic(self, retriever):
        q = QueryViews("animal welfare network", "260 cadillac dr sacramento ca",
                       "animal welfare network", "260 cadillac dr sacramento ca")
        a = retriever.search(q)
        b = retriever.search(q)
        assert [(h.ref_id, h.rank) for h in a[0]] == [(h.ref_id, h.rank) for h in b[0]]
        assert [(h.ref_id, h.rank) for h in a[1]] == [(h.ref_id, h.rank) for h in b[1]]

    def test_scores_finite_and_positive(self, retriever):
        q = QueryViews("orelee s barbershop", "1795 westchester drive high point nc",
                       "orelee s barbershop", "1795 westchester drive high point nc")
        for lane in retriever.search(q):
            for h in lane:
                assert h.score > 0.0
                assert h.score == h.score  # not NaN

    def test_retrieval_returns_reference_ordinals(self, retriever):
        q = QueryViews("prime money", "17560 ellis road tahlequah ok", "prime money", "17560 ellis road tahlequah ok")
        name, addr = retriever.search(q)
        assert all(isinstance(h.ref_id, int) for h in name + addr)
        assert all(h.ref_id in range(5) for h in name + addr)

    def test_field_specific_document_frequencies_plan_separate_rare_terms(self):
        from ber.retrieval import plan_lane_terms

        name_df = {"name_rare": 1, "both": 10}
        address_df = {"both": 1, "name_rare": 10}
        assert plan_lane_terms(["both", "name_rare"], name_df, 1) == ["name_rare"]
        assert plan_lane_terms(["both", "name_rare"], address_df, 1) == ["both"]
        assert plan_lane_terms(["unknown", "both"], name_df, 1) == ["both"]

    def test_retriever_uses_each_fields_own_document_frequencies(self, tmp_path):
        build_index(tmp_path / "idx", [IndexDoc(0, "nameone", "addressone")], threads=1)
        retr = Retriever(tmp_path / "idx")
        q = QueryViews("nameone", "addressone", "nameone", "addressone")
        df = {"name_all": {"nameone": 1}, "addr_all": {"addressone": 1}}
        name, addr = retr.search(q, df_map=df, max_terms=1)
        assert [h.ref_id for h in name] == [0]
        assert [h.ref_id for h in addr] == [0]

    def test_joint_search_requires_a_name_and_an_address_match(self, tmp_path):
        build_index(tmp_path / "idx", [
            IndexDoc(0, "sunrise", "100 pearl"),
            IndexDoc(1, "sunrise", "999 cedar"),
            IndexDoc(2, "evening", "100 pearl"),
        ], threads=1)
        retr = Retriever(tmp_path / "idx")
        q = QueryViews("sunrise", "pearl", "sunrise", "pearl")
        df = {"name_all": {"sunrise": 2}, "addr_all": {"pearl": 2}}
        hits = retr.search_joint(q, df_map=df, max_terms=8, joint_k=3)
        assert [h.ref_id for h in hits] == [0]


class TestBuildIndex:
    def test_rebuild_is_idempotent(self, tmp_path):
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        build_index(d1, make_refs())
        build_index(d2, make_refs())
        q = QueryViews("prime money", "17560 ellis road tahlequah ok", "prime money", "17560 ellis road tahlequah ok")
        r1, r2 = Retriever(d1), Retriever(d2)
        assert [(h.ref_id, h.rank) for h in r1.search(q)[0]] == [(h.ref_id, h.rank) for h in r2.search(q)[0]]

    def test_empty_reference_list_builds_empty_index(self, tmp_path):
        d = tmp_path / "empty"
        build_index(d, [])
        r = Retriever(d)
        q = QueryViews("anything", "", "anything", "")
        assert r.search(q) == ([], [])
