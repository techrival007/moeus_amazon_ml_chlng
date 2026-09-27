"""Pair comparison features over candidate edges.

Contract from master.md section 8.5: compact, auditable schema where each
family answers a named error pattern; ID suffixes never leak; missingness is
distinct from conflict; retrieval context carries provenance.
"""

import math

import pytest

from ber.features import FEATURE_NAMES, pair_features
from ber.records import build_views


def views(eid, name, addr, country="US"):
    return build_views(eid, name, addr, country)


class TestFeatureVector:
    def test_names_aligned_and_finite(self):
        ref = views("S1-1", "Orelee's Barbershop", "1795 Westchester Drive, High Point, NC")
        tgt = views("S2-1", "Orelee s Barbershop", "1795 Westchester Drive, High Point, NC")
        v = pair_features(
            ref, tgt,
            fused_v=1.0, rank=0, is_name_top1=True, is_addr_top1=True,
            name_score=2.0, addr_score=2.0, margin=0.01, pool_size=2,
        )
        assert len(v) == len(FEATURE_NAMES)
        assert all(isinstance(x, float) for x in v)
        assert all(math.isfinite(x) for x in v)

    def test_identical_records_score_perfectly(self):
        ref = views("S1-1", "Orelee's Barbershop", "1795 Westchester Drive, High Point, NC")
        tgt = views("S2-1", "Orelee's Barbershop", "1795 Westchester Drive, High Point, NC")
        v = pair_features(ref, tgt, 1.0, 0, True, True, 2.0, 2.0, 0.01, 2)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["name_exact"] == 1.0
        assert d["addr_exact"] == 1.0
        assert d["name_jw"] == pytest.approx(1.0)
        assert d["name_tok_jaccard"] == pytest.approx(1.0)
        assert d["name_lev_norm"] == pytest.approx(1.0)

    def test_same_name_different_city(self):
        ref = views("S1-1", "Prime Money", "17560 Ellis Road, Tahlequah, OK")
        tgt = views("S2-1", "Prime Money", "1200 Market Street, San Francisco, CA")
        v = pair_features(ref, tgt, 0.9, 0, True, False, 2.0, 0.5, 0.01, 3)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["name_exact"] == 1.0
        assert d["addr_exact"] == 0.0
        assert d["addr_tok_jaccard"] < 0.3
        assert d["name_exact_addr_conflict"] == 1.0

    def test_missing_target_address_is_flagged_not_similar(self):
        ref = views("S1-1", "Nion", "7100 Brigham Road, Henrico County, VA")
        tgt = views("S2-1", "Nion", "")
        v = pair_features(ref, tgt, 0.9, 0, True, False, 2.0, 0.0, 0.01, 2)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["addr_target_missing"] == 1.0
        assert d["addr_jw"] == 0.0
        assert d["addr_tok_jaccard"] == 0.0
        assert d["name_exact"] == 1.0
        assert d["name_exact_addr_missing"] == 1.0

    def test_missing_reference_address(self):
        ref = views("S1-1", "Nion", "")
        tgt = views("S3-1", "Nion", "7100 Brigham Road, Henrico County, VA")
        v = pair_features(ref, tgt, 0.9, 0, True, False, 2.0, 0.0, 0.01, 2)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["addr_ref_missing"] == 1.0
        assert d["addr_jw"] == 0.0

    def test_shared_numeric_tokens(self):
        ref = views("S1-1", "Kelly Advisory", "301 1st Street, Chokio, MN")
        tgt = views("S2-1", "Kelley Advisory", "301 1st St, Chokio, MN")
        v = pair_features(ref, tgt, 0.95, 0, True, True, 2.0, 1.5, 0.01, 2)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["addr_shared_numeric"] >= 2  # "301" and "1st"/"1" numeric tokens
        assert d["addr_numeric_jaccard"] > 0.0

    def test_no_shared_numeric_tokens(self):
        ref = views("S1-1", "Helios", "66 Edgewood Street, Bridgeport, CT")
        tgt = views("S2-1", "Helios", "4828 Hedges Avenue, Kansas City, MO")
        v = pair_features(ref, tgt, 0.9, 0, True, False, 2.0, 0.5, 0.01, 2)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["addr_shared_numeric"] == 0.0
        assert d["addr_numeric_jaccard"] == 0.0

    def test_word_order_transposition_captured(self):
        ref = views("S1-1", "Custom Wealth Services LLC", "5559 Orville Avenue, Columbus, OH")
        tgt = views("S2-1", "Wealth Custom Services LLC", "Orville Avenue 5559, Columbus OH")
        v = pair_features(ref, tgt, 0.8, 0, False, False, 1.0, 1.0, 0.02, 3)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["name_sorted_exact"] == 1.0  # same token multiset
        assert d["name_exact"] == 0.0

    def test_token_containment(self):
        ref = views("S1-1", "Global", "1 Main St, Austin, TX")
        tgt = views("S2-1", "Global Developers Medicals Private Limited", "1 Main St, Austin, TX")
        v = pair_features(ref, tgt, 0.7, 0, False, True, 1.0, 2.0, 0.01, 3)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["name_containment"] == 1.0
        assert d["name_tok_jaccard"] < 0.3

    def test_source_flag_from_target_id(self):
        ref = views("S1-1", "Helios", "66 Edgewood Street, Bridgeport, CT")
        tgt2 = views("S2-9", "Helios", "66 Edgewood Street, Bridgeport, CT")
        tgt3 = views("S3-9", "Helios", "66 Edgewood Street, Bridgeport, CT")
        v2 = pair_features(ref, tgt2, 1.0, 0, True, True, 2.0, 2.0, 0.01, 2)
        v3 = pair_features(ref, tgt3, 1.0, 0, True, True, 2.0, 2.0, 0.01, 2)
        d2 = dict(zip(FEATURE_NAMES, v2))
        d3 = dict(zip(FEATURE_NAMES, v3))
        assert d2["source_s2"] == 1.0 and d3["source_s2"] == 0.0

    def test_indic_script_flags(self):
        ref = views("S1-1", "राम मार्केटिंग", "KH No 570, New Delhi", "India")
        tgt = views("S2-1", "राम मार्केट", "KH No 570, New Delhi", "India")
        v = pair_features(ref, tgt, 1.0, 0, True, True, 2.0, 2.0, 0.01, 2)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["name_target_indic"] == 1.0
        assert d["name_ref_indic"] == 1.0

    def test_retrieval_context_passthrough(self):
        ref = views("S1-1", "Helios", "66 Edgewood Street")
        tgt = views("S2-1", "Helios", "66 Edgewood Street")
        v = pair_features(ref, tgt, 0.75, 3, False, True, 1.25, 0.5, None, 5)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["fused_v"] == 0.75
        assert d["rank"] == 3.0
        assert d["is_name_top1"] == 0.0
        assert d["is_addr_top1"] == 1.0
        assert d["name_lane_score"] == 1.25
        assert d["addr_lane_score"] == 0.5
        assert d["margin_g"] == 0.0  # missing margin encoded as zero
        assert d["pool_size"] == 5.0

    def test_joint_rescue_rank_and_score_reach_the_matcher(self):
        ref = views("S1-1", "Sunrise Trading", "100 Pearl Road")
        tgt = views("S3-1", "Sunrise", "Pearl Road")
        v = pair_features(ref, tgt, 0.0, 0, False, False, 0.0, 0.0,
                          None, 3, joint_rank=1, joint_score=11.5)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["joint_rank"] == 1.0
        assert d["joint_score"] == 11.5

    def test_address_typos_score_levenshtein_similarity(self):
        ref = views("S1-1", "Ray Innovative", "1101 405th Street, Woodland, WA")
        tgt = views("S2-1", "RAY INNOVATIVE", "1101- 405TH STREET, WOODDLAND CITY, WA")
        clean = views("S2-2", "RAY INNOVATIVE", "1101 405TH STREET, WOODLAND CITY, WA")
        v = pair_features(ref, tgt, 0.9, 1, True, True, 2.0, 2.0, 0.01, 2)
        d = dict(zip(FEATURE_NAMES, v))
        v2 = pair_features(ref, clean, 0.9, 1, True, True, 2.0, 2.0, 0.01, 2)
        d2 = dict(zip(FEATURE_NAMES, v2))
        assert 0.0 < d["addr_lev_norm"] < d2["addr_lev_norm"] <= 1.0

    def test_word_order_transposition_scores_sorted_name_similarity(self):
        ref = views("S1-1", "Galaxy It Private Limited", "17 Sri Ramsai Road, Mumbai")
        tgt = views("S2-1", "Private Galaxy It Limited", "B3/17 SRI RAMSAI ROAD, MUMBAI")
        v = pair_features(ref, tgt, 0.8, 1, False, True, 1.0, 1.0, 0.01, 2)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["name_sorted_exact"] == 1.0
        assert d["name_sorted_jw"] == pytest.approx(1.0)

    def test_address_component_reorder_flagged(self):
        ref = views("S1-1", "Anand Systems", "K-901, 9Th Floor, Delhi, North West")
        tgt = views("S2-1", "Anand Systems", "North West, 9Th Floor, Delhi, K-901")
        v = pair_features(ref, tgt, 0.9, 1, True, True, 2.0, 2.0, 0.01, 2)
        d = dict(zip(FEATURE_NAMES, v))
        assert d["addr_sorted_exact"] == 1.0
        assert d["addr_exact"] == 0.0

    def test_shared_df_weight_uses_token_document_frequency(self):
        ref = views("S1-1", "Zzyzx Consulting", "1 Main St")
        tgt = views("S2-1", "Zzyzx Consultings", "1 Main St")
        df = {"zzyzx": 1, "consulting": 1_000_000, "main": 500_000, "st": 900_000}
        with_rare = pair_features(ref, tgt, 1.0, 0, True, True, 2.0, 2.0, 0.01, 2, token_df=df)
        d = dict(zip(FEATURE_NAMES, with_rare))
        assert d["name_shared_df_weight"] > 0.0
        without = pair_features(ref, tgt, 1.0, 0, True, True, 2.0, 2.0, 0.01, 2, token_df=None)
        d2 = dict(zip(FEATURE_NAMES, without))
        assert d2["name_shared_df_weight"] == 0.0

    def test_shared_df_weight_uses_actual_catalog_size(self):
        ref = views("S1-1", "Rare Acme", "1 Main St")
        tgt = views("S2-1", "Rare Acme", "1 Main St")
        df = {"rare": 1, "acme": 3}
        vector = pair_features(ref, tgt, 1.0, 0, True, True, 2.0, 2.0,
                               0.01, 2, token_df=df, token_df_n_docs=100)
        weight = dict(zip(FEATURE_NAMES, vector))["name_shared_df_weight"]
        expected = (math.log(101 / 2) + math.log(101 / 4)) / 2 + 1
        assert weight == pytest.approx(expected)

    def test_common_tokens_have_lower_weight_than_rare(self):
        ref_a = views("S1-1", "Zzyqx Partners", "1 Main St")
        tgt_a = views("S2-1", "Zzyqx Partner", "1 Main St")
        ref_b = views("S1-2", "Global Services", "1 Main St")
        tgt_b = views("S2-2", "Global Service", "1 Main St")
        df = {"zzyqx": 2, "partners": 3, "global": 800_000, "services": 700_000, "main": 500_000, "st": 900_000}
        va = pair_features(ref_a, tgt_a, 1.0, 0, True, True, 2.0, 2.0, 0.01, 2, token_df=df)
        vb = pair_features(ref_b, tgt_b, 1.0, 0, True, True, 2.0, 2.0, 0.01, 2, token_df=df)
        da, db = dict(zip(FEATURE_NAMES, va)), dict(zip(FEATURE_NAMES, vb))
        assert da["name_shared_df_weight"] > db["name_shared_df_weight"]

    def test_no_id_suffix_leakage(self):
        # entity ids must not influence features beyond source prefix
        ref1 = views("S1-111", "Helios", "66 Edgewood Street")
        ref2 = views("S1-999", "Helios", "66 Edgewood Street")
        tgt = views("S2-5", "Helios", "66 Edgewood Street")
        ctx = (1.0, 0, True, True, 2.0, 2.0, 0.01, 2)
        v1 = pair_features(ref1, tgt, *ctx)
        v2 = pair_features(ref2, tgt, *ctx)
        assert v1 == v2
