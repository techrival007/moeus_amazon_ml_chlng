"""Regression tests at the artifact-cache boundaries of the real pipeline."""

import json

import pyarrow as pa
import pyarrow.parquet as pq


def test_retrieval_cache_key_includes_traffic_df_and_query_settings(tmp_path):
    from ber.stages import _retrieval_fingerprint

    idx = tmp_path / "index" / "D" / "US"
    idx.mkdir(parents=True)
    (idx / "index.json").write_text(json.dumps({"catalog_fingerprint": "catalog-a"}))
    dm = tmp_path / "dfmap"
    dm.mkdir()
    (dm / "D_US.parquet").write_bytes(b"df-a")
    traffic = tmp_path / "src2_0000.parquet"
    traffic.write_bytes(b"traffic-a")
    fp = lambda lane_k=8, max_terms=16, joint_k=0: _retrieval_fingerprint(
        tmp_path, "D", "US", traffic, 2, lane_k=lane_k,
        max_terms=max_terms, joint_k=joint_k
    )
    original = fp()
    assert original != fp(lane_k=16)
    assert original != fp(max_terms=24)
    assert original != fp(joint_k=2)
    traffic.write_bytes(b"traffic-b")
    assert original != fp()
    traffic.write_bytes(b"traffic-a")
    (dm / "D_US.parquet").write_bytes(b"df-b")
    assert original != fp()


def test_enrichment_rebuilds_when_its_edges_change(tmp_path):
    from ber.stages import _ensure_enriched

    traffic = pa.table({
        "source": [2], "tgt_ord": [0], "name_norm": ["target"],
        "addr_norm": ["address"], "name_indic": [False],
        "addr_indic": [False], "addr_missing": [False],
    })
    catalog = pa.table({
        "ref_ord": [0, 1], "name_norm": ["a", "b"],
        "addr_norm": ["a", "b"], "name_indic": [False, False],
        "addr_indic": [False, False], "addr_missing": [False, False],
    })
    edge = tmp_path / "edge.parquet"
    enriched = tmp_path / "enriched.parquet"
    pq.write_table(pa.table({"source": [2], "tgt_ord": [0], "ref_ord": [0]}), edge)
    _ensure_enriched(edge, traffic, catalog, enriched, "first")
    assert pq.read_table(enriched).column("ref_ord").to_pylist() == [0]
    pq.write_table(pa.table({"source": [2], "tgt_ord": [0], "ref_ord": [1]}), edge)
    _ensure_enriched(edge, traffic, catalog, enriched, "second")
    assert pq.read_table(enriched).column("ref_ord").to_pylist() == [1]


def test_rebuilding_changed_catalog_replaces_old_index_documents(tmp_path):
    from ber.retrieval import QueryViews, Retriever
    from ber.stages import stage_index

    catalog = tmp_path / "env" / "D" / "catalog_US.parquet"
    catalog.parent.mkdir(parents=True)

    def write_catalog(name):
        pq.write_table(pa.table({
            "ref_ord": [0], "name_tri": [name], "name_word": [name],
            "name_foldtri": [name], "addr_tri": [""],
            "addr_word": [""], "addr_foldtri": [""],
        }), catalog)

    write_catalog("alpha")
    stage_index(tmp_path, "D", threads=1)
    write_catalog("bravo")
    stage_index(tmp_path, "D", threads=1)
    retriever = Retriever(tmp_path / "index" / "D" / "US")
    old_name_hits, _ = retriever.search(QueryViews("alpha", "", "alpha", ""))
    new_name_hits, _ = retriever.search(QueryViews("bravo", "", "bravo", ""))
    assert old_name_hits == []
    assert [h.ref_id for h in new_name_hits] == [0]


def test_dfmap_counts_each_token_once_per_index_field_and_document(tmp_path):
    from ber.stages import stage_dfmap

    env = tmp_path / "env" / "D"
    env.mkdir(parents=True)
    pq.write_table(pa.table({
        "name_tri": ["abc abc", "abc"], "name_word": ["abc", "abc"],
        "name_foldtri": ["abc", "abc"], "addr_tri": ["abc", "xyz"],
        "addr_word": ["abc", "xyz"], "addr_foldtri": ["abc", "xyz"],
    }), env / "catalog_US.parquet")
    stage_dfmap(tmp_path, "D")
    table = pq.read_table(tmp_path / "dfmap" / "D_US.parquet")
    rows = {(r["field"], r["token"]): r["df"] for r in table.to_pylist()}
    assert rows == {("name_all", "abc"): 2, ("addr_all", "abc"): 1,
                    ("addr_all", "xyz"): 1}
    meta = json.loads((tmp_path / "dfmap" / "D_US.json").read_text())
    assert meta["n_docs"] == 2


def test_feature_worker_uses_dfmap_metadata_passed_as_worker_path(tmp_path):
    from ber.features import FEATURE_NAMES
    from ber.stages import feature_worker

    enriched = tmp_path / "enriched.parquet"
    output = tmp_path / "features.parquet"
    dm = tmp_path / "dfmap.parquet"
    pq.write_table(pa.table({
        "tgt_ord": [0], "ref_ord": [0], "source": [2],
        "ref_name_norm": ["rare acme"], "ref_addr_norm": ["1 main"],
        "ref_name_indic": [False], "ref_addr_indic": [False], "ref_addr_missing": [False],
        "tgt_name_norm": ["rare acme"], "tgt_addr_norm": ["1 main"],
        "tgt_name_indic": [False], "tgt_addr_indic": [False], "tgt_addr_missing": [False],
        "margin": [0.5], "fused_v": [1.0], "name_rank": [1], "addr_rank": [1],
        "is_name_top1": [True], "is_addr_top1": [True], "name_score": [1.0],
        "addr_score": [1.0], "pool_size": [1],
        "joint_rank": [0], "joint_score": [0.0],
    }), enriched)
    pq.write_table(pa.table({"field": ["name_all"], "token": ["rare"], "df": [1]}), dm)
    dm.with_suffix(".json").write_text(json.dumps({"n_docs": 100}))
    feature_worker((str(enriched), str(output), "test-fingerprint", str(dm)))
    table = pq.read_table(output)
    assert table.num_rows == 1
    assert table.column("name_shared_df_weight").to_pylist()[0] > 1
    assert table.column_names[3:] == list(FEATURE_NAMES)


def test_test_stage_rejects_stale_model_and_feature_policy(tmp_path):
    import pytest

    from ber.stages import FEATURE_VERSION, RETRIEVAL_VERSION, stage_test

    model_dir = tmp_path / "models"
    policy_dir = tmp_path / "policy"
    model_dir.mkdir()
    policy_dir.mkdir()
    (model_dir / "lgbm_v1.txt").write_text("model-v2")
    bundle = {"feature_version": FEATURE_VERSION, "retrieval_version": RETRIEVAL_VERSION,
              "model_fingerprint": "some-other-model", "policy": {"kind": "threshold", "tau": 0.5},
              "calibrator": {"a": -1, "b": 1}}
    (policy_dir / "release.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="model fingerprint"):
        stage_test(tmp_path, tmp_path / "output", lane_k=16, joint_k=2)

    from ber.records import sha256_file
    bundle["model_fingerprint"] = sha256_file(model_dir / "lgbm_v1.txt")
    bundle["feature_version"] = "obsolete-feature-version"
    (policy_dir / "release.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="feature version"):
        stage_test(tmp_path, tmp_path / "output")

    bundle["feature_version"] = FEATURE_VERSION
    bundle["retrieval_recipe"] = {"budget": 2, "lane_k": 8,
                                  "max_terms": 16, "joint_k": 2}
    (policy_dir / "release.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="retrieval recipe"):
        stage_test(tmp_path, tmp_path / "output", lane_k=16, joint_k=2)


def test_retrieval_worker_marks_joint_rescue_evidence_on_scored_edges(tmp_path):
    from ber.retrieval import IndexDoc, build_index
    from ber.stages import retrieve_worker

    idx = tmp_path / "idx"
    build_index(idx, [IndexDoc(0, "sunrise", "100 pearl"),
                      IndexDoc(1, "sunrise", "999 cedar")], threads=1)
    traffic = tmp_path / "traffic.parquet"
    output = tmp_path / "edges.parquet"
    dm = tmp_path / "dfmap.parquet"
    pq.write_table(pa.table({
        "tgt_ord": [0], "source": [2], "name_norm": ["sunrise"],
        "addr_norm": ["100 pearl"], "name_fold": ["sunrise"],
        "addr_fold": ["100 pearl"],
    }), traffic)
    pq.write_table(pa.table({
        "field": ["name_all", "addr_all", "addr_all"],
        "token": ["sunrise", "100", "pearl"], "df": [2, 1, 1],
    }), dm)
    retrieve_worker((str(idx), str(traffic), str(output), "revision-one",
                     1, 8, str(dm), 16, 2))
    edges = pq.read_table(output).to_pylist()
    assert any(e["ref_ord"] == 0 and e["joint_rank"] == 1 and e["joint_score"] > 0
               for e in edges)


def test_fresh_audit_reserves_whole_background_identity_groups(tmp_path):
    from ber.pipeline import stage_reserve_audit

    manifest = tmp_path / "manifest"
    manifest.mkdir()
    original = manifest / "split.parquet"
    pq.write_table(pa.table({
        "entity_id": ["S1-a", "S2-a", "S3-a", "S1-f", "S2-f", "S1-d", "S2-unlinked"],
        "group_id": [10, 10, 10, 11, 11, 12, -1],
        "role": ["B", "B", "B", "F", "F", "D", "B"],
        "is_anchor": [True, False, False, True, False, True, False],
    }), original)
    stage_reserve_audit(tmp_path, fraction=1.0, seed=13)
    table = pq.read_table(original)
    roles = dict(zip(table.column("entity_id").to_pylist(), table.column("role").to_pylist()))
    assert roles == {"S1-a": "A2", "S2-a": "A2", "S3-a": "A2",
                     "S1-f": "F", "S2-f": "F", "S1-d": "D", "S2-unlinked": "B"}
    assert (manifest / "split_before_a2.parquet").exists()
    stage_reserve_audit(tmp_path, fraction=1.0, seed=13)
    assert pq.read_table(original).column("role").to_pylist() == table.column("role").to_pylist()


def test_validation_target_traffic_preserves_per_source_test_densities(tmp_path):
    from ber.cli import _quotas_and_densities

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "ingest.json").write_text(json.dumps({
        "country_distribution": {
            "test_s1": {"US": 10, "India": 5},
            "test_s2": {"US": 20, "India": 15},
            "test_s3": {"US": 30, "India": 5},
        },
        "test_target_density_per_ref": {"US": 5.0, "India": 4.0},
    }))
    quotas, density = _quotas_and_densities(tmp_path)
    assert quotas == {"US": 10, "India": 5}
    assert density == {"US": {2: 2.0, 3: 3.0},
                       "India": {2: 3.0, 3: 1.0}}


def test_stage_prunes_shards_removed_from_the_current_traffic_set(tmp_path):
    from ber.stages import stage_retrieve

    env = tmp_path / "env" / "D" / "traffic_US"
    edges = tmp_path / "edges" / "D" / "US"
    idx = tmp_path / "index" / "D" / "US"
    dfmap = tmp_path / "dfmap"
    for d in (env, edges, idx, dfmap):
        d.mkdir(parents=True)
    (idx / "index.json").write_text(json.dumps({"catalog_fingerprint": "cat"}))
    pq.write_table(pa.table({
        "tgt_ord": [0], "source": [2], "name_norm": ["sunrise"],
        "addr_norm": ["100 pearl"], "name_fold": ["sunrise"], "addr_fold": ["100 pearl"],
    }), env / "src2_0000.parquet")
    pq.write_table(pa.table({"field": ["name_all"], "token": ["sunrise"], "df": [1]}),
                   dfmap / "D_US.parquet")

    stale = edges / "src2_0002.parquet"
    pq.write_table(pa.table({"gone": [1]}), stale)
    stale.with_suffix(".json").write_text(json.dumps({"fingerprint": "old"}))
    from ber.retrieval import IndexDoc, build_index
    build_index(idx, [IndexDoc(0, "sunrise", "100 pearl")], threads=1)

    stage_retrieve(tmp_path, "D", budget=1, workers=1, joint_k=0)
    assert not stale.exists()
    assert not stale.with_suffix(".json").exists()
    assert (edges / "src2_0000.parquet").exists()


def test_features_stage_prunes_stale_edge_and_enriched_shards(tmp_path):
    from ber.stages import stage_features

    env = tmp_path / "env" / "D"
    edges = tmp_path / "edges" / "D"
    features = tmp_path / "features" / "D"
    for country in ("US",):
        (env / f"traffic_{country}").mkdir(parents=True)
        (edges / country).mkdir(parents=True)
        (features / country).mkdir(parents=True)
        (env / f"catalog_{country}.parquet").parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "ref_ord": [0], "entity_id": ["S1-0"], "country": [country],
            "name_norm": ["a"], "addr_norm": ["b"], "name_fold": ["a"], "addr_fold": ["b"],
            "name_tri": ["a"], "name_word": ["a"], "name_foldtri": ["a"],
            "addr_tri": ["b"], "addr_word": ["b"], "addr_foldtri": ["b"],
            "name_indic": [False], "addr_indic": [False], "addr_missing": [False],
        }), env / f"catalog_{country}.parquet")
        pq.write_table(pa.table({
            "tgt_ord": [0], "entity_id": ["S2-0"], "source": [2], "country": [country],
            "name_norm": ["a"], "addr_norm": ["b"], "name_fold": ["a"], "addr_fold": ["b"],
            "name_indic": [False], "addr_indic": [False], "addr_missing": [False],
        }), env / f"traffic_{country}" / "src2_0000.parquet")
        pq.write_table(pa.table({
            "tgt_ord": [0], "ref_ord": [0], "source": [2], "fused_v": [1.0],
            "name_rank": [1], "addr_rank": [1], "name_score": [1.0], "addr_score": [1.0],
            "margin": [1.0], "pool_size": [1], "joint_rank": [0], "joint_score": [0.0],
            "is_name_top1": [True], "is_addr_top1": [True],
        }), edges / country / "src2_0000.parquet")
        for stale_name in ("src2_0002.parquet", "enriched_src2_0002.parquet"):
            pq.write_table(pa.table({"gone": [1]}), features / country / stale_name)
        pq.write_table(pa.table({"gone": [1]}), edges / country / "src2_0002.parquet")

    stage_features(tmp_path, "D", workers=1)
    kept = sorted(p.name for p in (features / "US").iterdir())
    assert kept == ["enriched_src2_0000.json", "enriched_src2_0000.parquet",
                    "src2_0000.json", "src2_0000.parquet"]
    assert not (edges / "US" / "src2_0002.parquet").exists()
