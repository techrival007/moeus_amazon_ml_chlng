# Local-first entity-resolution recovery plan

**Goal:** Improve the legitimate end-to-end macro-F0.5 toward 0.99 using only the supplied data and local compute until the G/VT quota is approved. Do not upload without auditing the actual exported rows. A hidden-test score cannot be certified locally.

**Environment:** `project/.venv/bin/python`; 10-core, 16 GiB Mac; 168 baseline unit tests passing. The existing A result is 0.9378349655, candidate oracle 0.9632872101; current test export maps target IDs to the wrong references. Keep old outputs as evidence, regenerate into a new directory.

**Execution, in independently verifiable increments:**

1. In `project/src/ber/tests/test_stages.py`, reproduce the mixed-S2/S3, out-of-order-reference bug on a tiny production-format fixture; confirm red. In `project/src/ber/stages.py`, correct ordinal mapping in the grouped writer; confirm green and run `project/.venv/bin/python -m pytest`. Re-score A against complete truth, regenerate both test TSVs from cached feature shards into a new output directory, and audit ID correspondence independently against scored `(ref_ord, source, tgt_ord)` edges.
2. Add failing tests for stale enriched feature shards and incomplete retrieval fingerprints; in `stages.py`, tie every cache key and enriched shard to edge/traffic/index/df-map content and all relevant runtime settings. Benchmark and assert that changing retrieval budget or term cap actually changes the downstream feature relation. Rebuilding an existing index must not silently append duplicate reference documents.
3. In `project/src/ber/tests/test_training.py`, reproduce a background-owned positive incorrectly counted as negative and confirm red; separate F-owned supervision from B-role background traffic in `stages.py`. In `project/src/ber/tests/test_features.py` and `test_retrieval.py`, exercise per-field document frequencies and missing terms; compute an IDF using actual catalog document count, not maximum aggregate token frequency. Rebuild/retrain in bounded shards and recalibrate on K.
4. Reserve an identity-disjoint audit partition from B *before* selecting new retrieval/model variants; exclude its components from supervised fits and background traffic in selection. Use D to attribute retrieval failures at true-owner rank and per country/script/missing-address. Compare bounded retrieval challengers (b=3/4, larger lane-k, name/address evidence, rare exact-match rescue) on equal held-out traffic, promoting only better `(macro-F, candidate count, runtime)` configurations. Recompute and inspect complete-truth candidate oracle first; aim for at least 0.998 before promising 0.99.
5. Once blocking is strong enough, fix verified scorer errors with features and a clean LightGBM model, choose only the decision threshold on K, and run the sealed audit once. Aim for at least 0.993 audit macro-F0.5 (with honest per-country slices). If no CPU configuration reaches this, document the measured retrieval/scoring gap and reserve only that part for the locked `g6e.16xlarge`.
6. Run the strongest reproducible model over every test reference, including France. Independently check that both TSVs contain all 1,732,544 S1 rows, every match is among the exact edges scored, and all targets exist and belong to their reference's country. Run the official validator with `--check-ids`, then stage the runnable package. Retain the two portal submissions until evidence warrants an upload.

**Work rules:** Test-first for each behavior-changing edit. Preserve source artifacts and old output files. Fingerprint/version caches when changing retrieval or feature semantics. Sample-before-full-sweep on the 16 GiB Mac. Use only provided data, license-compliant models, and exact per-reference macro-F0.5.

**Measured outcomes (2026-09-27, all local, Mac 10-core/16 GiB):**

- Export-order defect fixed with regression tests (`write_scored_outputs`); regenerated baseline passed the official validator and an independent full-edge audit (`audit_scored_support`, 23,167,551 exported pairs == scored pairs).
- Training supervision restricted to F-role targets (background-owned targets no longer labeled negative): A-cohort macro-F 0.9378 → 0.9403.
- Cache hygiene: retrieval fingerprints bind index/traffic/dfmap content and every runtime setting; enriched shards rebuild on edge/traffic change; index rebuilds are atomic (replacing stale documents); stale shards from older configurations are pruned, never reprocessed.
- Per-field document-frequency maps (each token counted once per field per document) with catalog size as the IDF denominator; screening showed the old blended map was slightly better for ranking, so the promoted recipe kept per-field maps with the measured settings.
- Joint name+address conjunction rescue lane (`search_joint`, `compact_with_joint`) promoted after screening (+4.6pp link recall at ~1 extra candidate/query): full D-cohort candidate oracle 0.9627 → **0.9848**.
- Forward S1→target rescue screened and rejected (11% of remaining misses at 4 candidates/reference).
- Traffic densities fixed to per-source test proportions (validation traffic halved to deployment-realistic volume).
- Fresh identity-disjoint audit cohort A2 reserved from B (88,181 anchor groups) before any new selection; `verify_split` protects A2.
- Scoring: features v4 (+`addr_lev_norm`, `name_sorted_jw`, `addr_sorted_exact`), LightGBM leaves 127 / 1000 rounds. One-shot frozen-policy audit on A2: **macro-F0.5 0.97609** (US 0.98488 / India 0.96285 / singletons 0.97818; oracle 0.98528, blocking loss 0.0147, scoring loss 0.0092).
- Frozen release recipe: budget 2, lane_k 8, max_terms 16, joint_k 2; snapshots in `project/artifacts/` (lgbm_v5, release_v5).

**Remaining gap to 0.99 (measured):** blocking loss 0.0147 + scoring loss 0.0092. Stronger lexical settings screened out as poor trade-offs (+1pp recall for 2-3x candidates and runtime). The residual misses are missing-address, transliteration-variant, and generic-name links — exactly the cases a multilingual neural retriever/cross-encoder addresses; that work is reserved for the approved `g6e.16xlarge`.
