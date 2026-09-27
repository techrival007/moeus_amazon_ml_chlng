# Business Entity Resolution — Amazon ML Challenge 2026

Retrieval-first entity resolution over three noisy business-record sources,
built and measured end-to-end on the full dataset (12.5M train records,
11.7M test records, 7.6M labeled links).

## Approach (one paragraph)

Index the deduplicated Source-1 references per country with a merged
trigram+word+accent-fold term stream (tantivy BM25). For every Source-2/3
record, retrieve the top references in two field lanes (name, address),
fuse the lanes with reciprocal-rank fusion (offset 60), add a joint
name-AND-address conjunction rescue lane (top-2), keep the top-2 fused
(strict reverse budget, exact ties expand) plus the joint rescues, and
feed the survivors to a LightGBM pair classifier over 37 comparison
features. A global threshold on Platt-calibrated scores, selected on
held-out complete reference rows with the exact per-reference macro-F0.5
scorer, produces the final match sets.

## Measured results (frozen release policy)

| Cohort | Role | macro-F0.5 | Oracle ceiling | Notes |
|---|---|---:|---:|---|
| D (selection) | 5% entities | **0.9718** | 0.9848 | tau selected on complete D rows; owner challenger rejected (LCB −0.0004) |
| K (calibration) | 5% entities | **0.9725** | 0.9851 | Platt a=−11.82 b=3.34; tau=0.2584 calibrated |
| A2 (fresh audit, untouched) | 5% entities reserved from B after all fixes | **0.9761** | 0.9853 | one-shot frozen policy; US 0.9849 / India 0.9629 / singletons 0.9782 |

Splits are identity-component-disjoint (F/D/K/A = 5/5/5/5%, A2 reserved
from B before any new selection, B = reusable background catalog and
traffic), so no truth edge crosses a protected cohort, and evaluation
catalogs match observed per-country test sizes (US 663,106 / India
809,986, capped by availability at 759,710).

## Reproduction

Environment (uv, Python 3.13):

```bash
cd project
uv sync                                   # install pinned deps + editable ber
uv run pytest                             # 192 unit tests (scorer, splits,
                                          # fusion, joint lane, decisions,
                                          # cache invalidation, export/audit)
```

Pipeline (all stages resumable; shard fingerprints reject stale mixes):

```bash
DATA=../student_resource/dataset
uv run python -m ber.cli ingest  --dataset-dir $DATA --workers 6
uv run python -m ber.cli split   --seed 7
uv run python -m ber.cli reserve-audit --fraction 0.05 --seed 13
for C in fit D K A2; do
  uv run python -m ber.cli env      --cohort $C
  uv run python -m ber.cli index    --cohort $C --threads 6
  uv run python -m ber.cli dfmap    --cohort $C
  uv run python -m ber.cli retrieve --cohort $C --budget 2 --joint-k 2 --workers 8
  uv run python -m ber.cli features --cohort $C --workers 8
done
uv run python -m ber.cli train    --rounds 1000 --threads 8 --seed 0   # 127 leaves via TrainConfig
uv run python -m ber.cli select   # D: policy selection
uv run python -m ber.cli calibrate  # K: Platt + frozen tau
uv run python -m ber.cli audit    --cohort A2  # one-shot frozen-policy audit
uv run python -m ber.cli test     --output-dir output --joint-k 2 --workers 8
uv run python -m ber.cli validate --output-dir output
```

Wall-clock on Apple M-series (10 cores, 16 GiB): ingest 118 s, split 162 s,
per train cohort ~20-25 min (retrieval dominates; ~640k queries at
per-source-calibrated traffic), training ~60 s, selection ~40 s,
calibration ~25 s, audit ~20 s, full test run ~3 h (9.97M queries at
~2-6 ms/query across 8 workers, then features ~3 min and vectorized
decisions + export ~4 min).

## Design decisions (measured)

- **Reverse retrieval (target→reference)** with strict per-target budget 2
  plus a **joint conjunction rescue lane** (name AND address terms,
  top-2): screening showed +4.6pp link recall for ~1 extra candidate per
  query; the full-D candidate oracle rose 0.9627 → 0.9848.
- **Merged per-lane term streams** (trigrams + words + accent-folded
  trigrams in ONE BM25 field per lane): 2 searches/query instead of 6.
- **Per-field document frequencies** (each token counted once per field
  per document; IDF denominator = actual catalog size): query planning and
  the shared-token weight both use field-correct statistics.
- **Supervision restricted to F-role targets**: background-owned positives
  are excluded from fit labels instead of being mislabeled negative
  (A-cohort score +0.0025 from this fix alone).
- **Threshold policy over owner-exclusivity**: the owner challenger lost
  on D in every configuration tried (LCB −0.0004 to −0.0009); the simpler
  threshold policy ships.
- **Cache hygiene**: every shard fingerprint binds index/traffic/df-map
  content and all runtime settings; enriched shards rebuild when edges
  change; index rebuilds atomically replace stale documents; shards
  removed from the traffic set are pruned, never silently reprocessed.
- Candidate support is exported exactly as scored (b=2 support plus joint
  rescues, including rejected candidates); `audit_scored_support` verifies
  byte-level equality between candidate_pairs.tsv and the scored edge set.
- **Fresh audit discipline**: A2 was reserved from B and protected
  (`verify_split`) before any of the new retrieval/model variants were
  selected; the reported audit number is one-shot under the K-frozen
  policy.

## Layout

```
src/ber/            pipeline modules (records, normalize, splits, env,
                    retrieval, candidates, features, training, decisions,
                    metrics, export, audit, pipeline, stages, cli)
src/ber/tests/      192 unit tests incl. official-validator parity
data/               generated artifacts (gitignored): records, manifest,
                    env catalogs/traffic, indexes, edges, features, models,
                    policy, reports
output/            matching_results.tsv + candidate_pairs.tsv
reports/            stage JSON reports with fingerprints and timings
```

## Honesty notes

- A2-cohort numbers are one-shot (policy frozen on K before audit).
- Blocking loss (0.0147) exceeds scoring loss (0.0092): the residual misses
  are transliteration-variant, missing-address, and generic-name links —
  measured next steps are a multilingual neural retriever (blocking) and a
  compact multilingual cross-encoder (scoring) on the difficult pairs.
- Stronger lexical settings (lane_k 32, budget 4, joint_k 8) were screened
  and rejected: +1pp recall for 2-3x candidates and runtime.
- No external data, APIs, or lookups were used at any stage; dependencies:
  numpy, pyarrow, lightgbm (MIT), tantivy (MIT), rapidfuzz (MIT).
- License/parameter compliance: all learned components are locally trained
  artifacts (LightGBM); the deployed stack contains no pretrained weights.
