# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A retrieval-first pipeline: country-partitioned tantivy BM25 indexes over
Source-1 references with merged trigram+word+accent-folded term streams,
two-lane (name/address) retrieval with reciprocal-rank fusion, a strict
reverse budget of 2, a joint name-and-address conjunction rescue lane
(`AND` of rare name and address terms, top-2), and a LightGBM pair
classifier over 37 comparison features, scored by a globally thresholded,
Platt-calibrated decision policy selected on held-out complete reference
rows with the exact macro-F0.5 metric. Frozen-policy audit macro-F0.5 on a
fresh, identity-disjoint audit cohort: **0.9761** (US 0.9849, India 0.9629,
singletons 0.9782); both submission files pass the official validator with
`--check-ids`, and an independent audit verifies every exported candidate
equals a pair the model actually scored (33,405,124 edges).

---

## 2. Methodology

### 2.1 Problem Analysis

Full-dataset measurements (all counts streamed from the supplied TSVs,
fingerprints in `data/reports/ingest.json`): 2,206,821 train / 1,732,544
test Source-1 entities; 5,034,616 + 5,285,603 train and 4,887,273 +
5,082,316 test Source-2/3 records; 7,638,365 labeled links with zero
reused targets (each target has at most one S1 owner); 123,247 true
singletons (5.58%); truth cardinalities 0..11 (4,776 references above 8);
France is 14.98% of test references with no labeled training analog; Indian
target records contain substantial native-script text (up to 546,606
records with Indic-block names per source); target addresses are sometimes
empty (2.3%); train S1 records are 100% ASCII while test France contains
73,335 accented addresses. All 7,638,365 observed true links are
within-country.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (retrieval-first, reverse direction)
**Core Innovation:** a joint name+address conjunction rescue lane that
recovers links missed by both individual lanes (full-cohort candidate oracle
0.9627 -> 0.9848), plus identity-component focal/background validation
cohorts at production catalog scale with the decision policy selected on
the exact macro-F0.5 of complete rows.

---

## 3. Candidate Generation (Blocking)

- **Direction:** reverse — every Source-2/3 record queries the country-
  partitioned S1 index; a reference may receive any number of targets
  (no reference capacity; multi-match rows up to 11 preserved).
- **Indexing:** tantivy BM25, one index per country (US 663,106;
  India 809,986; France 259,452 references), each document carrying two
  merged term streams (name lane, address lane) built from the union of
  character trigrams, word tokens (len>=2), and accent-folded trigrams.
  The fold view strips combining marks only over Latin bases, preserving
  Devanagari/Tamil text byte-for-byte. Index rebuilds are atomic replaces;
  stale shards from older configurations are pruned, never reprocessed.
- **Query planning:** per lane, the rarest 16 known terms by per-field
  document frequency (each token counted once per field per document);
  unknown terms are dropped (cannot match).
- **Fusion:** two lanes (name, address), each top-8, fused by reciprocal
  rank (offset 60); lanes with no terms are inactive (blank address ->
  address lane inactive).
- **Joint rescue lane:** one conjunction query per record — (rarest name
  terms) AND (rarest address terms) — top-2 results added to the fused
  pool when not already present. Screening showed +4.6pp link recall for
  ~1 extra candidate/query; a forward S1->target rescue lane and an
  anyascii transliteration lane were screened and rejected (weak or
  marginal at their cost).
- **Compaction:** strict reverse budget 2 (exact fused-score tie
  expansion) plus the bounded joint rescue; deterministic order
  (v desc, address rank, name rank, reference id).
- **Candidate volume (test):** 33,405,124 scored edges over 9,969,589
  queries (3.35 candidates/query; 19.3 per reference row; 813 references
  with zero candidates).
- **Measured support ceilings (macro-F0.5 oracle on the frozen support):**
  D 0.9848, K 0.9851, fresh A2 0.9853 — blocking loss 0.0147.

## 4. Matching Model

**Features used (37):**
- Name: JaroWinkler, Levenshtein-normalized, token Jaccard/overlap,
  exact-normalized, sorted-token exact, sorted-token JaroWinkler, length
  ratios/diffs, containment, shared-token smooth-idf weight (denominator =
  actual catalog size), script indicators.
- Address: JaroWinkler, Levenshtein-normalized, token Jaccard, exact,
  sorted-token exact, shared numeric tokens, numeric Jaccard, length
  ratios/diffs, target/reference missing flags.
- Cross-field: exact name + zero-shared-token address conflict; exact
  name + missing address.
- Retrieval context: fused v, rank, margin, top-1 flags, lane scores,
  pool size, joint-lane rank and score, source.
- Missingness is always distinct from conflict ("NA" is literal text).

**Model type:** LightGBM binary classifier (127 leaves, min 50/leaf,
max_bin 127, force_col_wise, lr 0.05, 1000 rounds, fixed seed) trained on
1,757,307 fit-cohort edges (20.9% positive) with real retrieved negatives
and competing references. Supervision is restricted to F-role targets;
background-owned positives are never labeled negative.

**Threshold selection method:** exact breakpoint sweep of the macro-F0.5
of complete reference rows (equal-score groups enter atomically; empty
prediction included) on the D cohort; Platt calibration fitted on
K-reference edges across complete per-source-calibrated traffic; final
tau=0.2584 (calibrated) frozen on K. The owner-exclusivity challenger
(margin-gated unique-top assignment) was evaluated on D and rejected
(bootstrap LCB -0.0004/-0.0005 macro-F0.5 in every configuration tried).

## 5. Results & Error Analysis

- **F_0.5 Score (macro, frozen policy, fresh identity-disjoint A2
  cohort): 0.9761** (D selection 0.9718, K calibration 0.9725; one-shot
  audit, policy frozen on K before audit).
- Slices (A2): US 0.9849, India 0.9629, singletons 0.9782, linked 0.9760.
- **Common false positives (wrong merges):** exact-name co-tenants with
  conflicting addresses; generic names ("Global Services") sharing common
  tokens; numeric agreement from unrelated house numbers.
- **Common false negatives (missed matches):** native-script targets whose
  references are transliterated Latin (India slice 2.2 points below US);
  targets with missing addresses and abbreviated names; references crowded
  out of lane top-8 by common-term noise. These residual misses motivate
  the multilingual-neural follow-up (below).

## 6. Conclusion

A compact, measured, CPU-only pipeline reaches 0.976 frozen-audit
macro-F0.5 with 3.35 candidates per query, complete official-validator
compliance, and end-to-end reproducibility from raw TSVs. Blocking loss
(0.0147) still exceeds scoring loss (0.0092); the residual misses are
transliteration, missing-address, and generic-name links that lexical
BM25 cannot see at this budget — the measured next step is a multilingual
neural retriever/cross-encoder for exactly those lanes.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — all source under `src/` (ber package:
records, normalize, splits, env, retrieval, candidates, features,
training, decisions, metrics, export, audit, pipeline, stages, cli) with
192 unit tests (exact scorer conventions, fusion/compaction, joint-lane
semantics, threshold sweeps vs brute force, cache invalidation, stale-
shard pruning, official-validator parity) and a README with stage-by-stage
reproduction commands. Entry point: `uv run python -m ber.cli <stage>`.
All shards are fingerprinted and resumable; mismatched fingerprints force
recomputation; every cache key binds the full input content (index,
traffic, frequency maps) and all runtime settings.

### B. Additional Results

Stage reports with timings and fingerprints: `data/reports/*.json`
(per-cohort retrieval ~900-1500s over ~640k queries, features ~15-90s,
evaluation 17-40s; full test retrieval ~9,600s over 9,969,589 queries,
features 195s, test scoring + export 271s on a 10-core laptop).
Validation: strict audit zero failures; official validator PASS with
`--check-ids` (1,732,544 required rows, 9,969,589 valid target IDs);
independent scored-support audit PASS (exported candidate file == the
exact set of pairs the model scored).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
