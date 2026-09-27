# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

A retrieval-first pipeline: country-partitioned tantivy BM25 indexes over
Source-1 references with merged trigram+word+accent-folded term streams,
two-lane (name/address) retrieval with reciprocal-rank fusion, strict
reverse budget 2, and a LightGBM pair classifier over 32 comparison
features, scored by a globally thresholded, Platt-calibrated decision
policy selected on held-out complete reference rows with the exact
macro-F0.5 metric. Frozen-policy audit macro-F0.5: **0.9378** (US slice
0.9501, India 0.9194); both submission files pass the official validator
with `--check-ids`.

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
**Core Innovation:** merged per-lane term streams + identity-component
focal/background validation cohorts at production catalog scale, with the
decision policy selected on the exact macro-F0.5 of complete rows.

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
  Devanagari/Tamil text byte-for-byte.
- **Query planning:** per lane, the rarest 16 known terms (document
  frequency from a per-country df map over both lanes' indexed streams);
  unknown terms are dropped (cannot match). Measured on the fit cohort:
  owner-recall cost <= 0.0023 for a 3.4x speedup (India 19.6 -> 5.75
  ms/query); India@24 slightly exceeded uncapped recall (0.9334 vs
  0.9300) because dropping common terms cleans the top-8 ranking.
- **Fusion:** two lanes (name, address), each top-8, fused by reciprocal
  rank (offset 60); lanes with no terms are inactive (blank address ->
  address lane inactive).
- **Compaction:** strict reverse budget 2 with exact fused-score tie
  expansion; deterministic order (v desc, address rank, name rank,
  reference id). Missing runner-up is not treated as a margin.
- **Candidate volume (test):** 10,947,202 candidates over 1,732,544
  references (mean 6.32/reference; 17,646 references with zero
  candidates); 23,167,551 scored edges total; 2.26 candidates per query.
- **Measured support ceilings (macro-F0.5 oracle on the frozen b=2
  support):** D 0.9627, K 0.9638, A 0.9633 — blocking loss 0.036-0.037.

## 4. Matching Model

**Features used (32):**
- Name: JaroWinkler, Levenshtein-normalized, token Jaccard/overlap,
  exact-normalized, sorted-token exact, length ratios/diffs, containment,
  shared-token smooth-idf weight, script indicators.
- Address: JaroWinkler, token Jaccard, exact, shared numeric tokens,
  numeric Jaccard, length ratios/diffs, target/reference missing flags.
- Cross-field: exact name + zero-shared-token address conflict; exact
  name + missing address.
- Retrieval context: fused v, rank, margin, top-1 flags, lane scores,
  pool size, source.
- Missingness is always distinct from conflict ("NA" is literal text).

**Model type:** LightGBM binary classifier (63 leaves, min 100/leaf,
max_bin 127, force_col_wise, lr 0.05, 300 rounds, fixed seed) trained on
2,934,053 fit-cohort edges (12.0% positive) with real retrieved negatives
and competing references.

**Threshold selection method:** exact breakpoint sweep of the macro-F0.5
of complete reference rows (equal-score groups enter atomically; empty
prediction included) on the D cohort; Platt calibration (a=-23.54,
b=3.41) fitted on K-reference edges across complete traffic; final
tau=0.5362 (calibrated) frozen on K. The owner-exclusivity challenger
(margin-gated unique-top assignment) was evaluated on D and rejected:
it traded 279 fewer false pairs for 782 more missed pairs (bootstrap
LCB -0.0009 macro-F0.5).

## 5. Results & Error Analysis

- **F_0.5 Score (macro, frozen policy, untouched A cohort): 0.9378**
  (D selection 0.9372, K calibration 0.9384; one-shot audit).
- Slices (A): US 0.9501, India 0.9194, singletons 0.9276, linked 0.9384.
- **Common false positives (wrong merges):** exact-name co-tenants with
  conflicting addresses; generic names ("Global Services") sharing common
  tokens; numeric agreement from unrelated house numbers.
- **Common false negatives (missed matches):** native-script targets with
  transliterated references (India slice 3 points below US); references
  whose lane top-8 was crowded out by common-term noise (pool sizes
  concentrate at 15-16); owner recall at b=2 measured 0.93 (India) /
  0.96 (US) on the fit cohort.

## 6. Conclusion

A compact, measured, CPU-only pipeline reaches 0.938 frozen-audit
macro-F0.5 with 6.32 candidates per reference, complete official-validator
compliance, and end-to-end reproducibility from raw TSVs. Blocking loss
(0.036) now exceeds scoring loss (0.025), so the highest-value next step
is retrieval (budget 3, adaptive compaction, or a trained multilingual
retriever), not a heavier matcher.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — all source under `src/` (ber package:
records, normalize, splits, env, retrieval, candidates, features,
training, decisions, metrics, export, audit, pipeline, stages, cli) with
168 unit tests (exact scorer conventions, fusion/compaction, threshold
sweeps vs brute force, official-validator parity) and a README with
stage-by-stage reproduction commands. Entry point:
`uv run python -m ber.cli <stage>`. All shards are fingerprinted and
resumable; mismatched fingerprints force recomputation.

### B. Additional Results

Stage reports with timings and fingerprints: `data/reports/*.json`
(ingest 118s, split 162s with leakage check passed over 7.6M truth edges,
per-cohort retrieval ~800s / features ~550s / eval 18-41s, full test
retrieval 5,329s over 9.97M queries, test scoring + export 62s).
Validation: strict audit zero failures; official validator PASS with
`--check-ids` (1,732,544 required rows, 9,969,589 valid target IDs).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
