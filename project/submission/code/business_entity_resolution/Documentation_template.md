# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

The pipeline retrieves first: per-country tantivy BM25 indexes over the
Source-1 references, two field lanes (name, address) fused by
reciprocal rank, a strict reverse budget of 2 plus a joint name-AND-address
rescue lane. The candidates are then scored by a **two-stage LightGBM
cascade**: 46 pair features, then competition features derived from the
stage-1 scores.

Final matches come from an **owner-exclusive policy**: each target goes only
to its best reference, above τ. Everything is trained and selected inside a
**dense, test-faithful environment** in which every reference competes with
all its sibling and ownerless records at test density.

Public leaderboard: **0.952** (up from 0.815 for the earlier sparse-cohort
release). Both output files pass the official validator with
`--check-ids`, and no target is assigned to more than one reference.

**Key lesson (measured):** the original validation cohorts scored only
focal references and carried 0.3–0.6 targets per catalog reference (test:
5.76). That inflated the local score to 0.976 against 0.815 on the
leaderboard. Rebuilding validation to match test density closed the gap to
0.01 and drove every later gain.

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

**Approach type:** blocking + two-stage classifier (reverse retrieval,
owner-exclusive decisions).

**Core innovations:**

1. **Dense test-faithful environment `W`**:
   - The catalog per country is sized to test: all held-out D/K/A/A2 and F
     references plus B fill.
   - Traffic is every target owned by a catalog reference, plus ownerless
     orphans (unlinked targets and targets of references left out), up to
     the test density per catalog reference.
   - Result: US 5.76 targets/ref with 40% ownerless; India 5.10 with 32%.
   - Training rows are only edges whose reference and target are both
     F/B-role, which is 80% of all labels (the earlier model used 5%).
   - Policies are selected on D and checked on K; A2 is reserved.
2. **Owner exclusivity**, which matches the ground-truth constraint that
   each target has at most one owner.
3. **Stage-2 competition features:**
   - the target's best-other-reference gap
   - rank, count and number of high scores inside the target group and the
     reference group
   - computed from out-of-fold stage-1 scores (2 folds by reference)
4. **Sibling-discrimination features (v5):**
   - all digit runs, including single-digit house numbers
   - accent-folded Jaro-Winkler
   - a legal-form-free "name core" (Jaccard, exact, symmetric difference)

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

**Features (46, stage 1):**
- Name: Jaro-Winkler, Levenshtein-normalized, token Jaccard and overlap,
  exact, sorted-token exact and JW, length ratio and difference,
  containment, shared-token smooth-idf weight, script indicators.
- Address: JW, Levenshtein, token Jaccard, exact, sorted exact, shared
  numeric tokens and numeric Jaccard, length ratio and difference,
  missing flags.
- Cross-field: exact name with a conflicting address; exact name with a
  missing address.
- Retrieval context: fused RRF value, rank, margin, top-1 flags, lane BM25
  scores, pool size, joint-lane rank and score, source.
- v5 additions:
  - all-digit-run Jaccard, conflict flag, first-number equality and
    symmetric difference
  - accent-folded name and address JW
  - name-core Jaccard, exact match and symmetric-difference size

**v6 cross-script features (+4):** a rule-based Indic→Latin transliterator
(`ber/translit.py`) covers all nine Indic Unicode blocks through their shared
ISCII-parallel layout. A consonant skeleton (vowels dropped, phonetically
close consonants and voicing pairs merged) then gives name-skeleton Jaccard,
JW and core-Jaccard, plus address-skeleton Jaccard. This lets "अरिहंत
प्रोडक्ट्स प्राइवेट लिमिटेड" and "Arihant Products Private Limited" compare as
`arnt prdkts prvt lmtd`. India W-D rose from 0.939 to 0.947 at equal
strictness.

**Stage-2 features (+9):**
- stage-1 score
- target group: size, rank, gap to the best other reference, number ≥0.5
- reference group: size, rank, gap to its best target, number ≥0.5

**Models:**
- LightGBM binary (127 leaves, lr 0.05, max_bin 127, min 100/leaf).
- Stage 1: 500 rounds per fold, 2 folds split by reference, giving
  out-of-fold scores on the training rows and a fold average elsewhere.
- Stage 2: 400 rounds.
- Training set: 16.0M dense-env edges with real retrieved competitors;
  26.9M edges in total.

**Decision policy (selected on W-D, checked on W-K):**
- Owner-exclusive: a target is kept only for its unique top-scoring
  reference (margin γ=0.02 over the runner-up; exact ties abstain), and
  only if its score is ≥ τ.
- The W optimum is τ=0.707 (D 0.9601).
- The shipped τ=0.85 costs 0.003 on W (D 0.9567). It was chosen because the
  real test carries about 2× more edges in the uncertain score band, and it
  raised the public LB from about 0.88 to 0.947.
- An expected-F0.5 per-reference top-k policy was also tested. It tied on
  D but reduced singleton accuracy, so it was not shipped.

## 5. Results & Error Analysis

| Model / policy | W-D | W-K | US | India | Singletons | Public LB |
|---|---:|---:|---:|---:|---:|---:|
| v5 model, sparse-cohort policy | – | 0.898 | – | – | – | 0.815 |
| v5 model, owner policy | 0.938 | 0.939 | 0.956 | 0.913 | 0.905 | 0.877 (τ from a 20%-scale env) |
| dense model, stage 2, τ=0.707 | 0.960 | 0.960 | 0.972 | 0.943 | 0.967 | – |
| dense model, stage 2, τ=0.85 | 0.957 | – | 0.969 | 0.939 | 0.981 | 0.947 |
| **+ v6 transliteration features, stage 2, τ=0.875 (final)** | 0.960 | 0.961 | 0.969 | 0.947 | 0.986 | **0.952** |

- **Cross-scale robustness:** the frozen release, scored in a separate
  20%-scale dense env, reaches K 0.9716, against 0.974 when re-tuned
  there. This supports transfer to France's smaller, unlabeled catalog.
- **Candidate ceiling in W:** 0.985 (blocking loss 0.015).
- **Common false positives:** siblings at the same address (e.g. "Advik
  Properties" vs "Advik Farms"), generic names sharing common tokens, and
  adjacent house numbers. These are reduced by owner exclusivity and the
  name-core and number features.
- **Common false negatives:** native-script Indian targets against
  transliterated references (India is the weakest slice at 0.94), missing
  addresses, and references crowded out of the lane top-8.

## 6. Conclusion

The decisive factor was **validation that looks like test**. Once every
reference was scored against test-density competitors, local numbers
tracked the leaderboard to within 0.01, and three changes moved the score
from 0.815 to 0.947:
- owner exclusivity
- training on 80% of the labels
- the stage-2 competition features

Remaining headroom is mostly blocking (candidate ceiling 0.985) and the
India native-script slice.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — all source under `src/` (ber package:
records, normalize, splits, env, retrieval, candidates, features,
training, decisions, metrics, export, audit, pipeline, stages, cli, dense,
dense_run) with 200 unit tests (exact scorer conventions, fusion/compaction, joint-lane
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
