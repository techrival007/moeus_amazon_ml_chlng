# Amazon ML Challenge 2026 — Master Solution and Execution Plan

**Revision:** 2.1 — independent review, actual-data audit, and adversarially revised execution contract.  
**Review date:** 2026-09-26.  
**Scope:** planning and evidence. No competition model has been trained or evaluated by this revision.  
**Authority:** the official statement and the newer candidate-generation ranking update supplied with the review request.  
**Inputs inspected:** the complete previous `master.md`, all seven supplied TSVs, `student_resource/README.md`, the documentation template, and the organizer's validator.

## Read this first

**Build a compact, retrieval-first resolver with a supervised tabular matcher.** Index the deduplicated Source 1 records, retrieve plausible reference owners for each Source 2/3 record, compact that retrieval output before classification, then transpose the retained edges into the required per-Source-1 candidate lists. Train LightGBM on identity-specific comparison features and real retrieved negatives. Select matches using validation of the actual macro-F₀.₅, including empty lists and retrieval misses.

The immediate neural experiment is **small multilingual retrieval**, especially for native-script Indian records and ambiguous lexical searches. The next is **a compact multilingual cross-encoder on difficult candidates**. Neither a generative LLM nor graph clustering is a prerequisite for a strong submission.

This recommendation is a build-first decision, not a claim that a tabular model will beat every neural model. The experiments below determine whether to replace or augment it. Complexity must improve the measured accuracy–candidate-size–runtime trade-off.

### The facts that changed the plan

| Finding from the supplied data | Consequence |
|---|---|
| 2,206,821 training reference entities; 1,732,544 test reference entities | Design for millions of entities, not a hypothetical 20–100K test set. |
| 9,969,589 test S2/S3 records | An all-pairs test join contains **17,272,751,604,416 pairs**. Indexed blocking is essential. |
| 7,638,365 labeled training links; each target occurs under exactly one reference | Reverse retrieval has a natural zero-or-one-owner interpretation, while a reference may receive many targets. |
| Only 123,247 training references are singletons: **5.58482%** | Empty predictions matter, but a speculative 35–50% singleton prior is inappropriate. |
| Training references have up to **11** matches; 4,776 have more than eight | An eight-candidate selection limit can lose true matches before modeling begins. |
| France is **14.97520%** of test references; India is **46.75125%** | France needs genuine transfer safeguards; native-script Indian matching is also a major priority. |
| A fixed 50-candidate forward policy yields **86,627,200** final pairs | Candidate compaction is both a ranking objective and an engineering requirement. |

**No challenge score forecast is provided.** Published pairwise F1, accuracy, retrieval recall, or language-benchmark results do not establish this competition's end-to-end macro-F₀.₅.

### Navigation

1. [Decision and scope](#decision)
2. [Review of the previous plan](#previous-plan)
3. [Rules and precedence](#rules)
4. [Observed dataset](#data)
5. [Formulation and metric](#metric)
6. [Candidate-size objective](#candidate-objective)
7. [Validation design](#validation)
8. [Record representations](#representations)
9. [Candidate discovery and compaction](#retrieval)
10. [Matching model and training](#matcher)
11. [Calibration and decisions](#decisions)
12. [Multilingual generalization](#generalization)
13. [Neural experiments](#neural)
14. [Experiment and error-analysis loop](#experiments)
15. [Compute and scaling](#compute)
16. [Implementation contracts and execution order](#execution)
17. [Compliance, export, and reproduction](#submission)
18. [Resolved handoff questions](#handoff)
19. [Adversarial review and future work](#adversarial)
20. [Research evidence and references](#evidence)
21. [Audit provenance](#provenance)

---

<a id="decision"></a>
## 1. Decision and scope

### 1.1 What we should build first

```text
Provided TSVs + immutable ground truth
        |
        +--> validation partitions and exact scorer
        |
        +--> raw-preserving record views
                 |
                 +--> country-partitioned S1 inverted indexes
                           |
          each S2/S3 record --> multi-view reference retrieval
                           |
                           +--> deterministic retrieval compaction
                                      |
                                      +--> frozen candidate edge relation C
                                      |      +--> candidate_pairs.tsv
                                      |
                                      +--> comparison features
                                               |
                                               +--> LightGBM pair scores
                                                        |
                                                        +--> calibrated decision policy
                                                        +--> reference-owner conflict policy
                                                                 |
                                                                 +--> matching_results.tsv
                                                                 +--> strict audit + package
```

The candidate edge relation is the shared contract between blocking, modeling, evaluation, and output. It is not reconstructed from accepted predictions.

**First runnable configuration:** two lexical field lanes, eight raw hits per active lane, reciprocal-rank fusion with offset 60, strict reverse budget **b=2**, one LightGBM classifier, and a global threshold learned from complete development rows. Budgets b=1/b=3, adaptation, owner handling, and neural components are controlled challengers. The initial b=2 choice is an engineering starting point; it is not the asserted final optimum.

### 1.2 Priority classes

| Class | Work | Admission rule |
|---|---|---|
| **Core** | Data contracts, scorer, leakage-safe splits, indexed lexical retrieval, compact candidates, LightGBM, threshold validation, complete exports, strict audit | Establish a reproducible end-to-end baseline immediately. |
| **High-value experiments** | Reverse-versus-forward retrieval at equal cost; adaptive candidate budgets; small multilingual dense retrieval; supervised retrieval tuning; selective multilingual cross-encoder; simple row-aware decisions | Promote only on held-out evidence and measured resource use. |
| **Optional exploration** | General F-measure Maximizer; nullable listwise ranking; pre-matching graph-assisted retrieval; local small-LLM adjudication; distillation | A specific residual error pattern must justify the experiment. |
| **Fallback** | Validated lexical blocker + one LightGBM model + simple threshold policy | Always retain a complete, runnable version. A GPU failure must not prevent submission. |

### 1.3 Confidence labels

- **Observed:** measured from the supplied files during this review.
- **Derived:** arithmetic or a mathematical result with stated assumptions.
- **External evidence:** a result verified in the cited source, under that source's protocol.
- **Hypothesis:** an improvement worth testing; no gain is promised.
- **Engineering default:** a concrete starting configuration or acceptance policy, not a learned truth or organizer rule.

The final architecture is deliberately extensible through a few well-defined interfaces. It is not an obligation to implement every experimental branch.

---

<a id="previous-plan"></a>
## 2. Review of the previous plan

The previous document was read in full before this replacement was designed. Its useful instincts survive; its unsupported commitments do not.

### 2.1 What it understood correctly

1. This is noisy identity resolution, not semantic similarity alone.
2. Blocking recall limits the matcher; string and learned signals may be complementary.
3. The official metric is per-reference macro-F₀.₅ with a special empty-set reward.
4. Multiple matches and singletons must both be supported.
5. France is absent from labeled training, so country-closed encodings are inappropriate.
6. Model licenses, actual parameter counts, output integrity, and reproducibility matter.
7. Entity-disjoint validation, hard negatives, ablations, and inference rehearsal are valuable.

### 2.2 What it misunderstood, overstated, or failed to establish

| Previous claim/design | Review conclusion | Replacement |
|---|---|---|
| Candidate size mainly affects runtime; roughly 40–50 candidates is a suitable default | Superseded by the new final-ranking rule; also too expensive to assume at this scale | Measure a compact-candidate Pareto frontier. |
| Training data is unavailable | The complete resource bundle is now present | Ground the design in the measurements in §4. |
| France's share is unknowable; assume one third | Country counts are observable without test labels | Use the observed 259,452/1,732,544 reference share. |
| Singleton scenarios around 35–50% motivate strong conservatism | Actual training rate is 5.58482% | Tune decisions from complete entity outcomes, not an invented prior. |
| Forecast 0.90–0.96 in-domain and 0.84–0.93 France macro-F₀.₅ | Different metrics, datasets, and evaluation protocols do not support these forecasts | Remove them; report measured results only. |
| A global threshold is “provably suboptimal,” implying a 1–3 point gain from the proposed decoder | It is not universally Bayes-optimal, but can be optimal in particular distributions and can beat a misspecified complex decoder | Keep it as a serious baseline. Validate alternatives. |
| Ratio-of-expected-counts selection handles singletons optimally | It can prefer a match even when the correct expected-score decision is empty | Corrected derivation and counterexample in §5. |
| Sorted marginal prefixes remain sufficient with correlated truths | False in general | Use the appropriate conditional joint statistics if testing GFM. |
| Monte Carlo automatically handles sibling correlation | Sampling requires a specified, learned joint model; otherwise it merely repeats an assumption | No production Monte Carlo decoder in the core. |
| ComEM proves the proposed multi-ID listwise LLM strategy | Its selection formulation assumes at most one matching candidate | Treat multi-match prompting as an unvalidated adaptation. |
| The +16.02 ComEM result is an open-7B gain | In the cited table it averages gains for GPT-3.5 Turbo and GPT-4o-mini | Retain the correct context in §20. |
| Generated ID-token logits are calibrated per-candidate match probabilities | They are next-token probabilities affected by output history and tokenization | Use explicit binary scores or classification heads and held-out calibration. |
| “Synthetic France” can calibrate French performance | Accent insertion and postcode replacement do not create representative French entities | Use label-preserving corruption stress tests, explicitly not French estimates. |
| US↔India transfer bounds the France gap | It provides neither an upper nor a lower bound | Treat country holdout as a stress test. |
| Lower France's match prior because it is unseen | No supporting evidence | Use a pooled policy unless a valid experiment justifies otherwise. |
| Never use thresholds in blocking | A study of global similarity cutoffs does not prohibit validated, adaptive multistage filtering | Compare top-b retrieval and retrieval-only pruning empirically. |
| A SIFT1M HNSW result guarantees this task's ANN recall | Vector-search recall depends on this index and embedding distribution; it is not ER recall | Measure ANN fidelity and true-link recall separately. |
| A passing supplied validator proves all rules | ID existence is off by default; missing candidates and subset violations can be warnings | Add strict checks and use `--check-ids`. |
| Each member under 8B makes any ensemble certainly compliant | Per-model versus aggregate interpretation is unspecified; the old neural stack totals about 8.46B | Keep the total deployed learned stack below 8B by default. |
| All software dependencies must be MIT/Apache | The stated restriction concerns the final model, not an invented ban on BSD software | Audit model and dependency licenses separately. |

One reference is demonstrably mismatched: arXiv `2608.18115` is **Temporal Multi-Signal Fusion for Token-Level Hallucination Detection**, not the entity-matching stability study described in the old document. Its claimed 99.7% agreement and associated improvement figures are withdrawn. [BADREF]

### 2.3 Where it was over-engineered

- Mandatory BGE-M3, cross-encoder, 7B generative model, calibration ensemble, within-source clustering, graph cleanup, and expensive expected-score decoding before any local baseline existed.
- Treating every possible technique as a stage in the final system.
- Exponential truth enumeration or 10,000-sample Monte Carlo per reference without a corpus-level runtime calculation.
- Elaborate French-looking augmentation before inspecting the actual French input distribution.
- Many historical competition scores from unrelated tasks, used as if they predicted entity-resolution improvements.

### 2.4 Where it was under-engineered

- Actual data scale, incoming-candidate skew, memory, and output sizes.
- Reverse retrieval into the deduplicated reference catalog.
- Adaptive candidate compaction under the updated ranking rule.
- Validation leakage through target siblings and negative-pair endpoints.
- Natural candidate distributions after sampling and their effect on calibration.
- The difference between a true singleton and a reference whose matches were missed by blocking.
- Native-script preservation: unconditional removal of combining marks can damage Indic text.
- Exact matching-model support across cascades, reruns, and graph expansion.
- Error attribution and statistical comparison of complete entity outcomes.

### 2.5 What we would do differently from a fresh start

Start with data contracts, a scorer, a compact indexed search experiment, and one supervised matcher. Use the deduplicated reference direction, evaluate actual hard negatives, and spend complexity on the largest measured source of loss. The strongest omitted avenue is **learning to retrieve the correct reference in very few candidates**, because it can improve matching, ranking, and compute simultaneously.

---

<a id="rules"></a>
## 3. Rules, precedence, and non-negotiable contracts

### 3.1 The ranking update takes precedence

The newer official update supplied in the review request states:

> Candidate generation counts toward the final ranking. We will review your candidate_pairs.tsv and the code that produces it when deciding final rankings, alongside your matching_results.tsv score. The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard.

Therefore:

- `matching_results.tsv` still drives the public/private leaderboard scalar.
- Final evaluation also considers candidate size and the code's scalability.
- The update does **not** publish a mathematical weighting, a lexicographic ordering, or an allowed score sacrifice.
- Do not invent an official combined score. Maintain a measured frontier and use the internal selection policy in §6.
- The older local README and PDF remain useful for schemas and metric details; their omission of the ranking update does not override it. [PS]

### 3.2 Required inputs

Current paths are under `student_resource/dataset/`:

```text
train/train_source1.tsv
train/train_source2.tsv
train/train_source3.tsv
train/train_ground_truth.tsv
test/test_source1.tsv
test/test_source2.tsv
test/test_source3.tsv
```

Each source has `entity_id`, `business_name`, `business_address`, `country`. Source comes from the file and ID prefix. Ground truth has `source1_entity_id`, `matched_entity_ids`.

Use explicit TSV parsing and preserve literal strings and empty values. For pandas, use `sep="\t", dtype=str, keep_default_na=False`. The implementation may prefer streaming CSV/Arrow, but must preserve the same semantics.

### 3.3 Required outputs

| File | Exact header | Meaning |
|---|---|---|
| `output/matching_results.tsv` | `source1_entity_id<TAB>matched_entity_ids` | Selected target IDs, comma-separated; empty field for no match |
| `output/candidate_pairs.tsv` | `source1_entity_id<TAB>candidate_entity_ids` | Exact final candidate support presented to the matching pipeline |

For both:

1. Exactly one row for every test S1 ID, including France and empty lists.
2. Only existing test S2/S3 IDs inside lists.
3. No duplicate S1 rows or repeated IDs within a list.
4. Matches must be a subset of candidates for the same S1.
5. UTF-8, literal tab separators, comma-separated IDs, no quoting of ID lists.
6. Preserve IDs exactly. Their numeric suffixes are opaque identifiers, not predictive features.

### 3.4 Resource and fair-play constraints

- The final model must have an MIT or Apache-2.0 license and at most 8 billion parameters.
- No external business-identity lookup, external entity-resolution services, geocoding, registries, or internet-derived business augmentation.
- Internet research about methods and licenses is distinct from looking up challenge entities. No challenge business identity was searched online during this review.
- Training uses the supplied supervision. Pretrained eligible model weights are a documented interpretation of the model rule; the CPU core does not depend on that interpretation.
- Both outputs, runnable code, pinned environment, and the filled methodology template belong in the final package.

---

<a id="data"></a>
## 4. What the supplied dataset actually contains

### 4.1 Full-scan inventory — observed

Counts below exclude headers. They were computed by streaming the TSVs, not extrapolated from file sizes.

| Split | Source | US | India | France | Total |
|---|---|---:|---:|---:|---:|
| Train | S1 | 1,323,633 | 883,188 | 0 | **2,206,821** |
| Train | S2 | 3,016,817 | 2,017,799 | 0 | **5,034,616** |
| Train | S3 | 3,170,056 | 2,115,547 | 0 | **5,285,603** |
| Test | S1 | 663,106 | 809,986 | 259,452 | **1,732,544** |
| Test | S2 | 1,871,330 | 2,312,565 | 703,378 | **4,887,273** |
| Test | S3 | 1,945,701 | 2,405,000 | 731,615 | **5,082,316** |

Training contains 12,527,040 source records; test contains 11,702,133. These are distinct from the additional 2,206,821 ground-truth rows.

Test reference weights are US 38.27355%, India 46.75125%, France 14.97520%. Public/private weights need not equal these full-test proportions.

### 4.2 Ground-truth structure — observed

| Quantity | Value |
|---|---:|
| Ground-truth reference rows | 2,206,821 |
| True singleton references | 123,247 |
| Singleton rate | 5.58482% |
| Total positive links | 7,638,365 |
| Mean true links per reference, including singletons | 3.461253 |
| S2 positive links | 3,693,619 |
| S3 positive links | 3,944,746 |
| S2 records unlinked to any training reference | 1,340,997 |
| S3 records unlinked to any training reference | 1,340,857 |
| Unlinked fraction of all training targets | 25.98641% |
| Targets appearing under multiple ground-truth references | **0** |
| Ground-truth links with different reference/target countries | **0** |

The full integrity scan found no duplicate ground-truth rows, duplicate IDs within a truth list, unknown truth references, missing truth references, invalid target prefixes, or unresolved positive target IDs.

Country-level truth counts:

| Country | References | Singletons | Positive links |
|---|---:|---:|---:|
| US | 1,323,633 | 73,896 | 4,578,522 |
| India | 883,188 | 49,351 | 3,059,843 |

**Interpretation:** many source records represent the same reference business, but each target record has one labeled reference owner at most. This is not one-to-one matching between S1 and S2, or between S1 and S3.

### 4.3 Match cardinalities — observed

| True matches per S1 | Number of S1 entities |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

S2 alone contributes as many as five matches to one reference; S3 contributes as many as six. There are 1,129,968 references with multiple S2 matches and 1,224,128 with multiple S3 matches.

**Do not hard-code 11 as a test-time maximum.** It is a training observation, whereas the official contract is zero, one, or many.

### 4.4 Missingness and script differences — observed

Business names are nonempty in every scanned source row. S1 addresses are nonempty in both train and test. Target addresses are sometimes empty:

| Split/source | Country | Empty addresses | Names containing Indic-block characters |
|---|---|---:|---:|
| Train S2 | US | 111,121 | 0 |
| Train S2 | India | 57,846 | 474,345 |
| Train S3 | US | 110,968 | 0 |
| Train S3 | India | 64,948 | 278,524 |
| Test S2 | US | 55,107 | 0 |
| Test S2 | India | 52,764 | 546,606 |
| Test S2 | France | 21,537 | 0 |
| Test S3 | US | 55,317 | 0 |
| Test S3 | India | 59,240 | 320,639 |
| Test S3 | France | 21,541 | 0 |

Here “Indic” means a character in Unicode U+0900–U+0DFF, a reproducible script-range diagnostic, not automatic language identification.

All training S1 business names are ASCII. Hundreds of thousands of Indian target names and addresses are not. In test India, S2 has 550,573 addresses containing Indic-block characters and S3 has 550,483.

France S1 contains 40,789 non-ASCII names and 73,335 non-ASCII addresses. Ordinary accent variation is therefore directly relevant, but accent removal must not destroy other scripts.

Length measurements also matter: training US S1 addresses have mean 34.96 characters and p99 55; India has mean 77.70 and p99 134. These are **characters, not tokenizer tokens**. Benchmark actual tokenizer lengths before choosing neural sequence limits.

### 4.5 Deterministic matched-pair sample — observed, limited scope

Sample rule:

```text
int(sha256(source1_entity_id).hexdigest()[:16], 16) % 200 == 0
```

This selected 10,996 references: 6,530 US and 4,466 India, containing 37,937 positive links and 629 singletons. The hash is used for reproducible sampling only, never as a model feature.

The sample's comparison normalization was NFKC + case folding + punctuation-to-space, preserving Unicode letters, numbers, and combining marks. It was not a learned resolver.

- **4,308 sampled reference names** were nonunique under that normalization in the complete training reference catalog: **39.17788%**.
- **614 sampled reference addresses** were nonunique: **5.58385%**.
- No duplicate raw `(business_name, business_address, country)` payloads were found within the full training S1 file.
- No test S1 raw payload exactly equaled a training S1 raw payload.
- In the Indian positive-pair sample, 2,092/7,437 S2 pairs and 1,648/8,090 S3 pairs shared no normalized name token.
- 403 sampled Indian S2 pairs and 251 S3 pairs had Indic characters in both target name and address.
- No sampled positive pair had zero token overlap in **both** fields. Shared digits and common words count as tokens in this diagnostic; it is **not** a retrieval-recall result.

**Consequences:** name equality alone is not safe; address equality alone is not universally safe; joint evidence is valuable; native-script mismatches deserve a retrieval experiment; repeated common tokens must be down-weighted.

### 4.6 What remains unmeasured

No production retrieval recall, candidate budget, matcher quality, calibration quality, ANN fidelity, neural throughput, or leaderboard score has been measured in this revision. The full target-to-target exact-duplicate distribution, information-theoretic label ambiguity, and identity overlap under aggressive normalization are also unmeasured.

The next stage measures these properties where they affect a concrete decision. It does not invent them to justify an architecture.

---

<a id="metric"></a>
## 5. Formulation and the exact competition objective

### 5.1 Identity formulation

Let `A` be S1 and `B` be the union of S2/S3. For each reference `a`, truth is a set `T_a ⊆ B`; prediction is `S_a ⊆ C_a`, where `C_a` is its actual candidate support.

Equivalently, each target record seeks a reference owner or no owner:

```text
g(b) ∈ A ∪ {NONE}
```

The inverse formulation is supported by deduplicated S1 and the observed absence of reused target links. It does not turn the evaluation into target-level accuracy: final selection is still judged by per-reference F₀.₅.

No reference capacity of one is allowed. A conventional one-to-one Hungarian assignment would contradict the observed and official label structure.

Business identity is whatever the supplied labels define. Do not assume every same-brand branch is the same entity, or every different address implies a different legal business. Investigate such cases in training errors.

### 5.2 Scorer

For one reference, let `t = len(T)`, `s = len(S)`, `tp = len(T ∩ S)`:

```text
F(T,S) = 1                         if t = 0 and s = 0
       = 5*tp / (t + 4*s)          otherwise

macro_F = sum(F(T_a,S_a) for every required reference a) / number_of_references
```

This is equivalent to:

```text
F = 1.25*TP / (0.25*t + s)
  = 5*TP / (5*TP + 4*FP + FN)
```

The count form assigns coefficient four to FP relative to FN in the denominator. The statement's “precision 2×” explanation is informal; neither wording nor these coefficients imply a universal probability threshold of 0.8 or a constant cost per error.

| Truth | Prediction | Exact score |
|---|---|---:|
| empty | empty | 1 |
| empty | one false ID | 0 |
| one true ID | empty | 0 |
| `{a}` | `{a,x}` | 5/9 ≈ 0.555556 |
| `{a,b}` | `{a}` | 5/6 ≈ 0.833333 |
| `{a,b}` | `{a,b,x}` | 5/7 ≈ 0.714286 |
| `{a,b,c}` | `{a,b,x}` | 2/3 ≈ 0.666667 |

The all-empty prediction has training score **0.0558482**, derived from the observed singleton fraction. That is a baseline, not a score floor or a guarantee about test.

Important details:

- Compute the mean of entity scores, not F₀.₅ from pooled pair counts or from averaged precision and recall.
- Include references whose candidate set is empty.
- Keep truth complete when scoring a smaller candidate set; unretrieved truths remain false negatives.
- Diagnostics by source are useful, but the official row score uses the union of S2/S3 predictions.
- Duplicate or unknown IDs are data-contract violations; do not silently repair malformed submissions in the scorer.

### 5.3 The most useful blocking ceiling

For fixed candidates, let `r_a = len(T_a ∩ C_a)`. The best possible matcher restricted to those candidates predicts exactly the retrieved truths:

```text
oracle_a = 1                                  if len(T_a) = 0
         = 5*r_a / (len(T_a) + 4*r_a)          otherwise
oracle_macro = mean(oracle_a)
```

This is a metric-aligned support ceiling. It can differ substantially from micro pair recall.

Use the decomposition:

```text
blocking_loss = 1 - oracle_macro
scoring_and_decision_loss = oracle_macro - actual_macro_F
```

Both are nonnegative under valid fixed-support outputs. They tell us whether to improve retrieval or matching next.

### 5.4 Correcting the previous expected-score decoder

The previous approximation used:

```text
1.25 * sum(selected marginal probabilities) /
    (0.25 * sum(all candidate marginal probabilities) + number selected)
```

This is a **ratio of expectations**, not the expectation of the score. It also ignores true matches outside the candidate set.

Counterexample, verified by exact rational calculation:

- One candidate matches with probability 0.48; otherwise the reference is a true singleton.
- Predicting empty has expected score **0.52**.
- Predicting the candidate has expected score **0.48**.
- The old nonempty approximation returns **0.535714**, incorrectly exceeding even the correct empty utility.

A separate issue is dependence. Two candidates can each have marginal probability 0.4 while their optimal decision differs:

| Joint truth distribution | Best action |
|---|---|
| Empty with probability .6; both true with probability .4 | Empty: expected score .6 |
| Empty with probability .2; exactly either candidate with probability .4 each | Both: expected score 4/9 ≈ .444444 |

Perfectly calibrated marginal scores alone do not determine the optimal action.

### 5.5 A valid advanced formulation: GFM

General F-measure Maximization is established research, not a new heuristic. [GFM1] [GFM2] [GFM3]

Let `M = |T|` include all true matches, including those outside `C`; `Y_i` indicates whether candidate `i` is true. For prediction size `k > 0`:

```text
Delta(i,k) = sum over t>=1 of:
             1.25 * P(Y_i=1, M=t | observed context) / (0.25*t + k)

E[F(S,T) | context] = sum(Delta(i,k) for i in S), when len(S)=k
```

For each `k`, choose the `k` largest `Delta(i,k)` values, then compare those actions with empty, whose utility is `P(M=0 | context)`. The optimal sets need not be nested across `k`, and the ranking is not generally the ranking of marginal match probabilities.

Given these weights, decoding is polynomial. Under independent Bernoulli truths, probability-ranked prefixes and quadratic dynamic programming can compute expected F without exponential enumeration. [YE]

**Why this is optional:** estimating reliable joint count statistics is a genuine learning problem. Candidate omissions, correlated aliases, country shift, and owner constraints can invalidate simplified assumptions. Exact optimization of a wrong posterior is not a performance guarantee. Candidate-wise GFM also does not by itself solve globally coupled owner conflicts.

Do not deploy this decoder until it beats the simpler policies in §11 under held-out evaluation. No Monte Carlo loop over millions of references belongs in the core.

---

<a id="candidate-objective"></a>
## 6. Optimize candidate quality and compactness together

### 6.1 Measurements required for every candidate policy

Report on the same held-out references and complete truth:

1. Final macro-F₀.₅.
2. Candidate-support oracle macro-F₀.₅.
3. Micro link recall: retained truth links / all truth links.
4. Macro candidate recall among non-singletons.
5. Fraction of non-singletons retaining at least one true candidate.
6. Fraction retaining every true match.
7. Total candidates and mean per **all** S1 rows, including empty rows.
8. Median, p90, p95, p99, maximum candidate count; fraction of empty candidate rows.
9. Candidate precision on labeled data.
10. Retrieval time, feature time, scoring time, peak RAM/VRAM, and index size.
11. The above by country, source, truth-cardinality bucket, missing address, script, and ambiguity.

Define global reduction ratio against `N1*(N2+N3)`. If a within-country denominator is also reported, label it separately. At this scale almost every reasonable blocker has a reduction ratio near one; mean candidate count is more interpretable.

### 6.2 Why reverse budgets are attractive

For test, `M = N2+N3 = 9,969,589`, `N1 = 1,732,544`. Retaining at most `b` references per target gives at most `b*M` edges before any additional rescue lane:

| Reverse budget per target | Maximum pairs | Mean per S1, if every target fills its budget |
|---:|---:|---:|
| 1 | 9,969,589 | 5.7543 |
| 2 | 19,939,178 | 11.5086 |
| 3 | 29,908,767 | 17.2629 |
| Forward fixed 50 per S1 | 86,627,200 | 50.0000 |

These are arithmetic bounds, **not recall claims or final selected budgets**. Reverse retrieval can issue more queries than forward retrieval; query count and total latency must be compared.

It allows an S1 entity to collect all of its target variants without a small per-S1 cap. Its weakness is that errors for different variants can accumulate, and popular references can receive large incoming lists. Report that tail rather than hiding it with an unmeasured truncation.

### 6.3 Internal selection policy, until an official ranking formula exists

Keep all nondominated configurations in `(macro_F, mean_candidates, runtime)` space. Never turn candidate count into a fictional official score.

**Engineering default:** register one reference run before a compaction experiment: the best development configuration among the initial uniform-budget CPU baselines. Keep that run fixed for the comparison. Accept a smaller policy only when the **one-sided 95% lower confidence bound** on `F_compact - F_reference` is at least **-0.001 absolute F**. Use paired bootstrap resampling of reference identity components, 2,000 replicates, and a fixed recorded seed. This 0.001 is a chosen tolerance, not a predicted improvement or organizer threshold.

Register guard slices before looking at the new result: US, India, true singletons, rows with a true Indic-script target, and rows with a true target whose address is missing. Slice membership comes from immutable input/truth metadata, not the changing candidate policy. Require the corresponding one-sided lower bound to be at least **-0.003** for slices with at least 500 reference rows. Smaller slices remain explicitly inconclusive and require error inspection; they must not be advertised as proven non-inferior. An inconclusive overall comparison retains the reference run. The independent final audit checks the selected configuration; repeated development comparisons are not claimed to have simultaneous 95% coverage.

For candidate compaction specifically, track the oracle loss relative to a broader diagnostic pool. An initial internal loss budget of 0.001 macro-F is a useful strict gate; failure means investigate misses, not relabel the result as satisfactory.

If no compact configuration satisfies the gate, keep the higher-quality validated configuration and continue improving retrieval. Do not sacrifice large amounts of matching quality based on an unknown ranking weight.

### 6.4 Honest candidate accounting

Let `H` be a broad indexed retrieval pool and `C` the final candidate relation.

- Retrieval-only filtering may compact `H` into `C` before matching, as the statement permits.
- Once a pair is scored by the matching classifier in a submission run, it belongs in that run's declared matching support.
- If LightGBM scores `C` and a neural model scores only an uncertain subset, export **C**, not just the neural subset.
- Do not score 50 candidates, keep two winners, and call those two the blocking output.
- If an iterative rescue adds scored pairs, the declared support is the union of the pairs actually scored in that final run.
- Separate experimental runs have separate manifests. The final files must describe the actual selected run.

The core avoids an ambiguous learned pair-classifier “blocker.” Supervised record-embedding retrieval is a legitimate experimental blocking method; rebranding the same pair classifier as a filter solely to shrink the reported candidate file is not the design.

---

<a id="validation"></a>
## 7. Validation: prevent leakage and preserve retrieval difficulty

### 7.1 Split real identity groups, not pair rows

Construct components from each S1 and **all of its labeled S2/S3 targets**. Assign the entire component to one partition. Every augmentation descendant inherits that partition.

For the protected focal holdouts `D/K/A` defined below, the disjointness rule protects both endpoints. Reusable background records have a separate, explicit role:

- A held-out target must not appear as a negative training pair for another reference.
- A held-out reference must not appear in a labeled training pair.
- All matched siblings remain together even if a particular blocker cannot retrieve them.
- Target-only records unlinked to S1 remain necessary distractors. Partition them too; keep exact duplicate target payloads together when detected.
- Do not join components using all candidate edges or merely a shared generic name; that can create giant meaningless components.

Reliable near-identical reference families may be grouped for an additional leakage stress test without changing their official truth labels.

### 7.2 Separate focal holdouts from catalog background

A split into small independent catalogs changes retrieval competition and makes score margins difficult to transfer. A larger final audit alone does not repair thresholds selected on a much smaller development catalog.

**Engineering default for the first controlled study:** assign identity components to five roles, stratified by country and truth-size bucket:

| Role | Initial share | Purpose |
|---|---:|---|
| `F` — fitting | 5% | Supervised positive groups and fitting queries; approximately 110K references and 382K positive links |
| `D` — development | 5% | Feature/model/policy selection on unseen focal entities |
| `K` — final calibration | 5% | Fit calibration and thresholds for the exact selected release checkpoint |
| `A` — locked audit | 5% | Final untouched focal outcomes |
| `B` — reusable background | 80% | Large realistic reference catalog and background target traffic |

Partition unlinked target groups too. Protect every endpoint in `D`, `K`, and `A`: none may appear in supervised fitting pairs, and one focal holdout's targets are not reused in another focal holdout's evaluation traffic. Near-identical payload groups that would undermine this protection remain together.

`B` is an explicitly reusable background catalog, **not a scored held-out entity cohort**. Its references may be negative counterparts for fitting queries, labeled using the fitting query's complete truth. Its own positive identity labels do not train the matcher. Its target records supply incoming distractor traffic during evaluation. This intentional background reuse must be disclosed; the reported generalization claim concerns the protected focal cohorts, not every record in the search environment.

There is ample supervision even at the initial 5% fitting share. Test a learning curve before growing it. If a larger fitting pool is justified, move complete background components into fitting in a registered new manifest while preserving `D/K/A`; compare models on a fixed common background-target panel. Do not automatically consume calibration or audit labels for a final refit.

### 7.3 Production-like catalog and traffic construction

For a focal cohort `G` in `D/K/A`:

1. Index **all focal references** plus a fixed, label-blind background sample from `F ∪ B`. Exclude the other protected focal cohorts from this primary evaluation catalog.
2. Choose country-specific index sizes from the observed test-reference counts, limited by the smallest available focal-plus-background catalog across the comparisons. Use the same target size per country for development, calibration, and audit. Never drop a focal reference to hit a quota.
3. Reserve every target owned by `G` and its assigned unlinked targets first. Add a fixed country/source-stratified sample of `B` targets as incoming background traffic, filling the remaining observed test-traffic quota when available. A quota limits only added background: retain all mandatory focal targets even if they exceed it. Assert that every focal truth target occurs in the query manifest. Do not synthesize duplicate rows to fill a shortage.
4. A background target whose owner is outside the indexed catalog is a legitimate no-owner distractor. Record the owner-present fraction, unmatched traffic, and any shortfall; these affect difficulty.
5. Run retrieval and the final owner/decision policy over this whole environment. Compute the reported macro-F and candidate statistics on **focal references only**. False links from background target queries into a focal reference must be included.
6. Record full-environment candidate counts, query counts, and runtime too. A focal score must not conceal the cost of processing the background.

This makes the scoring cohort relatively small while the search catalog remains large. With the initial roles, a US reference catalog can reach the observed test size. The available Indian catalog for a full 5% focal cohort is roughly 90% of training India S1, about 795K versus 810K in test. Cross-fitted development subcohorts may require a slightly smaller common catalog; exact sizes come from the manifest and are held comparable across policy comparisons. The available Indian target traffic is also smaller than test. Report these residual gaps and run a frozen full-reference-catalog pressure diagnostic; do not claim exact distribution matching.

For fitting queries, retrieve against fitting plus background references at comparable country-specific sizes. Use only fitting query truth to label training pairs. Protected `D/K/A` records never become hard negatives for fitting. This supplies realistic competing-reference negatives without feeding holdout identities to the learner.

Index statistics computed without labels are part of the inference algorithm. Supervised normalization maps, retrieval fine-tuning, and feature/model learning use fitting supervision only. An additional fully disjoint, small-catalog check can diagnose background memorization, but is not the sole source of deployment thresholds.

### 7.4 Development, final calibration, and release checkpoint

1. Fit candidate models on `F`; use grouped inner folds if a stacker or confidence router requires training scores.
2. Select model family, fitting sample size, candidate policy, and decision-policy family on `D` in the production-like environment above.
3. Freeze the actual release matcher checkpoint, normalization/encoder versions, candidate-generation algorithm, and decision-policy family.
4. Generate **natural candidates using that exact frozen recipe** for `K`. Fit its calibrator and choose only the registered threshold/margin operating points there. Do not reuse a balanced training pair sample for calibration.
5. Lock the entire bundle and evaluate once on `A`, including the same background construction, full focal truth, and all incoming false links.
6. **Deploy that audited checkpoint and its bound calibrator/policy.** Do not refit on 100% of labels after the audit and carry over its old thresholds.

A proposed larger-data refit is a new release candidate: retain `K/A` from fitting, regenerate natural calibration candidates, refit calibration/thresholds for the new checkpoint, and obtain a fresh audit assessment. If audit labels influenced the change, register a new untouched audit cohort before claiming an independent result.

Bind calibration to model, feature, encoder, normalization, and candidate-policy versions, and record its calibration-catalog fingerprint. Test naturally uses different index contents; it must use the same index-building recipe, with the new test index fingerprint recorded. Validate that score-margin features are normalized consistently across those catalogs.

The read-only structural audit in this document is not a model-validation result. It did not compute predictions or choose operating thresholds.

### 7.5 Country and corruption stress tests

- Fit on US, evaluate on India; reverse the direction in a separate run.
- For a genuine unseen-country test, do not fit calibration or thresholds using that held-out country's labels.
- Repeat relevant experiments with raw country tokens removed from neural serialization, and avoid learned closed-set country categories in the core.
- Apply label-preserving noise stress tests only after partitioning.
- Include missing-address, non-Latin name, non-Latin address, rare-name, common-name, source-specific, and high-cardinality slices.

These tests reveal brittleness. They do not estimate France's score or bound its transfer gap.

### 7.6 Compare policies on entities

Store per-reference scores for paired comparisons. Report mean difference, component-level bootstrap uncertainty, and country/source slices. Where repetitive name families create dependence, add a family-clustered sensitivity analysis.

A statistically small delta does not justify a large runtime increase. A positive global delta does not erase a serious regression on a high-risk slice. Avoid dozens of adaptive trials against the same locked set.

---

<a id="representations"></a>
## 8. Record and pair representations

### 8.1 Preserve evidence rather than canonicalizing it away

Keep the immutable raw fields and derive separate views:

| View | Purpose | Guardrail |
|---|---|---|
| NFKC + case-folded text | Case and compatibility variations | Raw text retained |
| Punctuation/space-normalized text | Formatting and token comparisons | Preserve meaningful separators in a parallel address/number view |
| Latin accent-folded view | French and injected-accent variants | Do not globally remove Indic combining marks, viramas, or vowel signs |
| Word tokens and character n-grams | Word-order changes, typos, abbreviations | Frequency weighting; no corpus-wide all-pairs matrix |
| Name-core/alias views | Legal-form variants, DBA/trade names, URLs in supplied text | Additive views, not destructive rewrites |
| Address components with confidence | Number, unit, street, locality-like fragments | Unknown and missing stay unknown |
| Script and missingness indicators | Routing and field reliability | Not stand-alone evidence of identity |

Do not unconditionally strip every `Mn` character from every language. This may look harmless on English/French and damage scripts present in hundreds of thousands of supplied records.

### 8.2 Country is an open string, not a fixed class list

Normalize whitespace/case in a separate country key and create partitions from whatever labels occur in the input. The current data has three labels, but the code must not enumerate only US/India or discard a fourth label.

All 7,638,365 observed true links are within country. **Default:** within-country retrieval, dynamically dispatched for every country present. This is an evidence-backed efficiency assumption, not a claim that country corruption is impossible. If later data or an organizer clarification contradicts it, add a measured cross-country rescue lane; do not silently drop rows.

No country-specific lower match prior is assigned to France.

### 8.3 Names: distinguish identity evidence from generic words

The sampled name-collision rate makes frequency information essential.

- Preserve legal-form tokens in the raw view; compare an additional name-core view.
- Learn common equivalences from training positive pairs, using support and ambiguity controls.
- Extract DBA/alias spans and URL/handle-like spans already present in the record. Never visit the URL.
- Separate exact rare-token agreement from agreement on words such as “global,” “services,” or “private.”
- Compare order-sensitive and order-insensitive views; token-set similarity alone can be misleading when a short generic name is contained in many longer names.
- A source-ID prefix may identify S2 versus S3. The rest of an ID must not enter the features.

### 8.4 Addresses: structured evidence with uncertainty

Extract numeric groups, potential street numbers, units, suffixes, and postal-like patterns. Retain original strings and component confidence.

Important distinctions:

- `12`, `12A`, `12/1`, and `12 bis` are not universally equivalent.
- A matching street number is weak evidence without compatible street/locality information.
- Missing address is different from an explicitly conflicting address.
- A candidate with a matching brand at a different unit or street may be a hard negative.
- “St” can denote street or saint; an unconditional replacement can harm names/localities.
- Postal patterns are optional signals, not mandatory blocking keys. Many supplied addresses do not contain an obvious postcode.
- State/province abbreviations and script correspondences may be learned from provided matched addresses. Do not import geographic databases.

The classifier learns how reliable these comparisons are in the supplied labels. There is no universal hard house-number veto or “same address means same business” rule.

### 8.5 Pair feature design follows observed confusions

Start with a compact, auditable feature schema, not a quota of 60 features:

1. Name character similarity, token overlap, token rarity, length ratios, and exact-view indicators.
2. Address character/token similarity with separate missingness flags.
3. Numeric/unit/postal agreement and credible contradictions.
4. Cross-field evidence: strong name with weak address; strong address with a transformed name; both generic.
5. Retrieval context: rank by lane, normalized score margin to another reference, agreement between name/address lanes, reference name frequency.
6. Source indicator, script indicators, token/character lengths, parser confidence.
7. Dense similarity only if a validated dense retrieval/feature experiment supplies it.

Every feature family must answer a named error pattern and have a leave-one-family-out test. Do not let ID suffixes, input order, truth cardinality, or true sibling membership leak into inference features.

---

<a id="retrieval"></a>
## 9. Candidate discovery: index the references, compact before matching

### 9.1 Core engine and direction

Use a disk-backed inverted index with efficient top-k retrieval. **Default engine:** Tantivy through its Python bindings; it supports BM25, configurable tokenization, memory mapping, and macOS/Linux deployment under MIT. [TANTIVY] [TANTIVY_PY]

Index each country's S1 reference records. Query using every S2/S3 record in that country. This keeps the indexed side smaller and exploits the at-most-one-reference-owner structure.

This orientation is a hypothesis about the best cost/quality balance, not a theorem. Benchmark a forward competitor at equal candidate count on a catalog-scale panel. Keep whichever wins the measured frontier.

Avoid:

- `rank_bm25`-style Python scoring of the entire reference corpus per query;
- `RapidFuzz.extract` against every reference as a global retrieval algorithm;
- dense `N_query × N_reference` similarity matrices;
- a sparse matrix product that materializes every nonzero similarity before top-k;
- enormous exact-name buckets emitted without refinement.

Returning only top-k results does not make a quadratic search implementation scalable.

### 9.2 Initial lexical retrieval lanes

Use separate field views so long addresses do not drown out names:

1. **Name lane:** character-trigram BM25 plus word-token information, using raw-normalized and name-core/alias views.
2. **Address lane:** character/token BM25 with numeric and component information retained.
3. **Joint lane:** field-weighted name/address evidence, tested against simple fusion of the first two lanes.
4. **Rare-key lane:** discriminative exact token/component conjunctions, such as a rare name fragment with a compatible numeric/street fragment.

These are indexed searches, not independent full Cartesian passes. The first executable baseline uses exactly two lanes: one safely normalized name field and one safely normalized address field, each with weight one. Joint/rare-key and alias-view retrieval are later additions only if they recover distinct true misses.

**Engineering starting grid:** retrieve 8 per active lexical lane, then compare 4/8/16 on a diagnostic panel. This creates `H(b)`, a raw pool of reference hypotheses for target `b`. These values are experimental budgets, not recall promises.

Store lane provenance, ranks, scores, and overflow/truncation events. Do not rank the union solely by the number of lanes that returned a candidate; several correlated lanes can agree on the wrong generic business.

### 9.3 Baseline fusion and deterministic ordering

Use reciprocal-rank fusion for the first two-lane baseline, with one-based ranks and an offset of 60:

```text
u(a,b) = sum over active lanes l returning reference a: 1/(60 + rank_l(a,b))
z(b)   = number_of_active_lanes / 61
v(a,b) = u(a,b) / z(b)                         # in [0,1]
g(b)   = v(best,b) - v(runner_up,b)             # only when a runner-up exists
```

A missing/blank field deactivates its lane. A candidate absent from an active lane contributes zero to that lane. No active lane or no returned hypotheses means an empty raw pool. Multiple correlated views inside one field must first collapse into one field-lane ranking; they do not get independent votes merely because more aliases were generated.

Order candidates by fused `v`, then address-lane rank, then name-lane rank; a missing rank sorts last. Use the exact reference ID only as the final stable enumeration tie-break. This does not make an ID semantic identity evidence. Log how often a strict budget cuts a fused-score tie; the adaptive policy below can retain tied candidates rather than silently pretending the tie is resolved.

This RRF baseline is a specified control, not a claim of optimal weighting. Field weights and raw-score alternatives require their own development comparison.

### 9.4 Compact the raw pool

First establish strict uniform reverse-budget baselines `b ∈ {1,2,3}` using the order above. Then test the following deterministic adaptive policy:

```text
If H is empty: retain none.
If both active field lanes rank the same reference first,
   and a runner-up exists, and g >= one_owner_gap:
       nominal budget = 1.
Otherwise:
       nominal budget = hard_budget if address is missing,
                        field leaders disagree, or the target has Indic text;
                        else 2.
Retain the ordered prefix of the nominal budget.
If the cutoff splits an exact fused-score tie, retain all members of that tie
   already in H and record the expansion.
```

Registered starting grid: `one_owner_gap ∈ {0.02, 0.05, 0.10}`, `hard_budget ∈ {2,3,4}`. Initial adaptive control is `(0.05, 3)`. If only one raw hypothesis exists, retain it; a missing runner-up is **not** evidence of a large margin.

The initial policy does not use an absolute RRF support floor: a nonempty two-lane pool necessarily has a normalized best score of at least 0.5, regardless of whether any record is the same business. RRF supplies relative ranking, not an absolute match-confidence test. A zero-candidate rule for weak but nonempty searches requires a separate, explicitly specified absolute-evidence experiment and the support-loss gate; it is not hidden inside this baseline.

Every rule uses available retrieval/input signals, not true match count. The Indic routing flag is the generic script diagnostic in §4, not a country whitelist. Exact-tie expansions can exceed the nominal budget, so the strict `b*M` bound in §6 does not apply to them; report their real cost. Returning zero candidates is a retrieval action, not a forced prediction that the target is globally unmatched.

Never apply a small unconditional cap to the **incoming S1 list** just because target-side budgets are small.

Candidate-retention thresholds and fusion weights are fitted on development retrieval outcomes, then frozen. Measure raw-pool recall, post-compaction recall, and oracle loss separately.

The first implementation should choose between a few transparent policies. A complex learned gating network is not required to establish a compact baseline.

### 9.5 Rare-key collisions and heavy hitters

Name-only exact lookup is unsafe and can be huge. For an oversized bucket:

1. Record its frequency and overflow.
2. Refine using available address/number/rare-token evidence or fall back to indexed ranking.
3. Preserve potentially relevant aliases through a measured alternative lane.
4. Never keep the first records by file order or silently discard the entire bucket.

High incoming degree at an S1 is an ambiguity signal and a memory concern, not a license to cap all references at the training maximum of 11 matches.

### 9.6 Cross-script and weak-evidence rescue

The core records which slices it cannot retrieve compactly. The first rescue experiment is multilingual-e5-small over serialized records; an offline transliteration view is another possible diagnostic, not a replacement for evidence.

Start by encoding the reference catalog once and querying the difficult slice. Audit a random sample of apparently easy queries with the rescue retriever too, so the gate's blind spots are measurable.

A dense lane must improve true-link recall at a fixed final candidate budget or reduce candidates at comparable quality. Adding dense candidates indiscriminately is not automatically a win under the ranking update.

### 9.7 Freeze the candidate relation before scoring

After retrieval compaction, emit a versioned relation:

```text
reference_ordinal, target_source, target_ordinal,
retrieval_provenance, retrieval_ranks, retrieval_scores, candidate_policy_version
```

Deduplicate edges, validate endpoints, and freeze the relation's fingerprint. Transpose/group it by S1 for `candidate_pairs.tsv`; include zero-candidate reference rows.

All exact-match shortcuts also enter this relation. If an experiment later expands it, regenerate the support manifest and audit all newly scored edges. No post-prediction rewrite of candidate support is permitted.

### 9.8 What candidate discovery is not allowed to assume

- The training maximum number of matches is the test maximum.
- Every target has an owner in S1.
- Same brand means same business.
- A high multilingual semantic similarity establishes identity.
- High ANN search recall establishes high true-link recall.
- All positive pairs will share a useful rare token because a sample shared some token.
- Smaller candidate files alone prove a scalable implementation.

---

<a id="matcher"></a>
## 10. Matching model and supervision

### 10.1 Core model: LightGBM binary classifier

Use a LightGBM binary classifier on the comparison and ambiguity features from §8. It is fast to iterate, works on CPU, handles feature interactions and missingness, and produces a score suitable for threshold validation. This is a practical prior, not a claim of universal superiority. [LGBM]

**Engineering starting configuration:** binary log loss, learning rate 0.05, 63 leaves, minimum 100 examples per leaf, L2 regularization, no class reweighting initially, bounded histogram memory, fixed seed. Compare a smaller 31-leaf model before adding model families.

Select checkpoints using complete entity outcomes and actual decision policies at a small number of checkpoints. Log loss remains useful as a convergence and probability-quality diagnostic. A differentiable “macro-F₀.₅ loss” is not necessary to optimize the final scored decisions.

### 10.2 Training examples

- Positive labels come from the immutable provided truth.
- Core negatives are real nonmatching edges returned by the current retrieval policy.
- Include unlinked target records: approximately 26% of training targets have no reference owner.
- Use competing references for the same target as hard negatives, particularly same-name/different-address and same-address/different-name cases.
- Do not fill training with overwhelmingly easy cross-country/random pairs that production blocking never presents.
- Keep all siblings in one partition; never sample a true sibling as a contrastive negative.

Gold-positive injection can be used for a separately marked supervised training augmentation. It must never enter validation/test candidate sets or candidate-recall measurements. A model's ability to score injected positives is not evidence that the blocker finds them.

### 10.3 Sampling and macro alignment

The full Cartesian space is extremely imbalanced, but a good compact candidate set may not be. Do not impose a 25–40% positive ratio because another paper used one.

Start by sampling reference components/target groups while retaining their natural candidate sets and unlinked distractors. If negative downsampling is needed, record the inclusion rule and weights.

Pure label-based resampling permits a class-prior log-odds correction. Hard-negative selection and context-dependent sampling generally do not reduce to a single prior correction. Calibrate on natural held-out candidates from the actual deployed policy.

Pair loss and entity macro-F are different objectives. Compare ordinary log loss with reference-balanced sample weights as an experiment; recheck probability calibration after changing weights. The final arbiter is full-truth macro-F₀.₅ at the chosen operating policy.

### 10.4 Hard-negative mining without leakage

1. Fit the first model on fitting-partition candidates.
2. Score a broader fitting-partition retrieval pool or use out-of-fold training scores.
3. Mine high-scoring wrong-reference candidates and hard missed positives.
4. Refit one model revision.
5. Evaluate on unchanged development/audit inputs with no truth injection.

If mining changes candidate distributions materially, rebuild the relevant calibration results. Do not mine held-out labels into the training set while retaining the old validation claim.

### 10.5 Ground-truth auditing

Inspect suspicious high-confidence disagreements and identical-input label conflicts. Record the evidence and distinguish entity ambiguity from an actual annotation problem.

The official truth remains the scoring authority. Do not silently relabel validation data to make a model look better. Training-only robust weighting is an experiment if systematic label noise is established; unrestricted self-relabeling is not the default.

### 10.6 Field noise and augmentation

With 7.6M positive links, data scarcity is not the primary assumption. First use observed variants.

Useful controlled experiments include casing, punctuation, token order, observed abbreviation alternatives, and limited field masking. Preserve identity-defining numbers unless the transformation is specifically supported by observed label behavior. Apply transformations after splitting and track their origin.

Do not generate fake French addresses by changing postcodes or legal forms and then treat the result as French ground truth. Do not automatically preserve a negative label after deleting its only distinguishing field.

---

<a id="decisions"></a>
## 11. Calibration, match selection, and owner conflicts

### 11.1 First decision baseline: global threshold

Choose `tau` to maximize macro-F₀.₅ on complete held-out reference rows:

```text
S_a = { b in C_a : score(a,b) >= tau }
```

This is deliberately simple. It is a strong experimental control and may be the best deployable policy when elaborate probability models are wrong.

Threshold search can be exact over score breakpoints: sort candidate edges by score, process tied scores together, update each affected row's `(predicted_count, true_positive_count)`, and maintain the total macro score. Initialize every row with the correct empty-prediction contribution. This avoids a large nested loop over thresholds and references.

### 11.2 Nullable reference ownership

The data supports at most one true S1 owner per target. The core includes a competing policy:

1. For each target, find its best and runner-up reference scores among candidates.
2. Keep the best only if it passes `tau` and a validated ambiguity-margin rule.
3. Otherwise leave the target unassigned.
4. Group accepted targets by reference; a reference can receive any number.

Use calibrated probability differences for this policy, with `gamma ∈ {0, 0.02, 0.05, 0.10}`. For each gamma, select tau by the exact breakpoint sweep on the resulting eligible winner edges. With two or more candidates, require a strictly unique top score and `p_best - p_second >= gamma`; exact top-score ties abstain even at gamma zero. With only one scored candidate, the margin is missing and the candidate is eligible based on tau alone. Never assign an omitted competitor a probability of zero.

Do not force an assignment merely because a candidate ranks first, and do not use ID magnitude to decide between equally supported owners. After eligibility is fixed, process equal winner scores together in threshold tuning. Freeze the winning `(calibrator, gamma, tau, tie policy)` tuple with the model and candidate recipe.

This constraint is a semantic modeling choice, **not an extra output-format rule stated by the organizer**. It is not automatically macro-F optimal: resolving a conflict changes whole-row utilities. Compare it with unconstrained thresholding on the same support, and keep the better justified policy. Investigate conflicts rather than hiding them.

### 11.3 Probability calibration

For ranking/thresholding alone, a well-validated raw score can suffice. For probability-dependent decisions or stacking, calibrate explicitly.

**Default candidate calibrator:** a monotone sigmoid/Platt mapping with both slope and intercept, fitted on raw model margins from natural held-out candidates. Temperature-only scaling cannot generally repair an intercept shift caused by positive oversampling.

The supervision mask and the score-comparison environment are different:

1. For final calibration, fit Platt only on candidate edges **whose reference is in K**, across the complete K-plus-background query traffic. Label each edge from the complete K truth. This includes incoming background→K negatives; it does not pool the reused F/B references' own labels or restrict fitting to K-owned target queries alone.
2. Apply that same monotone mapping to **all** scored candidate edges used in whole-environment owner competition, including background-reference competitors. Do not compare a calibrated focal probability with an uncalibrated competitor score. Calibration evidence is established on focal outcomes; background probability transfer is a modeling assumption assessed by the resulting focal decisions.
3. Perform ownership/selection with the full candidate support, then restrict threshold utility and reported validation metrics to K references. Never discard background competitors before deciding the owner.
4. For temporary development calibration, split D by identity components into two halves. Fit a mapping on one half's focal-reference edges using its full background traffic, evaluate the other half's complete focal outcomes, and swap. Use comparable country-sized catalogs and pool these cross-fitted development results to choose the policy family. Final K calibration is then redone for the selected actual checkpoint.

Store raw margins and calibrated values separately. Use the monotone raw/calibration-logit order to detect genuine winner ties, avoiding false ties caused only by rounded probabilities near zero or one. Probability margins use float64 and the selected mapping. A non-increasing fitted mapping is a calibration failure requiring investigation, not a reason to reverse identity rankings silently.

Evaluate Brier/log loss and reliability near the selected high-precision operating region, not only a global ECE dominated by easy negatives. Report calibration by source, script, and missingness where support is sufficient.

No arbitrary “estimated versus actual precision correlation must exceed 0.9” criterion is used. Calibration quality and final decision quality are related but not interchangeable.

### 11.4 Small row-aware challenger

If the threshold baseline loses on singleton/start-versus-addition decisions, test two thresholds:

```text
if best candidate score < tau_start:
    predict empty
else:
    retain the best candidate and any additional candidate above tau_extra
```

For this challenger, fix the order: nullable-owner eligibility first, then row start/add thresholds on the surviving winner edges, with no reassignment loop. Treat this as a distinct policy family and compare complete resulting sets; do not claim that separately optimal stages imply an optimal composition. Select its parameters on development and final-calibration cohorts, never on the locked audit.

A separate model of `P(true row is empty)` is another gated experiment. Its label is an actual empty truth set, **not** “no positive candidate was retrieved.”

### 11.5 Advanced decisions

GFM, learned cardinality-conditioned utility, and globally coupled conflict optimization are optional. Promote them only if their out-of-fold probability estimates and inference cost produce a repeatable gain over the simple policies.

Do not:

- replace expectations of ratios with ratios of expectations and call them exact;
- assume an independent truth model merely because pair scores are calibrated;
- clip truth cardinality to candidate count;
- use generated multi-ID token probabilities as pair probabilities;
- claim a guaranteed gain from a theoretically richer decision rule.

---

<a id="generalization"></a>
## 12. Generalization: France, India, sources, and missing evidence

### 12.1 Priorities established by the data

France contributes about 15% of test references and no labeled training references. India contributes almost 47% and has large native-script target populations paired with predominantly Latin-script reference text. Both deserve attention; “all multilingual effort goes to France” is the wrong allocation.

The test target/reference ratio is 5.7543, versus approximately 4.68 in training. This does not identify test match cardinality or singleton prevalence, but it warns against blindly forcing training-derived counts on test.

### 12.2 Core robustness

- Dynamic country partitions, never a closed US/India one-hot representation.
- Raw text plus safe Unicode views; preserve native scripts.
- Separate name/address evidence and missingness.
- Frequency-normalized features that generalize to unseen business tokens.
- Train-derived equivalence maps with an identity-preserving fallback for unknown tokens.
- Pooled thresholds/calibration as the default for unseen country labels.
- No manual per-test-record identity lookup, relabeling, or patching.

Country-specific conventions such as French `bis`/`ter` suffixes are handled as structured text, not as reasons to rewrite an address into an external canonical location.

### 12.3 What unlabeled test inspection can establish

Allowed structural observations include country proportions, script/length distributions, missingness, token coverage, and runtime requirements using the provided files. They can reveal whether the code routes every record correctly.

They cannot reveal France's true singleton rate, match count distribution, recall, optimal threshold, or private leaderboard performance. A shift in score distributions alone does not establish a shift in true match prevalence.

### 12.4 Stress tests, not invented French benchmarks

Keep country holdout and label-preserving noise stress tests. Name them according to what they are: US→India transfer, India→US transfer, accent-fold robustness, address-drop robustness, and source-transfer tests.

Do not use a “synthetic France score” to choose a France-specific prior or report estimated French accuracy. Natural French error analysis must wait for legally available labels or remain an acknowledged uncertainty.

### 12.5 If generalization underperforms

Use stage attribution:

- True owner absent from `H`: improve representation/retrieval or add a cross-script lane.
- Owner in `H` but removed from `C`: revise compaction, not the classifier.
- Owner in `C` but scored poorly: improve comparison features or a multilingual matcher.
- Correct ordering but wrong decisions: recalibrate or revise the policy on legitimate validation.
- Many indistinguishable records: quantify information limits; do not invent missing business facts.

---

<a id="neural"></a>
## 13. High-value neural experiments

### 13.1 Experiment N1: multilingual-e5-small retrieval

**First neural candidate:** `intfloat/multilingual-e5-small`, MIT, approximately 118M parameters, 384-dimensional embeddings. [E5]

Serialize each record independently with field boundaries. Follow the checkpoint's required prefix convention and pooling/normalization. Record that convention in the index fingerprint; changing it invalidates the index.

Why small first:

- It is a cheap test of whether multilingual representations recover the identified cross-script misses.
- A reference-side 384D float32 matrix is about 2.48 GiB before ANN overhead.
- The large BGE-M3 model and 1024D vectors need not be paid for before a smaller model's marginal value is measured.

Test dense-only, lexical-only, and fused retrieval at the **same final candidate budget**. Measure ANN neighbor fidelity against exact search on a bounded panel separately from true-owner recall.

### 13.2 Experiment N2: supervised compact retrieval

If N1 improves some slices but lacks discrimination at small budgets, fine-tune the retriever on provided target→reference positives with hard competing-reference negatives.

- Train on fitting components only.
- Use reference deduplication to construct negatives; mask any same-owner positives within a batch.
- Mine hard negatives from fitting catalogs.
- Preserve source variation and native-script positives.
- Compare learning curves on successively larger samples before a full-data run.
- Rebuild all affected indexes after changing the encoder.

The objective is recall at a small candidate budget, plus the metric-aligned support ceiling, not generic sentence similarity or MTEB performance. Sudowoodo supplies useful methodological precedent, not a guaranteed reduction percentage. [SUDO]

### 13.3 Experiment N3: compact cross-encoder

Try a pair classifier initialized from `microsoft/mdeberta-v3-base` or `FacebookAI/xlm-roberta-base`, both roughly 278–279M-parameter multilingual checkpoints with MIT model-card declarations. [MDEBERTA] [XLMR]

mDeBERTa's model card describes CC100 multilingual pretraining. Its XNLI evaluation/language tags are not proof that pretraining covers only 16 languages, and XNLI superiority does not prove ER superiority.

Train a binary classification head on raw-plus-structured record pairs and retrieved hard negatives. Use bounded field-aware token allocation; measure truncation by script. Start with ordinary supervised fine-tuning and held-out calibration, not rationale generation.

Admission tests:

1. Does it repair a specific class of LightGBM errors?
2. Does a simple score blend improve complete entity outcomes on unchanged support?
3. Can it run within the measured budget, possibly on an uncertainty-routed subset?
4. Does the router miss neural improvements among apparently easy pairs? Audit a random easy sample.

The candidate file remains the union of the matching pipeline's scored pairs, even when only a subset receives neural scoring.

### 13.4 Optional local LLM

A local eligible small generator, for example Qwen3-4B, is a **late experiment**, not the main pipeline. Use it only when unresolved cases require evidence unavailable to the compact matcher and measured throughput is acceptable. [QWEN]

If used, prefer an explicit pairwise classification head or constrained Yes/No sequence scoring followed by held-out calibration. A generated explanation or candidate-ID token probability is not a calibrated identity posterior.

ComEM's one-choice selector may inspire **target→reference-or-NONE** ranking, whose cardinality fits this task's inverse direction. That is more defensible than directly copying its selector into a multi-match S1 output, but it still needs local evaluation and a null decision. [COMEM]

No external LLM service processes challenge records. Teacher-generated labels are unnecessary for the core because extensive supplied supervision already exists.

### 13.5 Parameter and license budget

| Component | Role | Approximate parameters | Evidence/status |
|---|---|---:|---|
| LightGBM | Core matcher | Small learned tree model; record exact artifact size/structure | MIT implementation; release our trained artifact under an eligible license |
| multilingual-e5-small | Retrieval experiment | 0.118B | MIT card; pin actual revision |
| mDeBERTa-v3-base | Matcher experiment | 0.278B | MIT card; includes embedding parameters |
| XLM-R-base | Alternative matcher, not mandatory extra member | 0.279B | MIT card |
| Qwen3-4B | Optional late experiment | 4.022B | Apache-2.0 card/config |

E5-small + mDeBERTa + Qwen3-4B totals approximately 4.42B before any additional distinct checkpoints. The previous BGE-M3 + mDeBERTa + Qwen2.5-7B stack totals about **8.46B**, so assuming every individually eligible model automatically makes the ensemble compliant was unsafe.

**Engineering default:** keep the aggregate of deployed distinct learned weights below 8B, count embeddings and heads, and count multiple checkpoint copies/ensemble members honestly. Quantization changes storage, not parameter count. Verify licenses of actual checkpoints, not only their architecture names.

---

<a id="experiments"></a>
## 14. Experiment plan and error-analysis loop

### 14.1 One experiment should answer one question

| Order | Experiment | Question | Keep it when |
|---|---|---|---|
| E0 | Exact scorer and all-empty baseline | Are set semantics and singleton handling correct? | All regression cases pass; observed empty baseline reproduced |
| E1 | Two-lane reverse lexical retrieval; b=1/2/3 | Is compact reference-owner discovery viable? | Measured support/cost frontier is established |
| E2 | Forward retrieval at equal pair budget | Does the preferred direction actually win? | Better oracle/final score at acceptable runtime |
| E3 | One LightGBM model on a selected uniform-budget support | How much loss remains after retrieval? | Establishes a valid CPU baseline |
| E4 | Adaptive compaction versus uniform b with that matcher | Can easy targets use fewer candidates without losing hard ones? | Non-inferior end-to-end quality with fewer candidates |
| E5 | Feature-family ablations | Which evidence distinguishes hard identities? | Improvement survives held-out comparison |
| E6 | Owner-conflict and simple threshold policies | Does exclusivity/abstention help the actual metric? | Measured complete-row gain on fixed support |
| E7 | Small multilingual retrieval | Are misses representational, particularly cross-script? | Recall/ceiling improves at equal budget |
| E8 | Supervised retrieval tuning | Can learned retrieval make candidate sets materially smaller? | Better frontier than E7 plus sufficient runtime |
| E9 | Compact neural matcher or selective blend | Are semantic/local alignment errors limiting the tabular model? | Repairs residual errors with a real net gain |
| E10 | Row-aware thresholds or GFM | Are decisions, rather than pair discrimination, limiting? | Beats simple policy under calibrated, disjoint validation |
| E11 | Final full-catalog rehearsal | Is the selected pipeline reproducible and timely? | Complete files, strict audit, measured safety margin |

The order after E3 is driven by loss decomposition. E4 is a challenger, not a prerequisite for a valid release. If blocking dominates, work on E7/E8 before another classifier. If blocking is already strong and false merges dominate, prioritize E5/E6/E9.

There are no promised improvement columns. The hypotheses are falsifiable.

### 14.2 Required baseline ladder

1. All-empty predictions as a scorer check.
2. Conservative joint-evidence rules on a valid candidate set.
3. Lexical retrieval + LightGBM + global threshold.
4. The same scores with nullable-owner conflict handling.
5. Optional challenger: adaptive candidates with the best simple matcher/policy; retain the best uniform budget if adaptation does not help.

This ladder prevents a large ensemble from concealing whether any of its expensive parts matters.

### 14.3 Error taxonomy

For each wrong reference, record the stage and evidence:

| Error type | Diagnosis | Next useful action |
|---|---|---|
| True owner not in raw pool | Search representation or indexing failure | New view/lane, tokenization, query budget, dense retrieval |
| Owner pruned from final support | Compaction too aggressive | Margin/budget/refinement change |
| Same common name, wrong location | Insufficient contradiction/rarity modeling | Address/unit/frequency features, hard negatives |
| Same address, distinct businesses | Co-location confused with identity | Name specificity, unit/alias semantics |
| Native-script true match scored low | Representation/alignment gap | Learned multilingual retrieval/matcher |
| Missing address and ambiguous name | Weak observable information | Better name evidence or calibrated abstention |
| Correct candidate order, wrong set size | Decision/calibration error | Threshold/row-aware policy |
| False link on a true singleton | Overconfident weak evidence | Natural-candidate calibration, unlinked-target negatives |
| Multiple targets lost from a real group | Per-reference cap or redundancy pruning | Remove invalid capacity assumption |
| Conflicting owners for one target | Target competition mishandled | Null/margin/utility conflict experiment |
| Country/script regression | Shortcut learning or parser bug | Safe views, held-out-country analysis |
| Duplicate/unknown IDs in output | Pipeline contract failure | Strict export/audit fix before more modeling |

Sample errors by lost **entity score**, not merely by number of wrong pairs. Also inspect high-confidence errors, threshold-near errors, and cases uniquely changed by an experimental component.

### 14.4 Experiment record

Each run records:

```text
run_id; data fingerprints; split manifest; code/config digest;
normalization version; index/encoder revision; candidate policy;
candidate-support fingerprint; training/sampling scheme;
model checkpoint; calibrator; decision policy;
macro_F and slices; oracle ceiling; candidate distribution;
wall-clock by stage; peak memory; error categories;
decision: retain / reject / rerun with stated reason
```

An unsuccessful experiment is useful evidence. Do not delete it from the history or reclassify it as a success because a public leaderboard moved.

### 14.5 Stop conditions

- A support/format violation blocks submission regardless of score.
- A component with no held-out gain is removed from the final path.
- A neural model with positive pair-level F1 but negative entity macro-F is rejected.
- A small candidate file produced through hidden expensive matching is not an acceptable blocker.
- A component that cannot complete a measured rehearsal with recovery time is not the release candidate.
- If no strong model experiment beats the CPU baseline, submit the validated CPU baseline.

---

<a id="compute"></a>
## 15. Computational reality and scaling

### 15.1 Observed environment versus competition resources

The current review host reports 16 GiB RAM and 10 logical CPUs. No GPU capability or rented compute budget is assumed. The read-only full-scan profiling completed on this host; this says nothing about neural inference throughput.

Choose the final deployment environment after measuring the baseline. Keep training/inference artifact formats portable and pin platform-sensitive dependencies.

### 15.2 Memory arithmetic

| Object | Raw payload calculation | Consequence |
|---|---|---|
| S1 test vectors, 384D float32 | **2.48 GiB** | Plausible before ANN/metadata overhead |
| S1 test vectors, 1024D float32 | **6.61 GiB** | Substantial on a 16 GiB host |
| All test targets, 384D float32 | **14.26 GiB** | Do not retain alongside everything else |
| All test targets, 1024D float32 | **38.03 GiB** | Exceeds this host's RAM before any index |
| 50 candidates/S1 × 48 float32 features | **15.49 GiB** | Feature matrix alone nearly exhausts RAM |

These are payload sizes, not peak memory estimates. Python strings, dictionaries, ANN links, tokenizers, model activations, cached pages, and framework overhead add more.

### 15.3 Storage/execution plan

- Use stable integer ordinals for internal joins, retaining an immutable mapping to exact original IDs.
- Keep strings in columnar/offset-based storage rather than millions of duplicated Python objects.
- Build per-country indexes sequentially if memory requires; discover countries dynamically.
- Stream target queries and features in bounded batches.
- Store candidate/scored edges in partitioned columnar files; use external sorting/grouping for exports.
- Cache reference tokenization/record views. Cache independent embeddings when used; cross-encoder pair activations are not reusable record embeddings.
- Stream target embeddings rather than materializing every target vector in RAM.
- Spill large incoming reference groups safely; do not silently truncate them.
- Release models/indexes before running a memory-heavy validation step.

No billion-record scalability claim follows from running this dataset. The architecture should avoid quadratic work, support sharding, and document actual indexing/query complexity; a production distributed system is outside hackathon scope.

### 15.4 Bounded-memory training contract

LightGBM training is not made out-of-core merely by streaming feature generation. Its constructed training Dataset and histograms must fit the selected environment.

**Initial 16 GiB-host policy:** cap a fitting attempt at four million candidate-feature rows, then measure peak memory. Use float32 numerical features and compact integer codes; use `max_bin=127`, bound `histogram_pool_size` initially to 512 MiB, and prefer `force_col_wise=true` to avoid the extra row-wise Dataset memory. These are starting settings, not a proof of memory use. [LGBM]

For four million rows and 48 float32 features, raw feature payload is 768 MB (about 0.715 GiB). A one-byte-per-bin illustrative binned payload is 192 MB, but LightGBM also needs labels, gradients, row indices, histograms, and implementation-specific copies. Measure the constructed Dataset and fitting process rather than budgeting from the binned payload alone.

Assemble versioned feature shards on disk. Construct the Dataset in a dedicated process after retrieval indexes and unnecessary record dictionaries have been released. Do not materialize raw strings alongside every training feature row. An initial process budget of **11 GiB peak RSS** leaves headroom on this host; exceeding it triggers a smaller training sample/batch or a provisioned larger host, not an OOM retry loop.

If the row cap is exceeded, select complete fitting query/reference groups with a recorded seed, preserving natural candidates and explicitly stratified no-owner targets. Record inclusion weights when this changes the sampling distribution. Run a training-size learning curve; a bounded representative sample is a legitimate final training set when larger fitting has not demonstrated benefit.

Provision an initial **50 GiB scratch allowance**, then replace that allowance with a preflight estimate based on measured index, raw-pool, feature-shard, score-ledger, and model sizes, including temporary sort space. Do not assume the initial allowance is sufficient. Retain only necessary run artifacts; never remove the selected model/support manifests or user data as ad hoc cleanup.

### 15.5 Time arithmetic, not throughput guesses

For `P` final pairs, measured end-to-end feature/scoring rate `q`, routed neural fraction `h`, and measured neural pair rate `r`:

```text
T_total = T_ingest + T_index + T_retrieve + T_features
        + P/q + h*P/r + T_group_decide_export + T_validate
```

Do not double-count feature time if `q` already includes it. Record both components to make the budget auditable.

For a **20M-pair illustrative workload**, pure scoring takes:

| Measured rate, if achieved | Arithmetic time |
|---:|---:|
| 100 pairs/s | 55.56 hours |
| 500 pairs/s | 11.11 hours |
| 1,000 pairs/s | 5.56 hours |

These are scenarios, not benchmarked rates. Similarly, ten million target queries at 1,000 queries/s require about 2.78 hours of retrieval alone. Query length, posting-list frequency, threading, and cache state matter.

### 15.6 Required performance probe

Before large training or full-test inference:

1. Build a production-scale reference index.
2. Query a fixed stratified panel with realistic countries/scripts and common-name cases.
3. Measure warm and cold retrieval, p50/p95/p99 latency, and total throughput.
4. Measure feature extraction and scoring separately with realistic batch sizes.
5. Measure feature-shard assembly, LightGBM Dataset construction, and fitting peak memory/time on successive training sizes.
6. Measure neural token lengths, throughput, and peak memory if applicable.
7. Run grouping/export and the validator on a representative output size.
8. Extrapolate from these measurements, then run an end-to-end rehearsal.

A published vLLM speedup or an unrelated four-A100 token rate is not a budget for this hardware. QLoRA memory use depends on sequence length, batch size, optimizer, and implementation; “7B fits in 16GB” is not an unconditional guarantee.

### 15.7 Failure recovery

Checkpoint completed query shards and bind them to input/index/config fingerprints. Resume only compatible shards. A classifier failure must not silently emit empty predictions for an unprocessed shard.

If a neural stage fails, use the complete validated CPU pipeline for the affected run or an explicitly validated hybrid fallback. If a parser/serializer fails, fix the data-contract problem; do not reinterpret failure as a singleton prediction.

---

<a id="execution"></a>
## 16. Implementation contracts and practical execution order

This is the implementation handoff. The paths and commands below are **planned interfaces**, not claims that the pipeline already exists. A separate architecture-choice document is not required before starting.

### 16.1 Proposed package layout

```text
code/business_entity_resolution/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    core.json
    selected.json
  artifacts/
    final/                     # selected model, preprocessing, calibration, manifests
  src/ber/
    cli.py                     # orchestrates explicit stages
    records.py                 # schemas, streaming parse, ID/ordinal mappings
    normalize.py               # immutable raw-preserving views
    splits.py                  # component assignments and leakage checks
    retrieval.py               # indexes, query lanes, retrieval provenance
    candidates.py              # compaction, immutable support, transpose
    features.py                # comparison features over explicit candidate edges
    training.py                # supervised fitting and experiment metadata
    scoring.py                 # batched model inference and score ledger
    decisions.py               # threshold/ownership policies
    metrics.py                 # exact entity score and support diagnostics
    export.py                  # complete TSVs and manifests
    audit.py                   # strict integrity and reproduction checks
    tests/                     # focused correctness tests
```

Keep this decomposition compact. Add a neural adapter or learned retriever only when an experiment earns its place. The final package must place all executable source under `src/` as required.

### 16.2 Data contracts

| Artifact | Required contents | Invariant |
|---|---|---|
| Record table | original ID, ordinal, source, raw name/address/country, versioned views | One ID→one record; raw values preserved |
| Truth relation | S1 ID→set of true target IDs | Complete supplied labels; unchanged during scoring |
| Split manifest | identity/target-only group→partition | No labeled pair endpoint leaks across protected partitions |
| Raw retrieval | target→ranked reference hypotheses with provenance | Generated without evaluation labels |
| Final candidates | unique reference–target edges + policy/index fingerprints | Exact matching input support |
| Features | candidate key + schema-versioned values | One feature row per expected scored edge |
| Scores | candidate key + model/calibrator revision + score | Complete and finite; no silent unscored edges |
| Decisions | reference→selected target set + policy revision | Selected set is a subset of support |
| Outputs | two exact-schema TSVs | Every required reference, including empties |

### 16.3 Minimal interface semantics

```text
normalize(record, learned_maps) -> RecordViews
retrieve(target, reference_index, retrieval_config) -> RawHypotheses
compact(raw_hypotheses, record_views, candidate_config) -> CandidateEdges
compare(reference, target, retrieval_context) -> FeatureVector
score(candidate_edges, features, model_bundle) -> ScoredEdges
decide(scored_edges, required_references, decision_policy) -> MatchSets
evaluate(truth, candidates, predictions, required_references) -> MetricReport
export(required_references, candidates, predictions, manifest) -> TwoTSVs
audit(inputs, support_ledger, score_ledger, outputs) -> PassOrFailure
```

`retrieve` and `compact` never receive held-out truth. `evaluate` always receives full truth. `export` iterates the authoritative S1 table, not only references that happened to receive predictions.

### 16.4 Build order and exit criteria

| Milestone | Build/test | Exit criterion |
|---|---|---|
| M0 — Trust the data | Schemas, manifests, exact scorer, split constructor | Reproduce inventory; metric regression cases pass; no partition leakage |
| M1 — Find candidates | Reference index, two lexical lanes, b=1/2/3, provenance | Report recall/ceiling/size/runtime and every missing-truth category |
| M2 — Working CPU solution | Features, one LightGBM, threshold search, exporters | Complete valid held-out outputs; measured score above trivial controls |
| M3 — Compact and diagnose | Compare adaptive retention and owner policies with the functioning uniform-budget baseline | Select a nondominated policy under §6, including the unchanged baseline if it wins |
| M4 — Highest-value upgrade | Retrieval or matcher experiment selected by loss attribution | Promote a measured improvement **or record no improvement and retain the baseline** |
| M5 — Release calibration and audit | Freeze actual checkpoint/recipe, calibrate on K, then evaluate A in the large background environment | Audit report includes failures, slices, uncertainty, and runtime |
| M6 — Final inference and rehearsal | Deploy that audited checkpoint and bound policy in a complete test run; no uncalibrated full-data refit | Both TSVs, strict support audit, bundled artifacts, reproduction evidence |
| M7 — Package | Documentation template, environment, README, zip | A fresh process can regenerate the selected outputs from the documented inputs |

Do not block M2 on an LLM, a graph, a neural retriever, or a future theoretical decoder.

### 16.5 Correctness cases worth automating

- Official worked example equals **5/7**, not exactly the rounded decimal 0.714.
- Empty truth/empty prediction = 1; either mismatched empty case = 0.
- Truth with an unretrieved member still contributes that member to the denominator.
- An empty candidate row for a non-singleton scores 0.
- Multiple S2 matches and multiple S3 matches remain representable.
- Accent folding preserves the original view and does not erase Indic marks.
- Missing address differs from a conflicting address; strings such as `NA` are not silently converted to null.
- Duplicate edges are deduplicated before scoring/export; duplicate source IDs fail input validation.
- Unknown country labels reach indexing and export through the generic path.
- A neural cascade does not shrink the declared support to its routed subset.
- Candidate export equals the scored-edge support ledger, including rejected candidates.
- Every test S1 appears once, and no unknown target appears in either output.
- Resume with a changed index/model/config is rejected rather than mixing incompatible shards.

These tests protect actual failure modes. They are not a requirement to write tests that merely mirror a reversible documentation change.

### 16.6 Time allocation

The actual event deadline, hardware budget, and portal limits are not established by the provided statement. Do not plan around an invented 72-hour allowance or two weeks of preparation.

After the first performance probe, reserve time for one complete final run, one recovery run of the largest stage, full validation, and packaging. Spend remaining time in the experiment order above. Prefer a complete early valid baseline over waiting for a sophisticated model to finish.

---

<a id="submission"></a>
## 17. Compliance, output generation, and reproducibility

### 17.1 What the actual validator does

`student_resource/utils/validate_submission.py` was read in full. [VALIDATOR]

Important differences between helper behavior and the full submission contract:

- `--check-ids` is **off by default**.
- Missing `candidate_pairs.tsv` can produce a warning without failing.
- Matches absent from the candidate file produce a warning without failing.
- The script accepts some header normalization; our exporter must still write the exact specified header.
- It loads mappings/sets and can use substantial RAM at full candidate scale.

The official rules, particularly the newer ranking update, are stricter than “exit code 0 under default options.”

### 17.2 Run the official validator with ID checks

From `student_resource/`, retaining the official command's argument order and adding its supported stronger flag:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

Run it after releasing model/index memory. Add a memory-bounded strict audit for all of the contract conditions. If the helper exhausts memory, use a suitably provisioned validation process and report what actually ran; do not call a partial check a complete pass.

### 17.3 Strict release audit

All of these are failures in our pipeline, even when the supplied helper only warns:

1. Missing candidate file.
2. Missing/duplicate/extra S1 rows.
3. Unknown, wrong-source, or repeated target IDs in a list.
4. A selected match absent from that reference's candidate set.
5. Candidate support inconsistent with the actual final scoring run.
6. An unprocessed query shard silently exported as empty.
7. Mixed model/index/policy versions.
8. Nondocumented manual changes to outputs.

Compare a canonical sorted edge digest from `candidate_pairs.tsv` with the support and scoring ledgers. Record unique pair counts as well as total scoring calls so cascades and retries remain auditable.

### 17.4 Model licenses versus software licenses

For every deployed learned artifact, record its exact model revision, full parameter count, weight license, tokenizer/config revisions, and any fine-tuning lineage. Keep aggregate deployed parameters below 8B by default.

Software dependencies have their own licenses. BSD-licensed pandas, SciPy, or scikit-learn are not automatically prohibited by a rule about the final model's license. Preserve notices and verify redistribution obligations; do not invent a requirement that every dependency must be MIT/Apache.

The core uses locally trained artifacts and supplied data. Optional pretrained components use verified eligible checkpoints. libpostal, external gazetteers, scraped business dictionaries, outside ER datasets for training, external teacher APIs, and oversized teachers are excluded from the chosen design.

### 17.5 Offline execution and data provenance

Acquire pinned dependencies and permitted model weights during environment preparation. Matching runs use local artifacts and supplied records only.

Set appropriate library offline modes and verify execution in an environment with network egress disabled. A source-code search for `requests` is useful hygiene but does not prove that transitive dependencies make no network calls.

Learning abbreviation or transliteration correspondences from supplied fitting pairs is documented training. External address/business augmentation is not used. Test pseudo-label training is off by default; it is unnecessary for the core and should not be declared permitted merely because it is self-generated.

### 17.6 Package contents

```text
<team_name>_submission.zip
  output/
    matching_results.tsv
    candidate_pairs.tsv
  code/
    business_entity_resolution/
      src/
      README.md
      requirements.txt
      pyproject.toml
      configs/
      artifacts/final/
  Documentation_template.md
```

Include the final learned models and all preprocessing/calibration/tokenizer assets needed for inference, or an explicitly permitted reproducible local reconstruction path. A model-name string that downloads unspecified current weights is not a self-contained artifact.

The dataset is a documented input rather than an accidental hidden dependency. The README supplies exact commands, working directories, environment setup, required hardware, expected runtime from measurement, and output locations. Pin actual tested versions during implementation; do not invent version pins in this plan.

### 17.7 Two reproduction modes

1. **Inference reproduction:** supplied raw data + bundled selected artifacts regenerate both TSVs. This is the minimum release requirement.
2. **Training reproduction:** split manifests, training configuration, preprocessing, base weights where needed, seeds, and environment reproduce the training procedure.

Fixed seeds alone do not guarantee bitwise-identical GPU training across devices. Preserve the selected final checkpoint and deterministic export order. Document remaining numerical variability rather than promising impossible cross-hardware identity.

### 17.8 Methodology document

Fill the provided `Documentation_template.md` during execution, with measured values rather than this plan's hypotheses. [TEMPLATE]

In particular, include candidate totals and distribution, blocking recall and oracle ceiling on validation, support-boundary definition, ranking-update response, all model/license choices, error categories, and end-to-end reproduction instructions.

The public leaderboard is a coarse external signal, not the training objective. Log the exact pipeline run behind every submission. The number of uploads follows actual portal limits, not an invented three-submission rule.

---

<a id="handoff"></a>
## 18. Resolved handoff questions

These decisions replace the old “wait for another plan” handoff. Remaining unknowable facts have an operational default, not a fabricated answer.

| Question | Decision/evidence | Consequence |
|---|---|---|
| Is the dataset available? | Yes, all train/test TSVs are under `student_resource/dataset/` | Start M0 immediately in the implementation stage. |
| Is the validator/template available? | Yes; both were inspected | Integrate the real interfaces, including `--check-ids`. |
| What is the singleton rate? | Training: 123,247/2,206,821 = 5.58482% | Use the observed baseline; test rate remains unknown. |
| Do S2/S3 contain repeated identities? | Yes; truth assigns multiple records from each source to the same S1 | Support all aliases; no one-to-one S1 capacity. |
| Can one target belong to several S1s? | No such case among 7,638,365 training links; S1 is officially deduplicated | Test nullable inverse ownership, with no S1 cardinality cap. |
| Does France need to be guessed into the output? | No; 259,452 actual France S1 rows are present | Export every authoritative test S1 ID. |
| Is France one third of test? | No; full-test reference share is 14.97520% | Replace the speculative weighting. |
| Are training/test raw S1 records identical? | No exact raw payload overlap found | Do not rely on a memorized exact-payload lookup. |
| Does ID ordering carry identity? | No justified semantic use | IDs only join, identify, hash-split, and break harmless output-order ties. |
| Should we use libpostal/geocoding? | Not in this design | Raw-preserving parsing and train-derived maps suffice for the core. |
| Should an external or >8B teacher generate labels? | No | Abundant provided labels remove the need; do not rely on an unapproved interpretation. |
| How strict is the parameter limit? | Use actual total parameters, including embeddings/heads; keep deployed aggregate below 8B | Avoid dependence on ambiguity around model names or ensembles. |
| Must every software package be MIT/Apache? | That is not the stated rule | Distinguish dependency notices from model eligibility. |
| Should we generate synthetic France? | No as a validation substitute | Use legitimate country/noise stress tests and acknowledge the transfer uncertainty. |
| Should we lower France's match prior? | No evidence supports doing so | Pooled fallback unless a valid labeled experiment supports a broader robust policy. |
| Should we pseudo-label test for training? | Off by default | Core is train-supervised; revisit only if rules explicitly permit and offline evidence warrants it. |
| Does the generator use identical train/test noise? | Unknown | Do not assume it; test multiple stress slices and inspect unlabeled drift. |
| Is a global threshold inadequate by definition? | No | Build and retain it as a measured baseline. |
| What if calibration-dependent decisions fail? | Return to the best validated threshold/owner policy | No arbitrary precision-correlation cutoff or forced GFM. |
| Can we train on all labels after the audit? | Not while reusing the old calibration/audit claim | Default release uses the audited checkpoint. A refit retains calibration reserves and repeats checkpoint-specific calibration and audit. |
| How are thresholds tested at realistic scale? | D/K/A are protected focal cohorts in large declared background catalogs | Match reference competition by country, include incoming background false links, and report catalog/traffic shortfalls. |
| Should we cluster all targets and expand matches? | Not in the core | Reverse retrieval already preserves multiple variants; graph retrieval needs evidence. |
| Which model is first? | One LightGBM matcher on compact candidates | Add neural components only for diagnosed residual loss. |
| Which neural experiment is first? | Multilingual-e5-small retrieval, then supervised compaction if useful | Directly targets cross-script recall and candidate-size ranking. |
| Do we need fixed 50 candidates? | No | Measure b=1/2/3 reverse baselines, adaptive compaction, and a forward comparator. |
| What is the final-ranking formula? | Not specified in the supplied update | Keep a Pareto frontier; use the documented internal non-inferiority default. |
| What are event duration, upload quota, and available GPUs? | Not established by the materials | Use milestone order and measured runtime; keep a CPU fallback. |
| What is the private split distribution? | Unknown | Avoid public-only tuning; report robust slices and preserve audit integrity. |
| Are ID lists ordered predictions? | No, matching is set-valued; order is not identity evidence | Sort deterministically for reproducibility. |
| Is default validator PASS sufficient? | No | Strict support/ID/completeness audit plus official stronger invocation. |
| What should be implemented next? | M0, M1, then M2 | The design is ready to execute without choosing between unrelated architecture menus. |

---

<a id="adversarial"></a>
## 19. Adversarial review and bounded future work

### 19.1 How the selected strategy could fail

| Challenge to our own recommendation | Evidence/uncertainty | Response and decisive experiment |
|---|---|---|
| Reverse lexical retrieval might need too many queries | Test has 5.75 targets per reference | Benchmark forward versus reverse at equal pair budgets and production catalog size. |
| A two-reference budget may drop ambiguous true owners | Common names are frequent; no retrieval result exists yet | Measure raw and compacted owner recall, especially common-name/missing-address cases. |
| Lexical signals may fail across scripts | Native-script target text is observed at scale | Small multilingual retrieval is the first neural test, with hard-slice reporting. |
| Generic multilingual embeddings may retrieve semantically similar but different businesses | Retrieval semantics are not identity | Supervised hard negatives and numeric/address evidence; evaluate at small k. |
| Country partitioning could miss a mislabeled record | No observed training cross-country link, but test labels are absent | Track country integrity and provide a measured rescue path if contradictory evidence appears. |
| LightGBM may miss difficult contextual equivalences | Strong structured evidence does not solve every alias | Cross-encoder challenger on fixed support and measured errors. |
| Owner exclusivity could hurt macro-F despite being truth-consistent | Row utilities are nonlinear; score uncertainty matters | Compare nullable-owner and unconstrained policies on complete rows. |
| Smaller validation catalogs could overstate retrieval quality | Similarity-search difficulty changes with catalog size | Large locked audit and explicit catalog-size stress tests. |
| A calibrated pair score may still give poor set decisions | Within-entity dependence and omitted truths exist | Simple row-aware challenger, then GFM only with learned joint statistics. |
| Candidate compression could look good while concealing expensive upstream work | Small final files alone prove little | Log raw retrieval counts, inspected postings/work, classifier support, and end-to-end runtime. |
| Neural inference could consume the deadline | Millions of pairs, hardware unmeasured | Measure before promotion; route selectively; keep a complete CPU release candidate. |
| A simpler exact-evidence policy might perform almost as well | Dataset may contain strong structured identity signals | Keep the conservative-rule baseline and quantify its residual errors before adding complexity. |

### 19.2 Stronger avenues not dismissed

These are ordered research directions, each with a trigger rather than an obligation:

1. **Learned compact retrieval:** high raw recall but excessive final candidate size. Train target→reference representations and measure recall at b=1/2/3.
2. **Nullable ranking:** correct reference usually present, but the binary matcher struggles with competing references. Compare a candidate-list model with an explicit NONE class; calibrate abstention.
3. **Cross-source bridge retrieval:** a target has no usable direct match evidence, while another target variant supplies it. Use bridges to propose candidates before matching, not to auto-merge connected components. Include every scored addition in support.
4. **Count-conditioned utility:** pair discrimination is strong but row-level start/add decisions remain systematically wrong. Fit shared GFM statistics or direct expected-utility regressors using full truth cardinalities and out-of-fold contexts.
5. **Noise-channel modeling:** observed matched-pair edits expose repeatable source-specific transformations. Learn equivalence/edit likelihoods with hard-negative controls instead of inventing unobserved transformations.
6. **Distillation:** a selective neural model provides reproducible gains but is too slow for broad use. Distill within the provided training data using eligible local teachers; compare against direct supervised training.
7. **Uncertainty guarantees:** only after defining exchangeability and selection assumptions. Generic conformal prediction does not automatically guarantee precision after a France distribution shift.

Publishing challenge data, labels, or derived examples later requires the organizer's data-release permission. An open research agenda is not permission to redistribute the supplied corpus.

### 19.3 What future workers must preserve

- Official rule precedence and exact scorer semantics.
- Data and support fingerprints.
- Honest separation of observed results from hypotheses.
- The ability to reproduce a complete compact baseline.
- Negative experimental results and changes to validation status.
- The right to replace this architecture when a controlled experiment proves a better one.

This document is a decision record, not a requirement to defend the current design indefinitely.

---

<a id="evidence"></a>
## 20. Research evidence: what was verified and what it supports

### 20.1 Primary-source evidence ledger

| Source | Verified finding and evaluation setting | Justified use here | What it does not establish |
|---|---|---|---|
| **Sparkly**, PVLDB 2023, Table 2 | On its 15-dataset evaluation, recall ranges 92.5–100% at k=10, 96.4–100% at k=20, and 98.7–100% at k=50. Its design indexes the smaller table and probes from the larger. [SPARKLY] | Indexed lexical top-k is a serious baseline; direction and budget matter | This challenge's recall at a chosen k, or a prohibition on all threshold-based compaction |
| **Sparkly**, additional Companies analysis | Reports 62% recall at k=50 on long company descriptions; without TF weighting, 33% [SPARKLY] | Representation and dataset differences matter | Universal near-perfect recall from lexical blocking |
| **Sudowoodo**, Table VII | Abt–Buy: 3,276 candidates at 88.6% recall, versus DL-Block 21,600 at 87.2%; the cited 84.8% reduction occurs at this operating region [SUDO] | Contrastive learned blocking is worth testing for compactness | An 84.8% reduction at 99% recall on our data |
| **Ditto**, employer case study §5 | Original tables 789,409 and 412,418; deduplicated to 788,094 and 62,511. 10,652,249 blocked pairs; 20K labeled sampled pairs, 39% positive, 3:1:1 split; held-out **96.53 pairwise F1** [DITTO] | Supervised pair transformers can be strong matchers | End-to-end macro-F₀.₅ on this task, French quality, or a numerical ceiling for our system |
| **ComEM**, §§3.1–3.4, Table 4 | Single-match selection setting. GPT-3.5: 64.02→81.60; GPT-4o-mini: 67.80→82.26. Average gain =16.02 points. Qwen2-7B selection: 62.11→74.93 [COMEM] | Candidate competition can help; inverse nullable ranking is an experiment | Direct validation of multi-ID S1 selection or a +16.02 open-7B claim |
| **AnyMatch**, Tables 2–4 | GPT-2 124M, leave-one-dataset-out over nine benchmarks: mean F1 81.96 versus GPT-4 86.36 [ANYMATCH] | Small supervised/transfer-trained matchers deserve consideration | Permission to train on external ER corpora, or deployment throughput on this host |
| **GFM / Bayes-optimal F research** | Exact expected-F optimization uses joint label/cardinality statistics; independence-based dynamic programs have explicit assumptions [GFM1] [GFM2] [GFM3] [YE] | Correct optional decision theory | Guaranteed gains from inaccurate posteriors or independence assumptions |
| **TransClean** | The reported +24.42 F1-point improvement concerns multi-source benchmark configurations with targeted/manual labeling, fine-tuning, pruning, and recovery [TRANSCLEAN] | Structured consistency can guide a later investigation | A predicted gain from automatic lightweight transitive closure |

All quoted research numbers retain their original metric and setting. None is converted into a projected challenge score. General-language scores such as MMLU and generic retrieval scores such as MIRACL are not used to select the winning ER model.

### 20.2 References and implementation sources

- **Official local statement:** [student_resource/README.md][PS]. The candidate-ranking update is supplied in the review request and quoted in §3.
- **Official validator:** [student_resource/utils/validate_submission.py][VALIDATOR].
- **Official methodology template:** [student_resource/Documentation_template.md][TEMPLATE].
- **Sparkly:** Paulsen, Govind, Doan, *Sparkly: A Simple Yet Surprisingly Strong TF/IDF Blocker for Entity Matching*, PVLDB 16(6), 2023. [Paper][SPARKLY].
- **Sudowoodo:** Wang, Li, Wang, *Sudowoodo: Contrastive Self-Supervised Learning for Multi-purpose Data Integration and Preparation*. [Paper][SUDO].
- **Ditto:** Li et al., *Deep Entity Matching with Pre-Trained Language Models*. [Full paper, v3][DITTO].
- **ComEM:** Wang et al., *Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching*. [Full paper, v3][COMEM].
- **AnyMatch:** Zhang et al., *AnyMatch*. [Full paper, v2][ANYMATCH].
- **GFM:** Dembczyński et al., *An Exact Algorithm for F-Measure Maximization*, NeurIPS 2011. [Paper][GFM1].
- **General Fβ estimation:** Dembczyński et al., *Optimizing F-Measure in Multi-Label Classification: Plug-in Rule Approach versus Structured Loss Minimization*, ICML 2013. [Paper][GFM2].
- **Dependence and Bayes decisions:** Waegeman et al., *On the Bayes-Optimality of F-Measure Maximizers*, JMLR 2014. [Paper][GFM3].
- **Independent-label dynamic programming:** Ye et al., *Optimizing F-measure: A Tale of Two Approaches*, ICML 2012. [Paper][YE].
- **TransClean:** de Meer Pardo et al., *Finding False Positives in Multi-Source Entity Matching under Real-World Conditions via Transitive Consistency*. [Paper][TRANSCLEAN].
- **Tantivy:** [engine and license][TANTIVY], [Python documentation and license][TANTIVY_PY].
- **LightGBM:** [parameters and training semantics][LGBM].
- **FAISS:** [index-selection guidance][FAISS]. Measure the actual index rather than copying an unrelated benchmark's recall.
- **Model cards/configurations:** [multilingual-e5-small][E5], [mDeBERTa-v3-base][MDEBERTA], [XLM-R-base][XLMR], [Qwen3-4B][QWEN].
- **Withdrawn citation check:** [arXiv 2608.18115][BADREF], whose actual topic does not support the old ER-stability claims.

[PS]: student_resource/README.md
[VALIDATOR]: student_resource/utils/validate_submission.py
[TEMPLATE]: student_resource/Documentation_template.md
[SPARKLY]: https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf
[SUDO]: https://arxiv.org/abs/2207.04122
[DITTO]: https://arxiv.org/html/2004.00584v3
[COMEM]: https://arxiv.org/html/2405.16884v3
[ANYMATCH]: https://arxiv.org/html/2409.04073v2
[GFM1]: https://papers.nips.cc/paper/4389-an-exact-algorithm-for-f-measure-maximization
[GFM2]: https://proceedings.mlr.press/v28/dembczynski13.html
[GFM3]: https://jmlr.org/papers/v15/waegeman14a.html
[YE]: https://arxiv.org/abs/1206.4625
[TRANSCLEAN]: https://arxiv.org/html/2506.04006v1
[TANTIVY]: https://github.com/quickwit-oss/tantivy
[TANTIVY_PY]: https://tantivy-py.readthedocs.io/en/latest/
[LGBM]: https://lightgbm.readthedocs.io/en/stable/Parameters.html
[FAISS]: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
[E5]: https://huggingface.co/intfloat/multilingual-e5-small
[MDEBERTA]: https://huggingface.co/microsoft/mdeberta-v3-base
[XLMR]: https://huggingface.co/FacebookAI/xlm-roberta-base
[QWEN]: https://huggingface.co/Qwen/Qwen3-4B
[BADREF]: https://arxiv.org/abs/2608.18115

---

<a id="provenance"></a>
## 21. Audit provenance and how to extend the plan

### 21.1 This revision's work

- Read all 1,224 lines of the previous master document.
- Read the actual organizer README, validator, and methodology template.
- Stream-profiled every supplied source TSV for counts, country composition, missingness, script indicators, and field lengths.
- Checked every training truth row and all positive target references for ownership, country agreement, and referential integrity.
- Examined the deterministic 10,996-reference sample described in §4.5 for field agreement and ambiguity.
- Checked full training S1 raw-payload duplication and exact train/test S1 raw-payload overlap.
- Independently rechecked the critical mathematical claims and research settings.
- Verified scorer examples, the singleton counterexample, a dependent-label prefix counterexample, and the support-oracle formula with exact/rational calculations and small exhaustive checks.

No competition matcher was trained, no candidate-retrieval quality result was fabricated, no output submission was produced, and no challenge entity was looked up externally. The intended repository edit for this stage is `master.md` only.

### 21.2 Input fingerprints

These fingerprints identify the exact corpus behind the observed numbers. Recompute the profile when any input changes.

| File, relative to `student_resource/dataset/` | Bytes | SHA-256 |
|---|---:|---|
| `train/train_source1.tsv` | 210069713 | `591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973` |
| `train/train_source2.tsv` | 489301488 | `6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed` |
| `train/train_source3.tsv` | 503705637 | `67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9` |
| `train/train_ground_truth.tsv` | 127015583 | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` |
| `test/test_source1.tsv` | 175022086 | `3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5` |
| `test/test_source2.tsv` | 509456422 | `79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd` |
| `test/test_source3.tsv` | 506002772 | `850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035` |

Previous master-document SHA-256 before this revision:

```text
568c58e893975d305b835b51c12f043a459bb58f2aed8e5e6c50e3b52c41fbd3
```

### 21.3 Reproduction specification for the structural audit

The implementation-stage `profile` command should reproduce the review's aggregates using:

1. UTF-8 `csv.reader(..., delimiter="\t")` over each source, validating exactly four columns.
2. Per-country row counts; blank means `not field.strip()`.
3. Non-ASCII means `not field.isascii()`; Indic-range detection is `[\u0900-\u0dff]`.
4. Character-length histograms with nearest-rank p50/p95/p99, explicitly not token counts.
5. Ground-truth comma-list parsing with an empty cell mapped to an empty set, after duplicate checks.
6. S1→country mapping and target→owner checks over all positive links.
7. Exact raw-payload equality for the duplicate/overlap checks; no normalization in those two full-scan checks.
8. The deterministic hash sample in §4.5, retaining all positive siblings for each selected reference.
9. Sample comparison normalization: NFKC, case-fold, retain alphanumerics and Unicode mark categories, map other characters to spaces, collapse whitespace.

Do not conflate these inexpensive structural measurements with retrieval or matcher evaluation. Persist their reproducible implementation and reports during M0; this planning revision intentionally creates no additional project files.

### 21.4 Promotion record for the next revision

When implementation produces evidence, update the relevant decision with:

```text
question -> experiment -> data/split/support hashes -> measured results
         -> error explanation -> selected change or rejection
```

Replace hypotheses with results, retire failed ideas, and preserve the audit trail. The next engineer should be able to answer both **what to run** and **why that configuration earned its place**.

**Final commitment:** start with M0–M2, establish the compact CPU baseline, then spend the next unit of effort on whichever stage the measured loss decomposition identifies. That is the path to the strongest defensible submission, rather than the largest untested architecture.
