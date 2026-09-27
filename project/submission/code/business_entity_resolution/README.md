# Business Entity Resolution — Amazon ML Challenge 2026

Retrieval-first entity resolution over three noisy business-record sources,
built and measured end-to-end on the full dataset (12.5M train records,
11.7M test records, 7.6M labeled links).

## Approach (one paragraph)

Index the deduplicated Source-1 references per country with a merged
trigram+word+accent-fold term stream (tantivy BM25). For every Source-2/3
record, retrieve the top references in two field lanes (name, address),
fuse the lanes with reciprocal-rank fusion (offset 60), add a joint
name-AND-address conjunction rescue lane (top-2), and keep the top-2 fused
(strict reverse budget, exact ties expand) plus the joint rescues. The
survivors are scored by a two-stage LightGBM cascade: stage 1 over 46
pair-comparison and retrieval features, stage 2 adding competition features
computed from stage-1 scores (the target's best other reference, rank and
margin inside its reference group). The final match sets come from an
owner-exclusive policy: each target goes to at most its best reference,
above a threshold τ. All training and selection happen inside a **dense,
test-faithful environment** (see below).

## Leaderboard history (what mattered)

| Upload | What changed | Public LB |
|---|---|---:|
| v5-joint | threshold policy tuned on sparse cohorts (local audit 0.976) | 0.815 |
| v6-safe | same model, owner-exclusive policy, stricter τ | 0.877 |
| v7-full_owner_t85 | model retrained in the dense env W + v5 features + stage 2, owner policy τ=0.85 | 0.947 |
| **v8_matched** | + v6 Indic transliteration skeleton features, owner τ=0.875 | **0.952** |

**Root cause of the 0.976 → 0.815 gap:** the original cohort environments
sized query traffic as test density × *focal* references, which gave
0.3–0.6 targets per catalog reference, and they scored only focal
references. Test has 5.76 targets per reference, ~40% of them without an
owner, and every reference is scored. The false positives on sibling and
ownerless records were therefore invisible locally.

**Fix: dense env `W`** (`pipeline.stage_env_dense`):
- The catalog per country is sized to test: every D/K/A/A2/F reference plus
  B fill.
- Traffic is every target owned by a catalog reference, plus orphans
  (unlinked targets and targets of references left out of the catalog), up
  to the observed test density per catalog reference. That gives US 5.76
  targets/ref with 40% ownerless; India 5.10 with 32%.
- Training rows are edges whose reference *and* target are F/B-role.
  Policies are selected on D, checked on K, and A2 is left for a one-shot
  audit. Every cohort's references are scored among all their test-like
  competitors.

## Measured results (dense env W, D/K cohorts, 110k refs each)

| Model / policy | W-D | W-K | US | India | Singletons |
|---|---:|---:|---:|---:|---:|
| old v5 model, old policy | – | 0.898 | – | – | – |
| old v5 model, owner policy (W-selected) | 0.938 | 0.939 | 0.956 | 0.913 | 0.905 |
| new stage 1 (46 feats) | 0.954 | 0.955 | 0.967 | 0.938 | 0.944 |
| new stage 2, owner τ=0.707 (W optimum) | 0.960 | 0.960 | 0.972 | 0.943 | 0.967 |
| **new stage 2, owner τ=0.85 (shipped)** | 0.957 | – | 0.969 | 0.939 | 0.981 |

- **Cross-scale robustness:** the frozen W release, scored in a separate
  20%-scale dense env (Wm, different catalog statistics), gets K 0.9716,
  against 0.974 for a policy re-tuned there. This matters for France, whose
  catalog is smaller and has no labels.
- **Why τ=0.85 over the W optimum:** the real test has about 2× more edges
  in the uncertain score band (more near-duplicate distractors). τ=0.85
  costs 0.003 on W, and the LB rose from 0.877 to 0.947.

## Reproduction

Environment (uv, Python 3.13):

```bash
cd project
uv sync
uv run pytest                    # unit tests incl. dense-env builders, owner/EFM policies
```

Pipeline (all stages resumable; shard fingerprints reject stale mixes):

```bash
DATA=../student_resource/dataset
export PYTHONPATH=src
uv run python -m ber.cli ingest  --dataset-dir $DATA --workers 7
uv run python -m ber.cli split   --seed 7
uv run python -m ber.cli reserve-audit --fraction 0.05 --seed 13
# dense, test-faithful training/selection environment
uv run python -m ber.cli env      --cohort W
uv run python -m ber.cli index    --cohort W --threads 16
uv run python -m ber.cli dfmap    --cohort W
uv run python -m ber.cli retrieve --cohort W --budget 2 --joint-k 2 --workers 14
uv run python -m ber.cli features --cohort W --workers 14
# test candidates + features
uv run python -m ber.cli env      --cohort test
uv run python -m ber.cli index    --cohort test --threads 16
uv run python -m ber.cli dfmap    --cohort test
uv run python -m ber.cli retrieve --cohort test --budget 2 --joint-k 2 --workers 14
uv run python -m ber.cli features --cohort test --workers 14
# two-stage model: out-of-fold stage 1 + stage 2; policy selected on W-D
uv run python -m ber.dense_run train --cohort W --variant full --threads 14
uv run python -m ber.dense_run evalx --cohort W --variant full --update-release
# shipped operating point: owner policy (gamma 0.02) with tau 0.85
#   (data/models/dense/full_owner_t85/release.json)
uv run python -m ber.dense_run test --variant full_owner_t85 --output-dir output/final
uv run python ../student_resource/utils/validate_submission.py \
  --matching output/final/matching_results.tsv --candidate output/final/candidate_pairs.tsv \
  --test-dir $DATA/test --check-ids
```

Keep retrieval/feature workers at or below ~14 on a 32-vCPU / 248 GiB
machine. Each worker holds a full country index plus a term-frequency map,
and 31 workers exhausted the machine.

Wall-clock on AWS g6e.8xlarge (32 vCPU, CPU only): dense env 85 s, index
~80 s, W retrieval ~45 min (7.95M queries), W features ~4 min, test
features ~6 min, two-stage training ~30 min (16.0M training edges, 500+400
rounds, 127 leaves), test scoring + export ~6 min.

## Design decisions (measured)

- **Dense env over sparse cohorts:** the sparse cohorts overstated the LB
  by 0.16. W's owner-policy K for the old model (0.939) tracks the LB
  (0.877), and the new release's W score (0.957) came within 0.01 of the LB
  (0.947).
- **Owner exclusivity:** ground truth assigns each target to at most one
  reference. The old export gave 154k targets to 2+ references, while every
  new export has 0.
- **v5 features:**
  - single-digit house numbers (dropped before by token-length filters;
    they affect 24% of French addresses)
  - accent-folded Jaro-Winkler for name and address
  - a "name core" that strips legal forms and generic words, to separate
    siblings such as "Advik Farms" and "Advik Properties"
- **Stage-2 competition features:** +0.005 W-K overall, with the gain
  concentrated in singletons (0.944 → 0.967).
- **Expected-F0.5 per-reference selection (EFM)** was implemented and
  tested. It tied on D (+0.0004) but lowered singletons to 0.935, so it was
  not shipped.
- **Raw-BM25-free variant:** equally robust across scales (Wm K 0.971 vs
  0.9716), so the full feature set ships.
- **Reverse retrieval** with budget 2 plus the joint rescue lane is
  unchanged from earlier revisions (candidate oracle 0.985 in W).

## Layout

```
src/ber/            pipeline modules (records, normalize, splits, env,
                    retrieval, candidates, features, training, decisions,
                    metrics, export, audit, pipeline, stages, cli,
                    dense [dense-env training/selection/policies], dense_run [driver])
src/ber/tests/      unit tests incl. official-validator parity and dense-env logic
data/               generated artifacts (gitignored)
output/             matching_results.tsv + candidate_pairs.tsv
```

## Honesty notes

- The W-D and W-K numbers are selection and check cohorts; τ=0.85 was
  chosen with leaderboard feedback (a single operating-point change).
- The remaining loss is mostly blocking: the candidate oracle in W is
  0.985, and India (0.94) is the weakest country (transliterated and
  native-script names).
- No external data, APIs, or lookups are used at any stage. Dependencies:
  numpy, pyarrow, lightgbm (MIT), tantivy (MIT), rapidfuzz (MIT), scipy
  (BSD).
- All learned components are LightGBM models trained locally on the
  supplied data; no pretrained weights are used.
