"""CLI orchestration for the entity-resolution pipeline.

Every stage is resumable: shard sidecar fingerprints make re-runes cheap,
and mismatched fingerprints force recomputation instead of silent mixing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ber import pipeline as pl
from ber import stages as st
from ber.training import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "student_resource" / "dataset"
DEFAULT_VALIDATOR = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"

COHORT_ROLES = {"fit": "F", "D": "D", "K": "K", "A": "A", "A2": "A2", "test": None}


def _data_dir(args) -> Path:
    return Path(args.data_dir)


def _quotas_and_densities(data_dir: Path) -> tuple[dict[str, int], dict[str, dict[int, float]]]:
    rep = json.loads((data_dir / "reports" / "ingest.json").read_text(encoding="utf-8"))
    test_refs = rep["country_distribution"]["test_s1"]
    quotas = {c: int(n) for c, n in test_refs.items()}
    densities = {
        country: {source: rep["country_distribution"][f"test_s{source}"].get(country, 0) / count
                  for source in (2, 3)}
        for country, count in quotas.items()
    }
    return quotas, densities


def main() -> int:
    ap = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[2] / "data"))
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--workers", type=int, default=6)

    p = sub.add_parser("split")
    p.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("reserve-audit")
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=13)

    p = sub.add_parser("env")
    p.add_argument("--cohort", required=True, choices=list(COHORT_ROLES))
    p.add_argument("--shard-size", type=int, default=100_000)

    p = sub.add_parser("index")
    p.add_argument("--cohort", required=True)
    p.add_argument("--threads", type=int, default=6)

    for name in ("dfmap", "retrieve", "features"):
        p = sub.add_parser(name)
        p.add_argument("--cohort", required=True)
        if name == "retrieve":
            p.add_argument("--budget", type=int, default=st.BUDGET)
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--lane-k", type=int, default=st.LANE_K)
            p.add_argument("--max-terms", type=int, default=st.MAX_TERMS)
            p.add_argument("--joint-k", type=int, default=0)
        if name == "features":
            p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("train")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("select")  # D
    p = sub.add_parser("calibrate")  # K
    p = sub.add_parser("audit")  # A or freshly reserved A2
    p.add_argument("--cohort", choices=("A", "A2"), default="A")

    p = sub.add_parser("test")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "output"))
    p.add_argument("--budget", type=int, default=st.BUDGET)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lane-k", type=int, default=st.LANE_K)
    p.add_argument("--max-terms", type=int, default=st.MAX_TERMS)
    p.add_argument("--joint-k", type=int, default=0)

    p = sub.add_parser("validate")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "output"))
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--official-validator", default=str(DEFAULT_VALIDATOR))

    args = ap.parse_args()
    data_dir = _data_dir(args)

    if args.stage == "ingest":
        print(json.dumps(pl.stage_ingest(Path(args.dataset_dir), data_dir, args.workers), indent=2)[:2000])
    elif args.stage == "split":
        print(json.dumps(pl.stage_split(data_dir, seed=args.seed), indent=2))
    elif args.stage == "reserve-audit":
        print(json.dumps(pl.stage_reserve_audit(data_dir, args.fraction, args.seed), indent=2))
    elif args.stage == "env":
        role = COHORT_ROLES[args.cohort]
        if args.cohort == "test":
            print(json.dumps(pl.stage_env_test(data_dir, shard_size=args.shard_size), indent=2))
        else:
            quotas, densities = _quotas_and_densities(data_dir)
            print(json.dumps(pl.stage_env_train(
                data_dir, args.cohort, role, quotas, densities, seed=7,
                shard_size=args.shard_size), indent=2))
    elif args.stage == "index":
        print(json.dumps(st.stage_index(data_dir, args.cohort, threads=args.threads), indent=2))
    elif args.stage == "dfmap":
        print(json.dumps(st.stage_dfmap(data_dir, args.cohort), indent=2))
    elif args.stage == "retrieve":
        print(json.dumps(st.stage_retrieve(data_dir, args.cohort, budget=args.budget,
                                           workers=args.workers, lane_k=args.lane_k,
                                           max_terms=args.max_terms, joint_k=args.joint_k), indent=2))
    elif args.stage == "features":
        print(json.dumps(st.stage_features(data_dir, args.cohort, workers=args.workers), indent=2))
    elif args.stage == "train":
        cfg = TrainConfig(num_rounds=args.rounds, num_threads=args.threads, seed=args.seed)
        print(json.dumps(st.stage_train(data_dir, cfg), indent=2))
    elif args.stage == "select":
        print(json.dumps(st.stage_select(data_dir, "D", "D"), indent=2))
    elif args.stage == "calibrate":
        sel = json.loads((data_dir / "reports" / "select_D.json").read_text(encoding="utf-8"))
        print(json.dumps(st.stage_calibrate(data_dir, "K", "K",
                                            selected_policy=sel["selected_policy"]), indent=2))
    elif args.stage == "audit":
        print(json.dumps(st.stage_audit(data_dir, args.cohort, COHORT_ROLES[args.cohort]), indent=2))
    elif args.stage == "test":
        print(json.dumps(st.stage_test(data_dir, Path(args.output_dir), budget=args.budget,
                                        workers=args.workers, lane_k=args.lane_k,
                                        max_terms=args.max_terms, joint_k=args.joint_k), indent=2))
    elif args.stage == "validate":
        print(json.dumps(st.run_validation(data_dir, Path(args.output_dir),
                                            Path(args.dataset_dir),
                                            Path(args.official_validator)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
