"""Dense-environment driver: gate / train / evalx / audit / test.

    python -m ber.dense_run gate  --cohort W
    python -m ber.dense_run train --cohort W --variant full
    python -m ber.dense_run train --cohort W --variant nobm25 --drop name_lane_score,addr_lane_score,joint_score
    python -m ber.dense_run evalx --cohort Wm --variant full     # cross-scale robustness
    python -m ber.dense_run audit --cohort W --variant full
    python -m ber.dense_run test  --variant full --output-dir output/v6

Each variant keeps its models and frozen release under data/models/dense/<variant>/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ber import dense as dn
from ber.features import FEATURE_NAMES
from ber.records import sha256_file
from ber.training import load_model

N_V4 = 37  # the frozen v5-release model was trained on the first 37 features


def _models_dir(data_dir: Path, variant: str | None = None) -> Path:
    d = data_dir / "models" / "dense"
    if variant:
        d = d / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def _feature_names(drop: str | None) -> tuple[str, ...]:
    dropped = {x for x in (drop or "").split(",") if x}
    unknown = dropped - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown features to drop: {sorted(unknown)}")
    return tuple(n for n in FEATURE_NAMES if n not in dropped)


def cmd_gate(data_dir: Path, cohort: str, old_model: Path, old_bundle: Path) -> None:
    """Score the submitted (0.815) model inside the dense env."""
    from ber.stages import platt_apply

    t0 = time.perf_counter()
    ctx = dn.build_context(data_dir, cohort, FEATURE_NAMES[:N_V4])
    bundle = json.loads(old_bundle.read_text())
    bst = load_model(old_model)
    s = platt_apply(bst.predict(ctx.edges.X), bundle["calibrator"]["a"], bundle["calibrator"]["b"])
    out = {"n_edges": int(s.size),
           "old_policy_D": dn.evaluate(ctx, "D", s, bundle["policy"]),
           "old_policy_K": dn.evaluate(ctx, "K", s, bundle["policy"])}
    sel = dn.select_policy(ctx, "D", s)
    out["reselected_on_D"] = sel
    out["reselected_K"] = dn.evaluate(ctx, "K", s, sel["best"])
    out["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    dn.report(data_dir, f"gate_{cohort}", out)
    (_models_dir(data_dir) / f"safe_policy_{cohort}.json").write_text(json.dumps(sel["best"]))


def cmd_train(data_dir: Path, cohort: str, variant: str, names: tuple[str, ...],
              rounds: int, threads: int, stage2: bool, rounds2: int) -> None:
    t0 = time.perf_counter()
    md = _models_dir(data_dir, variant)
    ctx = dn.build_context(data_dir, cohort, names)
    tm = dn.train_mask(ctx)
    out: dict = {"variant": variant, "features": list(names),
                 "n_edges": int(tm.size), "n_train_edges": int(tm.sum()),
                 "train_pos_rate": float(ctx.pos[tm].mean())}
    print(f"[dense:{variant}] context ready in {time.perf_counter() - t0:.0f}s; "
          f"train edges {tm.sum()}", flush=True)

    s1, models = dn.stage1_oof(ctx, rounds, threads)
    for k, m in enumerate(models):
        m.save_model(str(md / f"stage1_fold{k}.txt"))
    print(f"[dense:{variant}] stage-1 done at {time.perf_counter() - t0:.0f}s", flush=True)
    sel1 = dn.select_policy(ctx, "D", s1)
    out["stage1"] = {"select_D": sel1, "K": dn.evaluate(ctx, "K", s1, sel1["best"]),
                     "D": dn.evaluate(ctx, "D", s1, sel1["best"])}
    dn.report(data_dir, f"train_{cohort}_{variant}_partial", out)
    best = {"variant": variant, "stage": "stage1", "features": list(names),
            "policy": sel1["best"], "D": sel1["best"]["macro_f"]}

    if stage2:
        G = dn.group_features(ctx.edges.ref, ctx.edges.src, ctx.edges.tgt, s1)
        X2 = np.concatenate([ctx.edges.X, G], axis=1)
        del G
        bst2 = dn.fit_lgbm(X2[tm], ctx.pos[tm].astype(np.int8), rounds2, threads, seed=7)
        bst2.save_model(str(md / "stage2.txt"))
        s2 = bst2.predict(X2)
        del X2
        sel2 = dn.select_policy(ctx, "D", s2)
        out["stage2"] = {"select_D": sel2, "K": dn.evaluate(ctx, "K", s2, sel2["best"]),
                         "D": dn.evaluate(ctx, "D", s2, sel2["best"])}
        if sel2["best"]["macro_f"] >= best["D"] + 0.002:
            best = {**best, "stage": "stage2", "policy": sel2["best"], "D": sel2["best"]["macro_f"]}
    best["model_fingerprints"] = {p.name: sha256_file(p) for p in sorted(md.glob("stage*.txt"))}
    (md / "release.json").write_text(json.dumps(best, indent=2))
    out["release"] = best
    out["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    dn.report(data_dir, f"train_{cohort}_{variant}", out)


def _release(data_dir: Path, variant: str) -> dict:
    return json.loads((_models_dir(data_dir, variant) / "release.json").read_text())


def _release_scores(data_dir: Path, variant: str, rel: dict, X, ref, src, tgt) -> np.ndarray:
    md = _models_dir(data_dir, variant)
    s1 = np.mean([load_model(md / f"stage1_fold{k}.txt").predict(X) for k in range(2)], axis=0)
    if rel["stage"] == "stage1":
        return s1
    G = dn.group_features(ref, src, tgt, s1)
    return load_model(md / "stage2.txt").predict(np.concatenate([X, G], axis=1))


def cmd_evalx(data_dir: Path, cohort: str, variant: str, update_release: bool = False) -> None:
    """Frozen release scored in another dense env (e.g. 20%-scale Wm): catalog-scale robustness."""
    rel = _release(data_dir, variant)
    ctx = dn.build_context(data_dir, cohort, tuple(rel["features"]))
    s = _release_scores(data_dir, variant, rel, ctx.edges.X, ctx.edges.ref, ctx.edges.src, ctx.edges.tgt)
    sel = dn.select_policy(ctx, "D", s)
    out = {"variant": variant, "cohort": cohort,
           "frozen_policy_D": dn.evaluate(ctx, "D", s, rel["policy"]),
           "frozen_policy_K": dn.evaluate(ctx, "K", s, rel["policy"]),
           "reselected_best_D": sel["best"],
           "reselected_K": dn.evaluate(ctx, "K", s, sel["best"])}
    if update_release and sel["best"]["macro_f"] > out["frozen_policy_D"]["macro_f"]:
        # policy re-selection on D only (K reported, A2 untouched)
        rel["policy"] = sel["best"]
        rel["D"] = sel["best"]["macro_f"]
        (_models_dir(data_dir, variant) / "release.json").write_text(json.dumps(rel, indent=2))
        out["release_updated"] = True
    dn.report(data_dir, f"evalx_{cohort}_{variant}", out)


def cmd_audit(data_dir: Path, cohort: str, role: str, variant: str) -> None:
    """One-shot frozen-release audit (no tuning). A2 edges are never training rows."""
    rel = _release(data_dir, variant)
    ctx = dn.build_context(data_dir, cohort, tuple(rel["features"]))
    s = _release_scores(data_dir, variant, rel, ctx.edges.X, ctx.edges.ref, ctx.edges.src, ctx.edges.tgt)
    out = {"release": rel, "audit": dn.evaluate(ctx, role, s, rel["policy"])}
    dn.report(data_dir, f"audit_{cohort}_{role}_{variant}", out)


def cmd_test(data_dir: Path, output_dir: Path, variant: str | None, safe_policy: str | None,
             tau_override: dict | None) -> None:
    t0 = time.perf_counter()
    if variant is None:
        from ber.stages import platt_apply

        e = dn.load_edges(data_dir, "test", FEATURE_NAMES[:N_V4])
        bundle = json.loads((data_dir / "policy" / "release.json").read_text())
        s = platt_apply(load_model(data_dir / "models" / "lgbm_v1.txt").predict(e.X),
                        bundle["calibrator"]["a"], bundle["calibrator"]["b"])
        policy = json.loads(Path(safe_policy).read_text())
        rel = {"variant": "safe-v5-model", "policy": policy}
    else:
        rel = _release(data_dir, variant)
        e = dn.load_edges(data_dir, "test", tuple(rel["features"]))
        s = _release_scores(data_dir, variant, rel, e.X, e.ref, e.src, e.tgt)
        policy = rel["policy"]
    sel = dn.apply_policy(e.src, e.tgt, s, policy, e.ref)
    if tau_override:
        # per-country tau for countries without labels (guard), same policy kind
        for ci, c in enumerate(e.countries):
            if c in tau_override:
                m = e.country == ci
                sel[m] = dn.apply_policy(e.src, e.tgt, s, {**policy, "tau": tau_override[c]}, e.ref)[m]
    out = dn.export_test(data_dir, output_dir, e, sel)
    out["release"] = rel
    out["tau_override"] = tau_override
    for ci, c in enumerate(e.countries):
        m = e.country == ci
        out["per_country"][c]["mid_band_rate"] = float(((sel[m]) & (s[m] < 0.99)).mean())
    out["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    dn.report(data_dir, f"test_{Path(output_dir).name}", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[2] / "data"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--cohort", default="W")
    p = sub.add_parser("train")
    p.add_argument("--cohort", default="W")
    p.add_argument("--variant", required=True)
    p.add_argument("--drop", default=None, help="comma-separated feature names to exclude")
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--rounds2", type=int, default=400)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--no-stage2", action="store_true")
    p = sub.add_parser("evalx")
    p.add_argument("--cohort", default="Wm")
    p.add_argument("--variant", required=True)
    p.add_argument("--update-release", action="store_true")
    p = sub.add_parser("audit")
    p.add_argument("--cohort", default="W")
    p.add_argument("--role", default="A2")
    p.add_argument("--variant", required=True)
    p = sub.add_parser("test")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", default=None, help="omit to use the old model + --safe-policy")
    p.add_argument("--safe-policy", default=None)
    p.add_argument("--tau-override", default=None, help='JSON, e.g. {"France": 0.8}')
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    if a.cmd == "gate":
        cmd_gate(data_dir, a.cohort, data_dir / "models" / "lgbm_v1.txt", data_dir / "policy" / "release.json")
    elif a.cmd == "train":
        cmd_train(data_dir, a.cohort, a.variant, _feature_names(a.drop), a.rounds, a.threads,
                  not a.no_stage2, a.rounds2)
    elif a.cmd == "evalx":
        cmd_evalx(data_dir, a.cohort, a.variant, a.update_release)
    elif a.cmd == "audit":
        cmd_audit(data_dir, a.cohort, a.role, a.variant)
    elif a.cmd == "test":
        cmd_test(data_dir, Path(a.output_dir), a.variant, a.safe_policy,
                 json.loads(a.tau_override) if a.tau_override else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
