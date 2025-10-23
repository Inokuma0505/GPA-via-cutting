from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments_recall import run_gpa
from src.create_data.create_data import generate_dataset

# Cutting-plane solvers (guard imports; some require gurobipy)
HAS_GUROBI = True
try:
    import gurobipy as gp  # noqa: F401
except Exception:
    HAS_GUROBI = False

try:
    from src.cut.cut_oa_kelley import solve_price_optimization_oa_from_dataset
except Exception:
    solve_price_optimization_oa_from_dataset = None  # type: ignore

try:
    from src.cut.cut_disjunctive import ModelData as DisjData, Options as DisjOpts, DisjunctiveSplitCutOptimizer
except Exception:
    DisjData = None  # type: ignore
    DisjOpts = None  # type: ignore
    DisjunctiveSplitCutOptimizer = None  # type: ignore

try:
    from src.cut.cut_Benders import PriceOptData as BendData, logic_benders_solve
except Exception:
    BendData = None  # type: ignore
    logic_benders_solve = None  # type: ignore

try:
    from src.cut.cut_lag import PriceOptData as LagData, LCPConfig, lagrangian_cutting_plane
except Exception:
    LagData = None  # type: ignore
    LCPConfig = None  # type: ignore
    lagrangian_cutting_plane = None  # type: ignore


def value_Q(p: np.ndarray, S, f: np.ndarray) -> float:
    # S may be scipy.sparse
    Sp = S @ p
    return 0.5 * float(p @ Sp) - float(f @ p)


def value_profit(p: np.ndarray, a: np.ndarray, D, c: np.ndarray) -> float:
    Dp = D @ p
    return float((p - c) @ (a - Dp))


def check_feas(p: np.ndarray, p0: np.ndarray, delta: np.ndarray, k: int, l: Optional[np.ndarray], u: Optional[np.ndarray]) -> Dict[str, Any]:
    feas_box = True
    if l is not None:
        feas_box &= bool(np.all(p >= l - 1e-8))
    if u is not None:
        feas_box &= bool(np.all(p <= u + 1e-8))
    changed = (p <= p0 - delta + 1e-8) | (p >= p0 + delta - 1e-8)
    num_changed = int(np.count_nonzero(changed))
    feas_k = bool(num_changed <= k)
    return {"feas_box": feas_box, "feas_k": feas_k, "num_changed": num_changed}


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_gpa_once(ds, gpa_max_iter: int, gpa_tol: float, verbose: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    p, met = run_gpa(ds, max_iter=gpa_max_iter, tol=gpa_tol, L=getattr(ds, "lipschitz_L", None), verbose=verbose)
    met = dict(met)
    met.setdefault("algo", "GPA")
    return p, met


def run_oa_kelley_once(ds, max_iters: int, time_limit: Optional[float], verbose: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not HAS_GUROBI or solve_price_optimization_oa_from_dataset is None:
        raise RuntimeError("OA/Kelley solver is unavailable (needs gurobipy).")
    out = solve_price_optimization_oa_from_dataset(
        ds,
        solver="gurobi",
        max_iters=max_iters,
        time_limit=time_limit,
        trust_region=float(np.max(ds.delta)),
        verbose=verbose,
    )
    p = np.asarray(out["p"], float)
    met = {
        "algo": "OA_Kelley",
        "Q": float(out.get("obj")),
        "profit": value_profit(p, ds.a, ds.D, ds.c),
        "LB": float(out.get("LB", np.nan)),
    }
    return p, met


def run_disjunctive_once(ds, base: str, use_theta_OA: bool, time_limit: Optional[float], verbose: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not HAS_GUROBI or DisjData is None:
        raise RuntimeError("Disjunctive solver is unavailable (needs gurobipy).")
    data = DisjData(S=ds.S.toarray(), b=ds.f, p0=ds.p0, delta=ds.delta, l=ds.l, u=ds.u, k=int(ds.k))
    opts = DisjOpts(
        base_formulation=base,
        use_theta_OA=use_theta_OA,
        time_limit=time_limit,
        verbose=verbose,
        trust_radius=float(np.max(ds.delta)),
    )
    solver = DisjunctiveSplitCutOptimizer(data, opts)
    solver.build_model()
    t0 = time.time()
    p, info = solver.solve()
    t1 = time.time()
    Q = value_Q(p, ds.S, ds.f)
    met = {
        "algo": f"Disjunctive_{base}{'_theta' if use_theta_OA else ''}",
        "time_sec": float(info.get("runtime", t1 - t0)),
        "Q": Q,
        "profit": value_profit(p, ds.a, ds.D, ds.c),
        "gap": float(info.get("gap", np.nan)) if info.get("gap", None) is not None else np.nan,
    }
    return p, met


def run_benders_once(ds, max_iters: int, tol: float, verbose: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not HAS_GUROBI or BendData is None or logic_benders_solve is None:
        raise RuntimeError("Benders solver is unavailable (needs gurobipy).")
    data = BendData(S=ds.S.toarray(), f=ds.f, p0=ds.p0, delta=ds.delta, L=ds.l, U=ds.u, k=int(ds.k))
    t0 = time.time()
    res = logic_benders_solve(data, max_iters=max_iters, tol=tol, log=verbose)
    t1 = time.time()
    patt: Optional[List[str]] = res.get("pattern") if isinstance(res.get("pattern"), list) else None
    # Reconstruct p by solving the convex subproblem with fixed regime thresholds t_hat
    if patt is None:
        # default: no changes
        p = ds.p0.copy()
    else:
        # t_hat per pattern
        t_hat = np.array([ds.p0[i] if s == '0' else (ds.p0[i] + ds.delta[i] if s == '+' else ds.p0[i] - ds.delta[i]) for i, s in enumerate(patt)], dtype=float)
        # Solve convex QP: min Q(p) s.t. p == t for '0'; p >= t for '+'; p <= t for '-'; and l<=p<=u
        import gurobipy as gp
        from gurobipy import GRB
        n = ds.p0.size
        sub = gp.Model("sub_reconstruct")
        if not verbose:
            sub.Params.OutputFlag = 0
        sub.Params.Method = 2; sub.Params.Crossover = 0
        pvars = sub.addVars(n, lb=ds.l.tolist(), ub=ds.u.tolist(), vtype=GRB.CONTINUOUS, name="p")
        obj = gp.QuadExpr()
        S = ds.S.toarray()
        for i in range(n):
            for j in range(n):
                sij = S[i, j]
                if sij != 0.0:
                    obj += 0.5 * sij * pvars[i] * pvars[j]
        for i in range(n):
            fi = float(ds.f[i])
            if fi != 0.0:
                obj += - fi * pvars[i]
        sub.setObjective(obj, GRB.MINIMIZE)
        for i, s in enumerate(patt):
            if s == '0':
                sub.addConstr(pvars[i] == float(t_hat[i]))
            elif s == '+':
                sub.addConstr(pvars[i] >= float(t_hat[i]))
            else:
                sub.addConstr(pvars[i] <= float(t_hat[i]))
        sub.optimize()
        if sub.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Reconstruct subproblem failed status={sub.Status}")
        p = np.array([pvars[i].X for i in range(n)], dtype=float)

    Q = value_Q(p, ds.S, ds.f)
    met = {
        "algo": "Benders",
        "time_sec": float(t1 - t0),
        "Q": Q,
        "profit": value_profit(p, ds.a, ds.D, ds.c),
        "LB": float(res.get("LB", np.nan)),
        "UB": float(res.get("UB", np.nan)),
        "gap": float(res.get("gap", np.nan)),
        "iterations": int(res.get("iterations", -1)),
    }
    return p, met


def run_lagr_once(ds, max_iters: int, tol_gap: float, sub_time: float, master_time: float, verbose: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not HAS_GUROBI or LagData is None or LCPConfig is None or lagrangian_cutting_plane is None:
        raise RuntimeError("Lagrangian CP solver is unavailable (needs gurobipy).")
    data = LagData(S=ds.S.toarray(), f=ds.f, p0=ds.p0, delta=ds.delta, k=int(ds.k), l=ds.l, u=ds.u)
    cfg = LCPConfig(max_iters=max_iters, tol_gap=tol_gap, lam0=0.0, beta_polyak=1.0, sub_time_limit=sub_time, master_time_limit=master_time, sub_exact_every=1)
    t0 = time.time()
    out = lagrangian_cutting_plane(data, cfg)
    t1 = time.time()
    p = np.asarray(out["best_p"], float)
    met = {
        "algo": "LagrangianCP",
        "time_sec": float(t1 - t0),
        "Q": float(out.get("best_obj", np.nan)),
        "profit": value_profit(p, ds.a, ds.D, ds.c),
        "LB": float(out.get("lower_bound", np.nan)),
        "gap": float(out.get("gap", np.nan)),
    }
    return p, met


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GPA and 4 cutting-plane methods once and compare.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--bounds", choices=["absolute", "relative"], default="absolute")
    ap.add_argument("--nonneg", action="store_true", help="Enforce nonnegativity in bounds generation.")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--save-solutions", action="store_true")

    # GPA
    ap.add_argument("--gpa-max-iter", type=int, default=2000)
    ap.add_argument("--gpa-tol", type=float, default=1e-8)

    # OA/Kelley
    ap.add_argument("--oa-iters", type=int, default=30)
    ap.add_argument("--oa-time", type=float, default=None, help="Per-master solve time limit (seconds)")

    # Disjunctive
    ap.add_argument("--disj-base", choices=["convex_hull", "indicators"], default="convex_hull")
    ap.add_argument("--disj-theta", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Use theta-OA objective instead of MIQP direct Q (default: True; use --no-disj-theta to turn off)")
    ap.add_argument("--disj-time", type=float, default=None)

    # Benders
    ap.add_argument("--benders-iters", type=int, default=100)
    ap.add_argument("--benders-tol", type=float, default=1e-5)

    # Lagrangian CP
    ap.add_argument("--lag-iters", type=int, default=30)
    ap.add_argument("--lag-gap", type=float, default=1e-4)
    ap.add_argument("--lag-sub-time", type=float, default=10.0)
    ap.add_argument("--lag-master-time", type=float, default=10.0)

    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Prepare dataset
    ds = generate_dataset(
        n=args.n,
        delta_value=args.delta,
        seed=args.seed,
        bounds_scheme=args.bounds,
        nonnegativity=bool(args.nonneg),
    )

    # Output directory
    tag = args.tag or ""
    name = f"exp_cut_one_{tag}_{_now_tag()}" if tag else f"exp_cut_one_{_now_tag()}"
    outdir = Path(args.outdir) / name
    (outdir / "solutions").mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "summary.csv"
    summary_jsonl = outdir / "summary.jsonl"

    headers = [
        "seed", "n", "delta", "bounds", "algo",
        "time_sec", "Q", "profit", "gap", "LB", "UB",
        "feas_box", "feas_k", "num_changed",
    ]

    rows: List[Dict[str, Any]] = []

    def record(algo: str, p: np.ndarray, met: Dict[str, Any], t_elapsed: Optional[float] = None) -> None:
        Qv = met.get("Q", value_Q(p, ds.S, ds.f))
        Zv = met.get("profit", value_profit(p, ds.a, ds.D, ds.c))
        feas = check_feas(p, ds.p0, ds.delta, int(ds.k), ds.l, ds.u)
        row = {
            "seed": args.seed,
            "n": args.n,
            "delta": args.delta,
            "bounds": args.bounds,
            "algo": algo,
            "time_sec": float(met.get("time_sec", t_elapsed if t_elapsed is not None else np.nan)),
            "Q": float(Qv),
            "profit": float(Zv),
            "gap": met.get("gap", ""),
            "LB": met.get("LB", ""),
            "UB": met.get("UB", ""),
            **feas,
        }
        rows.append(row)
        if args.save_solutions:
            np.save(outdir / "solutions" / f"p_{algo.lower()}.npy", p)

    # GPA
    t0 = time.time()
    p_gpa, met_gpa = run_gpa_once(ds, args.gpa_max_iter, args.gpa_tol, args.verbose)
    t1 = time.time()
    met_gpa.setdefault("time_sec", t1 - t0)
    record("GPA", p_gpa, met_gpa, t1 - t0)

    if not HAS_GUROBI:
        print("[WARN] gurobipy not available. Skipping cutting-plane solvers.")
    else:
        # OA/Kelley
        if solve_price_optimization_oa_from_dataset is not None:
            t0 = time.time();
            try:
                p_oa, met_oa = run_oa_kelley_once(ds, args.oa_iters, args.oa_time, args.verbose)
                t1 = time.time()
                met_oa.setdefault("time_sec", t1 - t0)
                record("OA_Kelley", p_oa, met_oa, t1 - t0)
            except Exception as e:
                print(f"[WARN] OA_Kelley failed: {e}")
        # Disjunctive
        if DisjData is not None:
            try:
                p_disj, met_disj = run_disjunctive_once(ds, args.disj_base, args.disj_theta, args.disj_time, args.verbose)
                record(met_disj["algo"], p_disj, met_disj)
            except Exception as e:
                print(f"[WARN] Disjunctive failed: {e}")
        # Benders
        if BendData is not None and logic_benders_solve is not None:
            try:
                p_ben, met_ben = run_benders_once(ds, args.benders_iters, args.benders_tol, args.verbose)
                record("Benders", p_ben, met_ben)
            except Exception as e:
                print(f"[WARN] Benders failed: {e}")
        # Lagrangian CP
        if LagData is not None and lagrangian_cutting_plane is not None:
            try:
                p_lag, met_lag = run_lagr_once(ds, args.lag_iters, args.lag_gap, args.lag_sub_time, args.lag_master_time, args.verbose)
                record("LagrangianCP", p_lag, met_lag)
            except Exception as e:
                print(f"[WARN] LagrangianCP failed: {e}")

    # Write outputs
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})
    with summary_jsonl.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r) + "\n")

    # Pretty print to console
    print("=== exp_cut_one summary ===")
    for r in rows:
        print(
            f"{r['algo']:<15} time={r['time_sec']:.3f}s  Q={r['Q']:.6g}  profit={r['profit']:.6g}  "
            f"feas(k,box)=({r['feas_k']},{r['feas_box']})  changed={r['num_changed']}"
        )
    print(f"Results -> {summary_csv}")


if __name__ == "__main__":
    main()
