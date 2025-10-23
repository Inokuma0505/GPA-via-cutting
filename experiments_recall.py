# experiments.py
# 論文の再現実験：データ生成(5.1/5.3) → GPA → Gurobi 直接解法 → CSV/JSONL出力
# 依存: create_data.py, gpa.py, opt_gurobi.py（同一ディレクトリ or PYTHONPATH 上）

from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np

# --- 既存モジュールの取り込み ---
from config.config import (
    DEFAULT_CONFIG_PATH,
    ExperimentConfig,
    config_from_args,
    load_config,
    save_config,
)
from src.create_data.create_data import generate_dataset  # PriceOptDataset を返す
from src.GPA.gpa import gpa_optimize

# Gurobiは環境によって未導入の可能性があるので、インポートは安全に
try:
    from src.direct_gurobi.opt_gurobi import solve_price_optimization_gurobi
    HAS_GUROBI = True
except Exception as e:
    HAS_GUROBI = False


# ========== ユーティリティ ==========
def _to_dense(A):
    """scipy.sparse CSR 等が来たら dense に統一。"""
    try:
        import scipy.sparse as sp
        if sp.issparse(A):
            return A.toarray()
    except Exception:
        pass
    return np.asarray(A, dtype=float)


def value_Q(p: np.ndarray, S: np.ndarray, f: np.ndarray) -> float:
    """Q(p) = 1/2 p^T S p - f^T p"""
    return 0.5 * float(p.T @ (S @ p)) - float(f @ p)


def value_profit(p: np.ndarray, a: np.ndarray, D: np.ndarray, c: np.ndarray) -> float:
    """利潤 Z(p) = (p - c)^T (a - D p)"""
    return float((p - c) @ (a - D @ p))


def check_box_and_kchange(p: np.ndarray, p0: np.ndarray, delta: np.ndarray,
                          k: int, l: np.ndarray | None, u: np.ndarray | None,
                          tol: float = 1e-8) -> Dict[str, Any]:
    """ボックス制約と k-変更をざっくり検査（デバッグ用メトリクス）"""
    feas_box = True
    if l is not None:
        feas_box &= bool(np.all(p >= l - tol))
    if u is not None:
        feas_box &= bool(np.all(p <= u + tol))

    changed = (p <= p0 - delta + tol) | (p >= p0 + delta - tol)
    n_changed = int(np.count_nonzero(changed))
    feas_k = bool(n_changed <= k)

    return {"feas_box": feas_box, "feas_k": feas_k, "num_changed": n_changed}


# ========== ランナー：GPA ==========
def run_gpa(ds, max_iter=2000, tol=1e-8, L: float | None = None, verbose=False) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    D = _to_dense(ds.D)
    S = _to_dense(ds.S)
    p, info = gpa_optimize(
        a=ds.a, D=D, c=ds.c, p0=ds.p0, delta=ds.delta, k=ds.k,
        l=ds.l, u=ds.u, L=(L or getattr(ds, "lipschitz_L", None)),
        max_iter=max_iter, tol=tol, verbose=verbose,
    )
    t1 = time.time()

    Q = float(info.get("Q", value_Q(p, S, ds.f)))
    Z = float(info.get("Z", value_profit(p, ds.a, D, ds.c)))
    iters = int(info.get("n_iter", -1))
    chk = check_box_and_kchange(p, ds.p0, ds.delta, ds.k, ds.l, ds.u)

    met = {
        "algo": "GPA",
        "time_sec": float(info.get("time_sec", t1 - t0)),
        "iters": iters,
        "Q": Q,
        "profit": Z,
        "gap": np.nan,  # MIPGAPはなし
        **chk,
    }
    return p, met


# ========== ランナー：Gurobi 直接解法 ==========
def run_gurobi(ds, time_limit=60, threads=None, verbose=False, big_m=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not HAS_GUROBI:
        raise RuntimeError("Gurobi (gurobipy) が利用できません。opt_gurobi.py の読み込みに失敗しました。")

    t0 = time.time()
    D = _to_dense(ds.D)
    S = _to_dense(ds.S)

    kwargs = dict(
        p0=ds.p0, delta=ds.delta, k=ds.k, c=ds.c,
        a=ds.a, D=D, l=ds.l, u=ds.u,
        time_limit=time_limit, threads=threads, verbose=verbose
    )
    if big_m is not None:
        kwargs["big_m"] = float(big_m)

    res = solve_price_optimization_gurobi(**kwargs)
    t1 = time.time()

    p = res["p"]
    Q = value_Q(p, S, ds.f)
    Z = value_profit(p, ds.a, D, ds.c)
    chk = check_box_and_kchange(p, ds.p0, ds.delta, ds.k, ds.l, ds.u)

    met = {
        "algo": "Gurobi",
        "time_sec": float(res.get("time_sec", res.get("runtime", t1 - t0))),
        "iters": int(res.get("nodecount", -1)),   # 便宜上ノード数など
        "Q": Q,
        "profit": Z,
        "gap": float(res.get("mipgap", np.nan)),
        **chk,
    }
    return p, met


SAVE_CONFIG_SENTINEL = "__DEFAULT_EXPERIMENT_CONFIG__"


def _build_parser(defaults: ExperimentConfig, *, parents: list[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="論文の再現実験ランナー", parents=parents)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=defaults.seeds,
    )
    parser.add_argument(
        "--n_list",
        type=int,
        nargs="+",
        default=defaults.n_list,
    )
    parser.add_argument(
        "--delta_list",
        type=float,
        nargs="+",
        default=defaults.delta_list,
    )
    parser.add_argument(
        "--bounds",
        choices=["absolute", "relative"],
        default=defaults.bounds,
    )
    parser.add_argument(
        "--nonneg",
        action=argparse.BooleanOptionalAction,
        default=defaults.nonneg,
        help="価格の非負制約を課す（create_data側の引数）",
    )

    # GPA
    parser.add_argument("--gpa_max_iter", type=int, default=defaults.gpa.max_iter)
    parser.add_argument("--gpa_tol", type=float, default=defaults.gpa.tol)

    # Gurobi
    parser.add_argument(
        "--skip_gurobi",
        action=argparse.BooleanOptionalAction,
        default=defaults.gurobi.skip,
    )
    parser.add_argument(
        "--only_gurobi",
        action=argparse.BooleanOptionalAction,
        default=defaults.gurobi.only,
    )
    parser.add_argument("--gurobi_time", type=int, default=defaults.gurobi.time_limit)
    parser.add_argument("--gurobi_threads", type=int, default=defaults.gurobi.threads)
    parser.add_argument(
        "--gurobi_bigM",
        type=float,
        default=defaults.gurobi.big_m,
        help="境界なし版を強制したい場合のBig-M。通常は不要。",
    )

    # 出力
    parser.add_argument("--outdir", type=str, default=defaults.output.outdir)
    parser.add_argument(
        "--save_solutions",
        action=argparse.BooleanOptionalAction,
        default=defaults.output.save_solutions,
    )
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="実験設定を記述したJSONファイル。未指定の場合は config/experiment_settings.json を探索します。",
    )
    base_parser.add_argument(
        "--save-config",
        type=str,
        nargs="?",
        const=SAVE_CONFIG_SENTINEL,
        help="実行時の設定をJSONで保存します。パス未指定の場合は config/experiment_settings.json に保存します。",
    )

    config_ns, remaining = base_parser.parse_known_args()
    try:
        defaults = load_config(config_ns.config)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    parser = _build_parser(defaults, parents=[base_parser])
    args = parser.parse_args(remaining, namespace=config_ns)
    return args


def main():
    args = parse_args()
    if args.save_config is not None:
        target_path = DEFAULT_CONFIG_PATH if args.save_config == SAVE_CONFIG_SENTINEL else Path(args.save_config)
        saved_cfg = config_from_args(args)
        save_path = save_config(saved_cfg, target_path)
        print(f"[INFO] 設定を保存しました -> {save_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"exp_{ts}"
    (outdir / "solutions").mkdir(parents=True, exist_ok=True)

    # 出力ファイル
    csv_path = outdir / "summary.csv"
    jsonl_path = outdir / "summary.jsonl"

    # CSV ヘッダ
    headers = [
        "seed", "n", "delta", "bounds",
        "algo", "time_sec", "iters", "Q", "profit", "gap",
        "feas_box", "feas_k", "num_changed",
    ]

    with open(csv_path, "w", newline="") as f_csv, open(jsonl_path, "w") as f_js:
        writer = csv.DictWriter(f_csv, fieldnames=headers)
        writer.writeheader()

        for seed in args.seeds:
            for n in args.n_list:
                for delta in args.delta_list:

                    # --- データ生成（5.1/5.3 準拠） ---
                    ds = generate_dataset(
                        n=n, delta_value=delta, seed=seed,
                        bounds_scheme=args.bounds, nonnegativity=args.nonneg
                    )

                    # --- GPA ---
                    if not args.only_gurobi:
                        p_gpa, met_gpa = run_gpa(
                            ds, max_iter=args.gpa_max_iter, tol=args.gpa_tol,
                            L=getattr(ds, "lipschitz_L", None)
                        )
                        rec = dict(seed=seed, n=n, delta=delta, bounds=args.bounds, **met_gpa)
                        writer.writerow(rec); f_csv.flush()
                        f_js.write(json.dumps(rec) + "\n"); f_js.flush()
                        if args.save_solutions:
                            np.save(outdir / "solutions" / f"p_gpa_n{n}_d{delta}_s{seed}.npy", p_gpa)

                    # --- Gurobi（任意） ---
                    if not args.skip_gurobi:
                        try:
                            p_grb, met_grb = run_gurobi(
                                ds, time_limit=args.gurobi_time,
                                threads=args.gurobi_threads, big_m=args.gurobi_bigM
                            )
                            rec = dict(seed=seed, n=n, delta=delta, bounds=args.bounds, **met_grb)
                            writer.writerow(rec); f_csv.flush()
                            f_js.write(json.dumps(rec) + "\n"); f_js.flush()
                            if args.save_solutions:
                                np.save(outdir / "solutions" / f"p_grb_n{n}_d{delta}_s{seed}.npy", p_grb)
                        except Exception as e:
                            # Gurobi未導入などは JSONL にだけ記録
                            err = dict(
                                seed=seed, n=n, delta=delta, bounds=args.bounds,
                                algo="Gurobi", error=str(e)
                            )
                            f_js.write(json.dumps(err) + "\n"); f_js.flush()
                            print(f"[WARN] Gurobi skipped: {err}")

    print(f"[DONE] Results -> {csv_path}\n        Logs    -> {jsonl_path}\n        Dir     -> {outdir}")
    if not HAS_GUROBI and not args.skip_gurobi and not args.only_gurobi:
        print("[NOTE] opt_gurobi/gurobipy が読み込めなかったため、Gurobi 実験はスキップされました。")
        

if __name__ == "__main__":
    main()
