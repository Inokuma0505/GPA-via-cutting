from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.create_data.create_data import generate_dataset


def _latest_single_result_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("single_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _compute_Q_and_profit(
    p: np.ndarray,
    *,
    S,  # scipy.sparse matrix
    D,  # scipy.sparse matrix
    a: np.ndarray,
    c: np.ndarray,
    f: np.ndarray,
) -> Tuple[float, float]:
    # Q(p) = 1/2 p^T S p - f^T p
    # Z(p) = (p - c)^T (a - D p)
    # Use sparse matvecs to avoid densification
    Sp = S @ p
    Dp = D @ p
    Q = 0.5 * float(p @ Sp) - float(f @ p)
    Z = float((p - c) @ (a - Dp))
    return Q, Z


def _print_vec(name: str, v: np.ndarray, head: int, full: bool) -> None:
    print(f"[{name}] shape={v.shape}, min={np.min(v):.6g}, max={np.max(v):.6g}, mean={np.mean(v):.6g}")
    if full:
        np.set_printoptions(suppress=True, linewidth=160)
        print(v)
    else:
        h = min(head, v.size)
        np.set_printoptions(suppress=True, linewidth=160)
        print(f"  head({h}):", v[:h])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load saved GPA/Gurobi solutions from a results folder, compute Q and profit, and print them.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results/single_*/ directory. If omitted, the newest single_* under ./results is used.")
    parser.add_argument("--seed", type=int, default=None, help="Seed to regenerate the dataset (default: read from summary.csv if available, else 0).")
    parser.add_argument("--n", type=int, default=None, help="Dimension n (default: read from summary.csv if available, else required).")
    parser.add_argument("--delta", type=float, default=None, help="Delta value (default: read from summary.csv if available, else required).")
    parser.add_argument("--bounds", choices=["absolute", "relative"], default=None,
                        help="Bounds scheme for dataset regeneration (default: read from summary.csv if available, else 'absolute').")
    parser.add_argument("--nonneg", action="store_true", help="Apply non-negativity to bounds generation (default: off to match recall_one defaults).")
    parser.add_argument("--head", type=int, default=10, help="How many head elements of vectors to print when not using --full.")
    parser.add_argument("--full", action="store_true", help="Print full vectors (can be very large).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "results"

    # Resolve results directory
    if args.results_dir is None:
        target = _latest_single_result_dir(results_root)
        if target is None:
            raise SystemExit("No results/single_* directory found. Run recall_one.py with --outdir results first.")
        results_dir = target
    else:
        results_dir = Path(args.results_dir).resolve()
        if not results_dir.exists():
            raise SystemExit(f"results-dir not found: {results_dir}")

    solutions_dir = results_dir / "solutions"
    if not solutions_dir.exists():
        raise SystemExit(f"solutions directory not found: {solutions_dir}")

    # Try to read basic parameters from summary.csv
    summary_csv = results_dir / "summary.csv"
    seed = args.seed
    n = args.n
    delta = args.delta
    bounds = args.bounds
    if summary_csv.exists():
        try:
            import csv
            with summary_csv.open("r", newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                first = next(iter(reader))
                if seed is None and "seed" in first and first["seed"] != "":
                    seed = int(first["seed"])  # type: ignore[arg-type]
                if n is None and "n" in first and first["n"] != "":
                    n = int(first["n"])  # type: ignore[arg-type]
                if delta is None and "delta" in first and first["delta"] != "":
                    delta = float(first["delta"])  # type: ignore[arg-type]
                if bounds is None and "bounds" in first and first["bounds"] != "":
                    bounds = str(first["bounds"])  # type: ignore[arg-type]
        except Exception:
            pass

    # Apply defaults or fail if missing
    if seed is None:
        seed = 0
    if n is None or delta is None:
        raise SystemExit("n and delta must be provided (or present in summary.csv).")
    if bounds is None:
        bounds = "absolute"

    # Re-generate dataset consistently with recall_one.py settings
    ds = generate_dataset(
        n=n,
        delta_value=delta,
        seed=seed,
        bounds_scheme=bounds,
        nonnegativity=bool(args.nonneg),  # recall_one default is False; keep default unless specified
    )

    # Load saved solutions if present
    p_files = {
        "GPA": solutions_dir / "p_gpa.npy",
        "Gurobi": solutions_dir / "p_gurobi.npy",
    }

    any_loaded = False
    for name, fpath in p_files.items():
        if fpath.exists():
            p = np.load(fpath)
            Q, Z = _compute_Q_and_profit(p, S=ds.S, D=ds.D, a=ds.a, c=ds.c, f=ds.f)
            print("== {} ==".format(name))
            print(f"Q(p): {Q:.6f}, profit: {Z:.6f}")
            _print_vec("p", p, head=args.head, full=args.full)
            any_loaded = True
        else:
            print(f"[WARN] Solution file not found for {name}: {fpath}")

    if not any_loaded:
        raise SystemExit("No solution files found (expected p_gpa.npy and/or p_gurobi.npy).")


if __name__ == "__main__":
    main()
