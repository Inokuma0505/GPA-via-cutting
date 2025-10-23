from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from loguru import logger

from experiments_recall import HAS_GUROBI, run_gpa, run_gurobi
from src.create_data.create_data import generate_dataset

SUMMARY_FIELDS: Sequence[str] = (
    "seed",
    "n",
    "delta",
    "bounds",
    "algo",
    "time_sec",
    "iters",
    "Q",
    "profit",
    "gap",
    "feas_box",
    "feas_k",
    "num_changed",
)


def _configure_logging(log_path: Optional[Path]) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if log_path is not None:
        logger.add(log_path, level="INFO", enqueue=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPA and Gurobi (optional) once for a single random seed."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data generation.")
    parser.add_argument("--n", type=int, default=50, help="Number of products (problem dimension).")
    parser.add_argument("--delta", type=float, default=1.0, help="Price change budget per coordinate.")
    parser.add_argument(
        "--bounds",
        choices=["absolute", "relative"],
        default="absolute",
        help="Bounding scheme passed to the data generator.",
    )
    parser.add_argument(
        "--nonneg",
        action="store_true",
        help="Enforce non-negativity during data generation (default: disabled).",
    )
    parser.add_argument(
        "--skip-gpa",
        action="store_true",
        help="Disable the GPA run.",
    )
    parser.add_argument(
        "--skip-gurobi",
        action="store_true",
        help="Disable the Gurobi run.",
    )
    parser.add_argument("--gpa-max-iter", type=int, default=2000, help="Maximum GPA iterations.")
    parser.add_argument("--gpa-tol", type=float, default=1e-8, help="Stopping tolerance for GPA.")
    parser.add_argument("--gurobi-time", type=int, default=60, help="Time limit (seconds) for Gurobi.")
    parser.add_argument(
        "--gurobi-threads",
        type=int,
        default=None,
        help="Thread count for Gurobi (default: let the solver decide).",
    )
    parser.add_argument(
        "--gurobi-bigM",
        type=float,
        default=None,
        help="Override Big-M used in the direct Gurobi formulation.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional directory to save a summary (default: no files written).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional label appended to the output directory name.",
    )
    parser.add_argument(
        "--save-solutions",
        action="store_true",
        help="Persist decision vectors (requires --outdir).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for solvers where supported.",
    )
    return parser.parse_args()


def _prepare_output_dir(base: Optional[str], tag: Optional[str]) -> Optional[Path]:
    if base is None:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{tag}_{timestamp}" if tag else timestamp
    outdir = Path(base).expanduser() / f"single_{suffix}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_record(seed: int, n: int, delta: float, bounds: str, metrics: Dict[str, object]) -> Dict[str, object]:
    record = {
        "seed": seed,
        "n": n,
        "delta": delta,
        "bounds": bounds,
        **metrics,
    }
    return {key: _to_builtin(val) for key, val in record.items()}


def _print_record(record: Dict[str, object]) -> None:
    algo = record.get("algo", "unknown")
    lines: List[str] = []
    for field in SUMMARY_FIELDS:
        if field in record:
            lines.append(f"{field}: {record[field]}")
    extras = sorted(k for k in record if k not in SUMMARY_FIELDS)
    for key in extras:
        lines.append(f"{key}: {record[key]}")

    if lines:
        body = "\n".join(f"  {line}" for line in lines)
        logger.info("[{}]\n{}", algo, body)
    else:
        logger.info("[{}]", algo)


def _write_outputs(
    outdir: Path,
    records: List[Dict[str, object]],
    solutions: List[Tuple[str, np.ndarray]],
    save_solutions: bool,
) -> None:
    summary_jsonl = outdir / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec) + "\n")

    summary_csv = outdir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for rec in records:
            row = {field: rec.get(field, "") for field in SUMMARY_FIELDS}
            writer.writerow(row)

    if save_solutions and solutions:
        sol_dir = outdir / "solutions"
        sol_dir.mkdir(exist_ok=True)
        for algo, vec in solutions:
            fname = f"p_{algo.lower()}.npy"
            np.save(sol_dir / fname, vec)


def main() -> None:
    args = parse_args()
    if args.save_solutions and args.outdir is None:
        raise SystemExit("--save-solutions requires specifying --outdir.")

    outdir: Optional[Path] = None
    log_path: Optional[Path] = None
    if args.outdir is not None:
        outdir = _prepare_output_dir(args.outdir, args.tag)
        log_path = outdir / "run.log"

    _configure_logging(log_path)
    if outdir is not None:
        logger.info("Output directory prepared at {}", outdir)

    dataset = generate_dataset(
        n=args.n,
        delta_value=args.delta,
        seed=args.seed,
        bounds_scheme=args.bounds,
        nonnegativity=args.nonneg,
    )

    records: List[Dict[str, object]] = []
    solutions: List[Tuple[str, np.ndarray]] = []

    if not args.skip_gpa:
        p_gpa, met_gpa = run_gpa(
            dataset,
            max_iter=args.gpa_max_iter,
            tol=args.gpa_tol,
            L=getattr(dataset, "lipschitz_L", None),
            verbose=args.verbose,
        )
        rec = _make_record(args.seed, args.n, args.delta, args.bounds, met_gpa)
        _print_record(rec)
        records.append(rec)
        solutions.append(("GPA", p_gpa))

    if not args.skip_gurobi:
        if not HAS_GUROBI:
            logger.warning("Gurobi is not available; skipping the direct solve.")
        else:
            try:
                p_grb, met_grb = run_gurobi(
                    dataset,
                    time_limit=args.gurobi_time,
                    threads=args.gurobi_threads,
                    big_m=args.gurobi_bigM,
                    verbose=args.verbose,
                )
                rec = _make_record(args.seed, args.n, args.delta, args.bounds, met_grb)
                _print_record(rec)
                records.append(rec)
                solutions.append(("Gurobi", p_grb))
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Gurobi failed: {}", exc)

    if not records:
        logger.warning("No solver was executed. Enable GPA and/or Gurobi to obtain results.")
        return

    if outdir is not None:
        _write_outputs(outdir, records, solutions, args.save_solutions)
        logger.info("Results written to {}", outdir)


if __name__ == "__main__":
    main()
