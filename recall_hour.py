from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger

from experiments_recall import HAS_GUROBI, run_gpa, run_gurobi
from recall_one import SUMMARY_FIELDS, _configure_logging, _make_record, _print_record  # reuse helpers
from src.create_data.create_data import generate_dataset


def _parse_int_list(values: Iterable[int | str]) -> List[int]:
    return [int(v) for v in values]


def _parse_float_list(values: Iterable[float | str]) -> List[float]:
    return [float(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatedly run GPA (and optionally Gurobi) for up to the specified duration."
    )
    parser.add_argument("--duration-sec", type=float, default=3600.0, help="Total wall-clock time budget in seconds.")
    parser.add_argument("--seed-start", type=int, default=0, help="Initial RNG seed.")
    parser.add_argument(
        "--n-list",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="List of product counts to cycle through.",
    )
    parser.add_argument(
        "--delta-list",
        type=float,
        nargs="+",
        default=[1.0, 0.5],
        help="List of delta values to cycle through.",
    )
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
        help="Disable GPA runs.",
    )
    parser.add_argument(
        "--skip-gurobi",
        action="store_true",
        help="Disable Gurobi runs.",
    )
    parser.add_argument("--gpa-max-iter", type=int, default=2000, help="Maximum GPA iterations.")
    parser.add_argument("--gpa-tol", type=float, default=1e-8, help="Stopping tolerance for GPA.")
    parser.add_argument("--gurobi-time", type=int, default=60, help="Time limit (seconds) for each Gurobi solve.")
    parser.add_argument(
        "--gurobi-threads",
        type=int,
        default=None,
        help="Thread count for Gurobi (default: solver decides).",
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
        default="results",
        help="Base directory where outputs are written.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional label appended to the timestamped output directory name.",
    )
    parser.add_argument(
        "--save-solutions",
        action="store_true",
        help="Persist decision vectors (requires write access).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose solver logging where supported.",
    )
    return parser.parse_args()


def _prepare_output_dir(base: str, tag: Optional[str]) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"hour_{timestamp}" if tag is None else f"hour_{tag}_{timestamp}"
    outdir = Path(base).expanduser() / name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "solutions").mkdir(exist_ok=True)
    return outdir


def _open_outputs(outdir: Path):
    jsonl_path = outdir / "summary.jsonl"
    csv_path = outdir / "summary.csv"
    jsonl_fp = jsonl_path.open("w", encoding="utf-8")
    csv_fp = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_fp, fieldnames=SUMMARY_FIELDS)
    csv_writer.writeheader()
    return jsonl_fp, csv_fp, csv_writer


def _save_solution(outdir: Path, algo: str, seed: int, n: int, delta: float, iteration: int, vec: np.ndarray) -> None:
    fname = f"p_{algo.lower()}_seed{seed}_n{n}_d{delta}_iter{iteration}.npy"
    np.save(outdir / "solutions" / fname, vec)


def main() -> None:
    args = parse_args()
    if args.duration_sec <= 0:
        raise SystemExit("Duration must be positive.")
    if args.skip_gpa and args.skip_gurobi:
        raise SystemExit("At least one solver (GPA or Gurobi) must be enabled.")

    n_list = _parse_int_list(args.n_list)
    delta_list = _parse_float_list(args.delta_list)
    if not n_list or not delta_list:
        raise SystemExit("Provide at least one value for --n-list and --delta-list.")

    outdir = _prepare_output_dir(args.outdir, args.tag)
    _configure_logging(outdir / "run.log")
    logger.info("Output directory prepared at {}", outdir)
    jsonl_fp, csv_fp, csv_writer = _open_outputs(outdir)
    try:
        deadline = time.time() + args.duration_sec
        current_seed = args.seed_start
        iteration = 0
        warn_no_gurobi = False

        logger.info(
            "Running until {:.1f} seconds elapsed (deadline {}).",
            args.duration_sec,
            time.ctime(deadline),
        )

        while True:
            now = time.time()
            if now >= deadline:
                break

            for n in n_list:
                for delta in delta_list:
                    if time.time() >= deadline:
                        break

                    dataset = generate_dataset(
                        n=n,
                        delta_value=delta,
                        seed=current_seed,
                        bounds_scheme=args.bounds,
                        nonnegativity=args.nonneg,
                    )

                    if not args.skip_gpa:
                        p_gpa, met_gpa = run_gpa(
                            dataset,
                            max_iter=args.gpa_max_iter,
                            tol=args.gpa_tol,
                            L=getattr(dataset, "lipschitz_L", None),
                            verbose=args.verbose,
                        )
                        rec = _make_record(current_seed, n, delta, args.bounds, met_gpa)
                        _print_record(rec)
                        jsonl_fp.write(json.dumps(rec) + "\n"); jsonl_fp.flush()
                        csv_writer.writerow({field: rec.get(field, "") for field in SUMMARY_FIELDS}); csv_fp.flush()
                        if args.save_solutions:
                            _save_solution(outdir, "GPA", current_seed, n, delta, iteration, p_gpa)
                        iteration += 1

                    if not args.skip_gurobi:
                        if not HAS_GUROBI:
                            if not warn_no_gurobi:
                                logger.warning("Gurobi is not available; skipping direct solver runs.")
                                warn_no_gurobi = True
                        else:
                            try:
                                p_grb, met_grb = run_gurobi(
                                    dataset,
                                    time_limit=args.gurobi_time,
                                    threads=args.gurobi_threads,
                                    big_m=args.gurobi_bigM,
                                    verbose=args.verbose,
                                )
                                rec = _make_record(current_seed, n, delta, args.bounds, met_grb)
                                _print_record(rec)
                                jsonl_fp.write(json.dumps(rec) + "\n"); jsonl_fp.flush()
                                csv_writer.writerow({field: rec.get(field, "") for field in SUMMARY_FIELDS}); csv_fp.flush()
                                if args.save_solutions:
                                    _save_solution(outdir, "Gurobi", current_seed, n, delta, iteration, p_grb)
                                iteration += 1
                            except Exception as exc:  # pragma: no cover - defensive guard
                                err = {
                                    "seed": current_seed,
                                    "n": n,
                                    "delta": delta,
                                    "bounds": args.bounds,
                                    "algo": "Gurobi",
                                    "error": str(exc),
                                }
                                logger.warning("Gurobi failed: {}", exc)
                                jsonl_fp.write(json.dumps(err) + "\n"); jsonl_fp.flush()

                if time.time() >= deadline:
                    break

            current_seed += 1

        logger.info("Time budget exhausted. Outputs -> {}", outdir)

    finally:
        jsonl_fp.close()
        csv_fp.close()


if __name__ == "__main__":
    main()
