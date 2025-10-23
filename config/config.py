from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).with_name("experiment_settings.json")


# ==== 合成データ（5.3節）再現実験 用パラメータ定義 ====
# そのまま import / 実行コードに渡せる形の設定オブジェクト＋辞書化ユーティリティ


@dataclass(frozen=True)
class Uniform:
    """Uniform(low, high) range used when sampling experiment data."""

    low: float
    high: float


@dataclass(frozen=True)
class DemandGenConfig:
    """Specification for generating demand matrix and linear term."""

    diag: Uniform = Uniform(1.0, 10.0)
    offdiag_max_per_row: int = 5
    offdiag_ratio_max: float = 0.2
    p0: Uniform = Uniform(1.0, 10.0)
    linear_u_abs: Uniform = Uniform(1.0, 10.0)


@dataclass(frozen=True)
class BoundsConfig:
    """Box constraints applied to prices; None represents the unbounded case."""

    l: Uniform
    u: Uniform
    name: str


@dataclass(frozen=True)
class GPAConfig:
    """Gradient Projection Algorithm parameters for the paper experiments."""

    num_initial_points: int = 5
    L_safety: float = 1.25
    rel_improve_tol: float = 1e-6
    grad_norm_tol: float = 1e-6
    max_iter: int = 10_000
    random_init_box_clip: bool = True


@dataclass(frozen=True)
class GurobiConfig:
    """Direct Gurobi solver configuration."""

    time_limits_sec: Tuple[int, int] = (3600, 14400)
    mip_gap: float = 0.0
    threads: int = 1
    use_bigM_formulation: bool = True


@dataclass(frozen=True)
class ExperimentDesign:
    """Design covering product counts, delta values, and replication strategy."""

    rng_seed: int = 42
    n_list: Tuple[int, ...] = (10_000, 25_000, 50_000, 75_000, 100_000)
    delta_list: Tuple[float, ...] = (1.0, 0.5)
    k_ratio: float = 0.10
    num_instances: int = 10


DEMAND_GEN = DemandGenConfig()
BOUNDS_SCENARIOS: List[Optional[BoundsConfig]] = [
    None,
    BoundsConfig(l=Uniform(1.0, 5.0), u=Uniform(5.0, 10.0), name="[1,5]-[5,10]"),
    BoundsConfig(l=Uniform(1.0, 5.0), u=Uniform(10.0, 15.0), name="[1,5]-[10,15]"),
    BoundsConfig(l=Uniform(1.0, 5.0), u=Uniform(15.0, 20.0), name="[1,5]-[15,20]"),
]
GPA_PARAMS = GPAConfig()
GUROBI_OPTS = GurobiConfig()
DESIGN = ExperimentDesign()


def make_param_grid(
    design: ExperimentDesign = DESIGN,
    demand: DemandGenConfig = DEMAND_GEN,
    bounds_list: List[Optional[BoundsConfig]] = BOUNDS_SCENARIOS,
    gpa: GPAConfig = GPA_PARAMS,
    grb: GurobiConfig = GUROBI_OPTS,
) -> List[Dict[str, Any]]:
    """
    Enumerate configuration dictionaries for each experiment instance.

    Each entry captures both data-generation parameters and solver options.
    """
    grid: List[Dict[str, Any]] = []
    for n in design.n_list:
        for delta in design.delta_list:
            k = int(design.k_ratio * n)
            for bounds in bounds_list:
                bounds_name_hash = 0
                if bounds is not None:
                    bounds_name_hash = abs(hash(bounds.name)) % 97
                for inst in range(design.num_instances):
                    seed = (
                        design.rng_seed
                        + 10_000 * inst
                        + 100 * n
                        + (1 if bounds is None else bounds_name_hash)
                    )
                    grid.append(
                        {
                            "n": n,
                            "delta": delta,
                            "k": k,
                            "instance_id": inst,
                            "rng_seed": seed,
                            "demand_gen": asdict(demand),
                            "bounds": None if bounds is None else asdict(bounds),
                            "gpa": asdict(gpa),
                            "gurobi": asdict(grb),
                            "metrics": {
                                "use_adjusted_objective_gap": True,
                            },
                        }
                    )
    return grid


# ---- 例：使い方（実行側の for ループに合流させるだけ）----
if __name__ == "__main__":
    param_grid = make_param_grid()
    # for cfg in param_grid:
    #     result = run_one_experiment(cfg)
    #     log_result(result)


@dataclass
class GPASettings:
    """Parameters for the GPA algorithm."""

    max_iter: int = 2000
    tol: float = 1e-8


@dataclass
class GurobiSettings:
    """Parameters for the optional direct Gurobi solve."""

    skip: bool = False
    only: bool = False
    time_limit: int = 60
    threads: Optional[int] = None
    big_m: Optional[float] = None


@dataclass
class OutputSettings:
    """Where and how to persist experiment outcomes."""

    outdir: str = "results"
    save_solutions: bool = False


@dataclass
class ExperimentConfig:
    """Top-level configuration for reproduction experiments."""

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_list: list[int] = field(default_factory=lambda: [50, 100, 200])
    delta_list: list[float] = field(default_factory=lambda: [1.0, 0.5])
    bounds: str = "absolute"
    nonneg: bool = False
    gpa: GPASettings = field(default_factory=GPASettings)
    gurobi: GurobiSettings = field(default_factory=GurobiSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExperimentConfig":
        """Build an ExperimentConfig from a mapping, applying defaults where needed."""
        if not data:
            return cls()

        data = dict(data)
        gpa_data = data.pop("gpa", None)
        gurobi_data = data.pop("gurobi", None)
        output_data = data.pop("output", None)

        config = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if isinstance(gpa_data, dict):
            config.gpa = GPASettings(**{k: v for k, v in gpa_data.items() if k in GPASettings.__dataclass_fields__})
        if isinstance(gurobi_data, dict):
            config.gurobi = GurobiSettings(**{
                k: v for k, v in gurobi_data.items() if k in GurobiSettings.__dataclass_fields__
            })
        if isinstance(output_data, dict):
            config.output = OutputSettings(**{
                k: v for k, v in output_data.items() if k in OutputSettings.__dataclass_fields__
            })
        return config


def load_config(path: Optional[Path | str] = None) -> ExperimentConfig:
    """Load experiment settings from JSON, falling back to defaults if nothing exists."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        if path:
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        return ExperimentConfig()

    with cfg_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, path: Optional[Path | str] = None, *, overwrite: bool = True) -> Path:
    """
    Persist experiment settings as JSON.

    Parameters
    ----------
    config:
        The ExperimentConfig to serialise.
    path:
        Optional target location. Defaults to config/experiment_settings.json.
    overwrite:
        If False and the file exists, a RuntimeError is raised.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.exists() and not overwrite:
        raise RuntimeError(f"Config file already exists and overwrite=False: {cfg_path}")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    with cfg_path.open("w", encoding="utf-8") as fp:
        json.dump(config.to_dict(), fp, indent=2, ensure_ascii=False)
    return cfg_path


def config_from_args(args: Any) -> ExperimentConfig:
    """
    Build an ExperimentConfig from an argparse Namespace.

    Only attributes relevant to the configuration are read; missing attributes
    will fall back to their defaults.
    """
    defaults = ExperimentConfig()

    def _coerce_iterable(value: Any, fallback: Iterable[Any]) -> list[Any]:
        if value is None:
            return list(fallback)
        if isinstance(value, (str, bytes)):
            return list(fallback)
        if isinstance(value, Iterable):
            return list(value)
        return list(fallback)

    seeds = _coerce_iterable(getattr(args, "seeds", None), defaults.seeds)
    n_list = _coerce_iterable(getattr(args, "n_list", None), defaults.n_list)
    delta_list = _coerce_iterable(getattr(args, "delta_list", None), defaults.delta_list)
    bounds = getattr(args, "bounds", None) or defaults.bounds
    nonneg = getattr(args, "nonneg", None)
    if nonneg is None:
        nonneg = defaults.nonneg
    else:
        nonneg = bool(nonneg)

    gpa_settings = GPASettings(
        max_iter=getattr(args, "gpa_max_iter", defaults.gpa.max_iter),
        tol=getattr(args, "gpa_tol", defaults.gpa.tol),
    )

    gurobi_settings = GurobiSettings(
        skip=getattr(args, "skip_gurobi", defaults.gurobi.skip),
        only=getattr(args, "only_gurobi", defaults.gurobi.only),
        time_limit=getattr(args, "gurobi_time", defaults.gurobi.time_limit),
        threads=getattr(args, "gurobi_threads", defaults.gurobi.threads),
        big_m=getattr(args, "gurobi_bigM", defaults.gurobi.big_m),
    )

    output_settings = OutputSettings(
        outdir=getattr(args, "outdir", defaults.output.outdir),
        save_solutions=getattr(args, "save_solutions", defaults.output.save_solutions),
    )

    return ExperimentConfig(
        seeds=seeds,
        n_list=n_list,
        delta_list=delta_list,
        bounds=bounds,
        nonneg=nonneg,
        gpa=gpa_settings,
        gurobi=gurobi_settings,
        output=output_settings,
    )
