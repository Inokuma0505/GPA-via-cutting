#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切除平面 (Outer Approximation / Kelley) アルゴリズム
Price Optimization 問題 (本リポジトリの `opt_gurobi.solve_price_optimization_gurobi` と同じ二次形式)

目的（統一記法）:
	Q(p) = 1/2 p^T S p - f^T p       (凸二次, 最小化)
最大化したい期待利益（論文側の利得）:
	Z(p) = -Q(p) = f^T p - 1/2 p^T S p

本ファイルでは Q の凸最小化を OA (Kelley) 型の cutting-plane で解き、
結果として { 最良可行点 p*, その Q, Z (= -Q), 下界 LB (= theta の最大値), 反復履歴 } を返す。

特徴 / 実装上の注意:
 1. データセット `create_data.generate_dataset` で得られる `PriceOptDataset` と互換。
 2. 初期カットはデータセットで提供される 5 本の初期解 (init_1_... 〜 init_5_...) を利用（存在すれば）。
 3. ソルバ backend: 現状リポジトリ依存の関係で Gurobi のみ必須。他 (CPLEX, MOSEK) は Optional。
 4. Gurobi では indicator constraints を用いて (stay / up / down) の 3 値ロジックを直接モデル化。
 5. カーディナリティ制約: Σ (zR_i + zL_i) ≤ k。
 6. 各反復で得られる解 p^t に対し接線カット θ ≥ Q(p^t) + ∇Q(p^t)^T (p - p^t) を追加。
 7. 停止条件: |UB - LB| ≤ abs_tol または 相対ギャップ ≤ rel_tol。

返却結果 dict 形式:
  {
	 'p': p*,               # 最良（UB を与える）可行点
	 'zP','zR','zL': np.ndarray(int)  # 3 区分の最終値
	 'obj': UB (= Q(p*)),
	 'revenue': Z(p*),      # 期待利益（最大化値）
	 'LB': LB,              # OA 下界（θ の最大値）
	 'history': list[ {iter, LB, UB, gap_abs, gap_rel, runtime, ...} ]
  }

CLI 例:
  python -m src.cut.cut_oa_kelley --n 100 --solver gurobi --max-iters 30

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

# ==== Optional backends =====================================================
try:  # Gurobi
	import gurobipy as gp
	from gurobipy import GRB
except Exception:  # pragma: no cover
	gp = None  # type: ignore
	GRB = None  # type: ignore

try:  # CPLEX (optional)
	from docplex.mp.model import Model as CpxModel  # type: ignore
except Exception:  # pragma: no cover
	CpxModel = None  # type: ignore

try:  # MOSEK (optional)
	import mosek.fusion as mf  # type: ignore
except Exception:  # pragma: no cover
	mf = None  # type: ignore

# ==== Dataset import (robust to path) =======================================
try:
	# 推奨（src を PYTHONPATH に追加してある場合）
	from create_data.create_data import generate_dataset, PriceOptDataset  # type: ignore
except Exception:  # pragma: no cover
	# Fallback: 相対パスを追加
	import os, sys
	_THIS_DIR = os.path.dirname(__file__)
	_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "create_data"))
	if _SRC_DIR not in sys.path:
		sys.path.append(os.path.dirname(_SRC_DIR))
	try:  # 再挑戦
		from create_data.create_data import generate_dataset, PriceOptDataset  # type: ignore
	except Exception:
		generate_dataset = None  # type: ignore
		PriceOptDataset = None  # type: ignore

__all__ = [
	"OAConfig",
	"OAResult",
	"solve_price_optimization_oa",
	"solve_price_optimization_oa_from_dataset",
]


# =============================================================================
# Utility / Math helpers
# =============================================================================
@dataclass
class OAConfig:
	max_iters: int = 50
	abs_tol: float = 1e-6
	rel_tol: float = 1e-5
	time_limit: Optional[float] = None  # seconds (per master solve)
	trust_region: Optional[float] = None  # L∞ radius (not always used)
	verbose: bool = True


@dataclass
class OAResult:
	p: np.ndarray
	zP: np.ndarray
	zR: np.ndarray
	zL: np.ndarray
	obj: float  # UB in terms of Q(p)
	LB: float
	history: List[Dict[str, Any]]

	@property
	def revenue(self) -> float:
		return -float(self.obj)


def _grad_value(S: sp.spmatrix | np.ndarray, f: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, float]:
	"""Return gradient and value of Q(p)=0.5 p^T S p - f^T p."""
	if sp.issparse(S):
		Sp = S @ p
		val = 0.5 * float(p @ Sp) - float(f @ p)
	else:
		Sp = S.dot(p)
		val = 0.5 * float(p @ Sp) - float(f @ p)
	grad = Sp - f
	return grad, val


# =============================================================================
# Backend: Gurobi OA (indicator constraints)
# =============================================================================
class _GurobiOA:
	def __init__(
		self,
		S: sp.spmatrix | np.ndarray,
		f: np.ndarray,
		p0: np.ndarray,
		delta: np.ndarray,
		l: np.ndarray,
		u: np.ndarray,
		k: int,
		cfg: OAConfig,
	) -> None:
		if gp is None:
			raise ImportError("gurobipy が利用できません。pyproject.toml の依存関係を確認してください。")
		self.S = S
		self.f = f
		self.p0 = p0
		self.delta = delta
		self.l = l
		self.u = u
		self.k = int(k)
		self.cfg = cfg
		self.n = int(p0.size)
		self._build_master()

	# ------------------------------------------------------------------
	def _build_master(self) -> None:
		n = self.n
		m = gp.Model("oa_indicator_gurobi")
		if not self.cfg.verbose:
			m.Params.OutputFlag = 0
		if self.cfg.time_limit is not None:
			m.Params.TimeLimit = float(self.cfg.time_limit)

		p = m.addVars(n, vtype=GRB.CONTINUOUS, lb=self.l.tolist(), ub=self.u.tolist(), name="p")
		theta = m.addVar(vtype=GRB.CONTINUOUS, name="theta")
		zP = m.addVars(n, vtype=GRB.BINARY, name="zP")  # stay at p0
		zR = m.addVars(n, vtype=GRB.BINARY, name="zR")  # >= p0+δ
		zL = m.addVars(n, vtype=GRB.BINARY, name="zL")  # <= p0-δ
		y = m.addVars(n, vtype=GRB.BINARY, name="y")

		m.setObjective(theta, GRB.MINIMIZE)

		for i in range(n):
			m.addConstr(zP[i] + zR[i] + zL[i] == 1, name=f"onehot[{i}]")
			m.addConstr(y[i] == zR[i] + zL[i], name=f"ydef[{i}]")
		m.addConstr(gp.quicksum(y[i] for i in range(n)) <= self.k, name="card")

		# indicator constraints linking p
		for i in range(n):
			m.addGenConstrIndicator(zP[i], True, p[i] == float(self.p0[i]), name=f"ind_eq[{i}]")
			m.addGenConstrIndicator(zR[i], True, p[i] >= float(self.p0[i] + self.delta[i]), name=f"ind_up[{i}]")
			m.addGenConstrIndicator(zL[i], True, p[i] <= float(self.p0[i] - self.delta[i]), name=f"ind_dn[{i}]")

		self.m = m
		self.p = p
		self.theta = theta
		self.zP, self.zR, self.zL, self.y = zP, zR, zL, y
		self.cut_count = 0
		self.trust_pair: Optional[Tuple[List[Any], List[Any]]] = None

	# ------------------------------------------------------------------
	def set_trust_region(self, center: np.ndarray, radius: float) -> None:
		"""Optional L∞ trust region stabilization."""
		if self.trust_pair is not None:
			for ct in self.trust_pair[0] + self.trust_pair[1]:
				self.m.remove(ct)
		plus: List[Any] = []
		minus: List[Any] = []
		for i in range(self.n):
			plus.append(self.m.addConstr(self.p[i] - float(center[i]) <= radius, name=f"tr+[{i}]") )
			minus.append(self.m.addConstr(float(center[i]) - self.p[i] <= radius, name=f"tr-[{i}]") )
		self.trust_pair = (plus, minus)
		self.m.update()

	# ------------------------------------------------------------------
	def add_tangent_cut(self, pt: np.ndarray) -> None:
		grad, val = _grad_value(self.S, self.f, pt)
		# tangent: θ ≥ Q(pt) + grad^T (p - pt)  -> θ - grad^T p ≥ val - grad^T pt
		rhs = float(val - grad @ pt)
		expr = self.theta - gp.quicksum(float(grad[i]) * self.p[i] for i in range(self.n))
		self.m.addConstr(expr >= rhs, name=f"oa[{self.cut_count}]")
		self.cut_count += 1
		self.m.update()

	# ------------------------------------------------------------------
	def solve_master(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
		self.m.optimize()
		status = int(self.m.status)
		if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
			raise RuntimeError(f"Gurobi status={status}")
		n = self.n
		p_val = np.array([self.p[i].X for i in range(n)], dtype=float)
		theta_val = float(self.theta.X)
		info: Dict[str, Any] = {
			"status": status,
			"runtime": float(self.m.Runtime),
			"gap": float(self.m.MIPGap) if hasattr(self.m, "MIPGap") else None,
			"n_cuts": self.cut_count,
		}
		return p_val, theta_val, info

	# ------------------------------------------------------------------
	def pull_z(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		n = self.n
		zP = np.array([int(round(self.zP[i].X)) for i in range(n)], dtype=int)
		zR = np.array([int(round(self.zR[i].X)) for i in range(n)], dtype=int)
		zL = np.array([int(round(self.zL[i].X)) for i in range(n)], dtype=int)
		return zP, zR, zL


# =============================================================================
# (Optional) CPLEX backend (kept for completeness; not required for repo)
# =============================================================================
class _CplexOA:  # pragma: no cover - optional
	def __init__(self, S, f, p0, delta, l, u, k, cfg: OAConfig):
		if CpxModel is None:
			raise ImportError("docplex (CPLEX) が利用できません。")
		self.S, self.f, self.p0, self.delta, self.l, self.u, self.k = S, f, p0, delta, l, u, int(k)
		self.cfg = cfg
		self.n = int(p0.size)
		self._build_master()

	def _build_master(self) -> None:
		n = self.n
		m = CpxModel(name="oa_indicator_cplex")
		if not self.cfg.verbose:
			m.set_log_output(False)
		if self.cfg.time_limit is not None:
			m.parameters.timelimit = self.cfg.time_limit

		p = m.continuous_var_list(n, lb=self.l.tolist(), ub=self.u.tolist(), name="p")
		theta = m.continuous_var(name="theta")
		zP = m.binary_var_list(n, name="zP")
		zR = m.binary_var_list(n, name="zR")
		zL = m.binary_var_list(n, name="zL")
		y = m.binary_var_list(n, name="y")

		m.minimize(theta)
		for i in range(n):
			m.add_constraint(zP[i] + zR[i] + zL[i] == 1, ctname=f"onehot[{i}]")
			m.add_constraint(y[i] == zR[i] + zL[i], ctname=f"ydef[{i}]")
		m.add_constraint(m.sum(y) <= self.k, ctname="card")
		for i in range(n):
			m.add_indicator(zP[i], p[i] == float(self.p0[i]), name=f"ind_eq[{i}]")
			m.add_indicator(zR[i], p[i] >= float(self.p0[i] + self.delta[i]), name=f"ind_up[{i}]")
			m.add_indicator(zL[i], p[i] <= float(self.p0[i] - self.delta[i]), name=f"ind_dn[{i}]")
		self.m, self.p, self.theta = m, p, theta
		self.zP, self.zR, self.zL, self.y = zP, zR, zL, y
		self.cut_count = 0
		self.trust_pair = None

	def set_trust_region(self, center: np.ndarray, radius: float) -> None:
		if self.trust_pair is not None:
			for ct in self.trust_pair[0] + self.trust_pair[1]:
				self.m.remove(ct)
		plus = []; minus = []
		for i in range(self.n):
			plus.append(self.m.add_constraint(self.p[i] - float(center[i]) <= radius))
			minus.append(self.m.add_constraint(float(center[i]) - self.p[i] <= radius))
		self.trust_pair = (plus, minus)

	def add_tangent_cut(self, pt: np.ndarray) -> None:
		grad, val = _grad_value(self.S, self.f, pt)
		rhs = float(val - grad @ pt)
		expr = self.theta - self.m.sum(float(grad[i]) * self.p[i] for i in range(self.n))
		self.m.add_constraint(expr >= rhs, ctname=f"oa[{self.cut_count}]")
		self.cut_count += 1

	def solve_master(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
		sol = self.m.solve(log_output=self.cfg.verbose)
		if sol is None:
			raise RuntimeError("CPLEX returned no solution.")
		p_val = np.array([sol.get_value(v) for v in self.p])
		theta_val = float(sol.get_value(self.theta))
		info: Dict[str, Any] = {
			"status": int(self.m.get_solve_status().value),
			"gap": float(self.m.solve_details.mip_relative_gap) if self.m.solve_details else None,
			"runtime": float(self.m.solve_details.time) if self.m.solve_details else None,
			"n_cuts": self.cut_count,
		}
		return p_val, theta_val, info

	def pull_z(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		sol = self.m.solution
		zP = np.array([int(round(sol.get_value(v))) for v in self.zP], dtype=int)
		zR = np.array([int(round(sol.get_value(v))) for v in self.zR], dtype=int)
		zL = np.array([int(round(sol.get_value(v))) for v in self.zL], dtype=int)
		return zP, zR, zL


# =============================================================================
# (Optional) MOSEK backend (Big-M) - left mainly for parity
# =============================================================================
class _MosekOA:  # pragma: no cover - optional
	def __init__(self, S, f, p0, delta, l, u, k, cfg: OAConfig):
		if mf is None:
			raise ImportError("mosek.fusion が利用できません。")
		self.S, self.f, self.p0, self.delta, self.l, self.u, self.k = S, f, p0, delta, l, u, int(k)
		self.cfg = cfg
		self.n = int(p0.size)
		self._build_master()

	def _build_master(self) -> None:
		n = self.n
		M_up = (self.u - (self.p0 + self.delta)).astype(float)
		M_dn = ((self.p0 - self.delta) - self.l).astype(float)
		M_eq_up = (self.u - self.p0).astype(float)
		M_eq_dn = (self.p0 - self.l).astype(float)
		M = mf.Model("oa_bigM_mosek")
		if not self.cfg.verbose:
			M.setLogHandler(None)
		p = M.variable("p", n, mf.Domain.inRange(self.l.tolist(), self.u.tolist()))
		theta = M.variable("theta", 1, mf.Domain.unbounded())
		zP = M.variable("zP", n, mf.Domain.binary())
		zR = M.variable("zR", n, mf.Domain.binary())
		zL = M.variable("zL", n, mf.Domain.binary())
		y = M.variable("y", n, mf.Domain.binary())
		M.objective(mf.ObjectiveSense.Minimize, theta)
		M.constraint(mf.Expr.add(mf.Expr.add(zP, zR), zL), mf.Domain.equalsTo(1.0))
		M.constraint(mf.Expr.sub(y, mf.Expr.add(zR, zL)), mf.Domain.equalsTo(0.0))
		M.constraint(mf.Expr.sum(y), mf.Domain.lessThan(float(self.k)))
		for i in range(n):
			# zP=1 ⇒ p == p0
			M.constraint(mf.Expr.sub(p.index(i), float(self.p0[i])), mf.Domain.lessThan(M_eq_up[i] * (1.0 - zP.index(i))))
			M.constraint(mf.Expr.sub(float(self.p0[i]), p.index(i)), mf.Domain.lessThan(M_eq_dn[i] * (1.0 - zP.index(i))))
			# zR=1 ⇒ p ≥ p0+δ
			M.constraint(mf.Expr.sub(p.index(i), float(self.p0[i] + self.delta[i])), mf.Domain.greaterThan(-M_up[i] * (1.0 - zR.index(i))))
			# zL=1 ⇒ p ≤ p0-δ
			M.constraint(mf.Expr.sub(p.index(i), float(self.p0[i] - self.delta[i])), mf.Domain.lessThan(M_dn[i] * (1.0 - zL.index(i))))
		self.M, self.p, self.theta = M, p, theta
		self.zP, self.zR, self.zL, self.y = zP, zR, zL, y
		self.cut_count = 0

	def add_tangent_cut(self, pt: np.ndarray) -> None:
		grad, val = _grad_value(self.S, self.f, pt)
		rhs = float(val - grad @ pt)
		self.M.constraint(mf.Expr.sub(self.theta, mf.Expr.dot(grad.astype(float), self.p)), mf.Domain.greaterThan(rhs))
		self.cut_count += 1

	def solve_master(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
		if self.cfg.time_limit is not None:
			self.M.setSolverParam("optimizerMaxTime", float(self.cfg.time_limit))
		self.M.solve()
		p_val = np.array(self.p.level()).reshape(-1)
		theta_val = float(self.theta.level()[0])
		info: Dict[str, Any] = {"status": 0, "gap": None, "runtime": None, "n_cuts": self.cut_count}
		return p_val, theta_val, info

	def pull_z(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		zP = np.array(self.zP.level()).round().astype(int)
		zR = np.array(self.zR.level()).round().astype(int)
		zL = np.array(self.zL.level()).round().astype(int)
		return zP, zR, zL


# =============================================================================
# Driver (common iteration logic)
# =============================================================================
class OADriver:
	def __init__(
		self,
		S: sp.spmatrix | np.ndarray,
		f: np.ndarray,
		p0: np.ndarray,
		delta: np.ndarray,
		l: np.ndarray,
		u: np.ndarray,
		k: int,
		cfg: OAConfig,
		solver: str = "gurobi",
	) -> None:
		solver_l = solver.lower()
		if solver_l == "gurobi":
			self.backend = _GurobiOA(S, f, p0, delta, l, u, k, cfg)
		elif solver_l == "cplex":  # pragma: no cover
			self.backend = _CplexOA(S, f, p0, delta, l, u, k, cfg)
		elif solver_l == "mosek":  # pragma: no cover
			self.backend = _MosekOA(S, f, p0, delta, l, u, k, cfg)
		else:
			raise ValueError("solver must be one of {'gurobi','cplex','mosek'}")
		self.S, self.f = S, f
		self.cfg = cfg

	# ------------------------------------------------------------------
	def run(self, init_points: Iterable[np.ndarray]) -> OAResult:
		UB = float("inf")
		LB = -float("inf")
		history: List[Dict[str, Any]] = []

		init_list = [np.asarray(p, dtype=float) for p in init_points]
		if len(init_list) == 0:
			raise ValueError("At least one initial point is required for OA.")
		# seed cuts
		for pt in init_list:
			self.backend.add_tangent_cut(pt)
		# optional trust region (center = last init)
		if self.cfg.trust_region is not None and hasattr(self.backend, "set_trust_region"):
			try:
				self.backend.set_trust_region(init_list[-1], float(self.cfg.trust_region))  # type: ignore[arg-type]
			except Exception:
				pass

		p_best = init_list[0]
		Q_best = float("inf")

		for it in range(1, self.cfg.max_iters + 1):
			p_t, theta_t, info = self.backend.solve_master()
			grad_t, Q_t = _grad_value(self.S, self.f, p_t)
			LB = max(LB, theta_t)
			if Q_t < Q_best:
				Q_best = Q_t
				p_best = p_t.copy()
			UB = min(Q_best, UB)

			gap_abs = UB - LB
			gap_rel = gap_abs / max(1.0, abs(UB))
			history.append({
				"iter": it,
				"LB": LB,
				"UB": UB,
				"gap_abs": gap_abs,
				"gap_rel": gap_rel,
				**info,
			})

			if (gap_abs <= self.cfg.abs_tol) or (gap_rel <= self.cfg.rel_tol):
				break

			# add cut at current master solution
			self.backend.add_tangent_cut(p_t)

		zP, zR, zL = self.backend.pull_z()
		return OAResult(p=p_best, zP=zP, zR=zR, zL=zL, obj=UB, LB=LB, history=history)


# =============================================================================
# Public API (stand‑alone solver)
# =============================================================================
def solve_price_optimization_oa(
	*,
	p0: np.ndarray,
	delta: np.ndarray,
	k: int,
	c: np.ndarray,
	a: Optional[np.ndarray] = None,
	D: Optional[np.ndarray | sp.spmatrix] = None,
	S: Optional[np.ndarray | sp.spmatrix] = None,
	l: Optional[np.ndarray] = None,
	u: Optional[np.ndarray] = None,
	solver: str = "gurobi",
	max_iters: int = 50,
	abs_tol: float = 1e-6,
	rel_tol: float = 1e-5,
	time_limit: Optional[float] = None,
	trust_region: Optional[float] = None,
	initials: Optional[List[np.ndarray]] = None,
	verbose: bool = True,
) -> Dict[str, Any]:
	"""Kelley 型 OA による凸二次 (Q) 最小化 (同時に利益最大化 Z)。

	返却: dict(p, zP, zR, zL, obj=Q(p*), revenue=Z(p*), LB, history)
	"""
	# ---- 基本整形 ----
	p0 = np.asarray(p0, dtype=float).reshape(-1)
	delta = np.asarray(delta, dtype=float).reshape(-1)
	c = np.asarray(c, dtype=float).reshape(-1)
	n = p0.size
	assert delta.shape == (n,)
	assert c.shape == (n,)

	# ---- S, f の構築 ----
	if S is None:
		if a is None or D is None:
			raise ValueError("S を与えない場合は a と D が必要です。")
		a = np.asarray(a, dtype=float)
		if sp.issparse(D):
			S = (D + D.T).tocsr()
			f = a + D.T @ c  # D:T c (sparse) + a
		else:
			D = np.asarray(D, dtype=float)
			S = D + D.T
			f = a + D.T @ c
	else:
		S = S.tocsr() if sp.issparse(S) else np.asarray(S, dtype=float)
		if a is None or D is None:
			raise ValueError("f=a + D^T c を再現するため a, D も提供してください。")
		a = np.asarray(a, dtype=float)
		if sp.issparse(D):
			f = a + D.T @ c
		else:
			D = np.asarray(D, dtype=float)
			f = a + D.T @ c

	# ---- Bounds ----
	if l is None:
		l = np.full(n, -np.inf)
	if u is None:
		u = np.full(n, np.inf)
	l = np.asarray(l, dtype=float).reshape(-1)
	u = np.asarray(u, dtype=float).reshape(-1)

	cfg = OAConfig(
		max_iters=max_iters,
		abs_tol=abs_tol,
		rel_tol=rel_tol,
		time_limit=time_limit,
		trust_region=trust_region,
		verbose=verbose,
	)

	driver = OADriver(S, f, p0, delta, l, u, int(k), cfg, solver)

	# ---- 初期カット点 ----
	if not initials:
		# 無制約最小化 S p = f
		if sp.issparse(S):
			from scipy.sparse.linalg import spsolve  # type: ignore
			try:
				p_uncon = spsolve(S, f)
			except Exception:
				p_uncon = np.linalg.pinv(S.toarray()) @ f
		else:
			try:
				p_uncon = np.linalg.solve(S, f)
			except np.linalg.LinAlgError:
				p_uncon = np.linalg.pinv(S) @ f
		p_uncon = np.clip(p_uncon, l, u)
		initials = [p0.copy(), p_uncon]

	res = driver.run(initials)
	Z_best = -res.obj
	return {
		"p": res.p,
		"zP": res.zP,
		"zR": res.zR,
		"zL": res.zL,
		"obj": res.obj,
		"revenue": Z_best,
		"LB": res.LB,
		"history": res.history,
	}


def solve_price_optimization_oa_from_dataset(
	ds: Any,
	*,
	solver: str = "gurobi",
	max_iters: int = 50,
	abs_tol: float = 1e-6,
	rel_tol: float = 1e-5,
	time_limit: Optional[float] = None,
	trust_region: Optional[float] = None,
	verbose: bool = True,
) -> Dict[str, Any]:
	"""Dataset (`generate_dataset`) から直接 OA を実行するヘルパ。

	ds.initials のキー (init_1_p0, init_2_upper, ...) を検出して初期カットに利用。
	"""
	initials: List[np.ndarray] = []
	if hasattr(ds, "initials") and isinstance(ds.initials, dict):
		# 順序を意識して取り出し（p0 を最初に）
		for k_ in [
			"init_1_p0",
			"init_2_upper",
			"init_3_lower",
			"init_4_proj_unconstrained",
			"init_5_one_step_pg",
		]:
			if k_ in ds.initials:
				initials.append(np.asarray(ds.initials[k_], dtype=float))
	if not initials:
		initials = [np.asarray(ds.p0, dtype=float)]

	return solve_price_optimization_oa(
		p0=np.asarray(ds.p0, float),
		delta=np.asarray(ds.delta, float),
		k=int(ds.k),
		c=np.asarray(ds.c, float),
		a=np.asarray(ds.a, float),
		D=ds.D,  # sparse
		S=ds.S,
		l=np.asarray(ds.l, float),
		u=np.asarray(ds.u, float),
		solver=solver,
		max_iters=max_iters,
		abs_tol=abs_tol,
		rel_tol=rel_tol,
		time_limit=time_limit,
		trust_region=trust_region,
		initials=initials,
		verbose=verbose,
	)


# =============================================================================
# CLI
# =============================================================================
def _cli() -> None:  # pragma: no cover (manual usage)
	import argparse
	ap = argparse.ArgumentParser(description="OA (Kelley) cutting-plane for price optimization")
	ap.add_argument("--solver", type=str, default="gurobi", choices=["gurobi", "cplex", "mosek"])
	ap.add_argument("--n", type=int, default=50)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--max-iters", type=int, default=50)
	ap.add_argument("--abs-tol", type=float, default=1e-6)
	ap.add_argument("--rel-tol", type=float, default=1e-5)
	ap.add_argument("--time-limit", type=float, default=None)
	ap.add_argument("--trust-region", type=float, default=None)
	ap.add_argument("--quiet", action="store_true")
	args = ap.parse_args()

	if generate_dataset is None:
		raise RuntimeError("create_data.generate_dataset が見つかりません。")
	ds = generate_dataset(n=args.n, seed=args.seed)
	out = solve_price_optimization_oa_from_dataset(
		ds,
		solver=args.solver,
		max_iters=args.max_iters,
		abs_tol=args.abs_tol,
		rel_tol=args.rel_tol,
		time_limit=args.time_limit,
		trust_region=args.trust_region,
		verbose=not args.quiet,
	)
	gap_abs = out["obj"] - out["LB"]
	gap_rel = gap_abs / max(1.0, abs(out["obj"]))
	print("=== OA (Kelley) Result ===")
	print(f"solver        : {args.solver}")
	print(f"n, k          : {ds.meta['n']}, {ds.k}")
	print(f"Q(p*) (obj)   : {out['obj']:.6f}")
	print(f"Z(p*) revenue : {out['revenue']:.6f}")
	print(f"LB            : {out['LB']:.6f}")
	print(f"gap abs / rel : {gap_abs:.3e} / {gap_rel:.3e}")
	print(f"changes (raise/lower): {int(out['zR'].sum()) + int(out['zL'].sum())}")


if __name__ == "__main__":  # pragma: no cover
	_cli()

