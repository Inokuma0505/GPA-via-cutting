#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cut_disjunctive.py
===================
3 分岐 (stay / raise / lower) を持つ価格最適化問題に対する
Disjunctive & Split (Balas) + Lift-&-Project 風カッティングプレーン実装。

目的関数（他ファイルとの整合性）:
  Q(p) = 1/2 p^T S p - b^T p (凸二次, 最小化)。利得最大化なら Z(p) = -Q(p)

本ファイルでは 2 つの基本定式化:
  (A) convex_hull  : 3 区間和集合の理想凸包 (tight LP)
  (B) indicators   : indicator 制約ベース。凸包不等式は user cut として追加可。

オプションで:
  * θ 変数 + Kelley OA タンジェントカット (use_theta_OA=True)
  * カーディナリティ Σ y_i ≤ k 用の cover cut（単純な (k+1)-cover）
  * split convex hull 切除 (base=indicators の補強)
  * 信頼領域 |p - p0|_∞ ≤ R

CLI デモ: ファイル末尾の demo() を参照。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import math
import numpy as np

try:  # Gurobi はオプション（開発環境でインストール推奨）
	import gurobipy as gp  # type: ignore
	from gurobipy import GRB  # type: ignore
except Exception:  # pragma: no cover
	gp = None  # type: ignore
	GRB = None  # type: ignore

__all__ = [
	"ModelData",
	"Options",
	"DisjunctiveSplitCutOptimizer",
	"demo",
]


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class ModelData:
	S: np.ndarray               # (n,n) PSD (convex quadratic term)
	b: np.ndarray               # (n,) linear term so that Q(p) = 0.5 p^T S p - b^T p
	p0: np.ndarray              # (n,) baseline prices
	delta: np.ndarray           # (n,) minimal change magnitudes δ_i > 0
	l: np.ndarray               # (n,) lower bounds for prices
	u: np.ndarray               # (n,) upper bounds for prices
	k: int                      # cardinality budget for changes (number of items allowed to change)


@dataclass
class Options:
	base_formulation: str = "convex_hull"   # "convex_hull" or "indicators"
	use_theta_OA: bool = True               # MILP + θ + OA tangents (else MIQP direct Q(p))
	root_cut_rounds: int = 3                # ルートで集中的にカット（単純版: callback 内のみ）
	max_cuts_per_node: int = 200            # セーフティ上限
	cut_tolerance: float = 1e-6
	add_cover_cuts: bool = True
	add_split_cuts: bool = True             # base=indicators のとき凸包不等式を user cut 追加
	trust_radius: Optional[float] = None    # |p - p0|_inf ≤ R
	mip_focus: Optional[int] = 1            # Gurobi Param MIPFocus
	time_limit: Optional[float] = None      # 秒
	verbose: bool = True


# =============================================================================
# Optimizer class
# =============================================================================
class DisjunctiveSplitCutOptimizer:
	def __init__(self, data: ModelData, opts: Options):
		self.data = data
		self.opts = opts

		self.n = int(data.S.shape[0])
		assert data.S.shape == (self.n, self.n)
		assert data.b.shape == (self.n,)
		# Defensive copies
		self.S = np.array(data.S, dtype=float)
		self.b = np.array(data.b, dtype=float)
		self.p0 = np.array(data.p0, dtype=float)
		self.delta = np.array(data.delta, dtype=float)
		self.l = np.array(data.l, dtype=float)
		self.u = np.array(data.u, dtype=float)
		self.k = int(data.k)

		# Gurobi objects (build_model で設定)
		# モジュール未導入環境でも型チェックが落ちないよう Any 扱い
		self.model: Any = None  # type: ignore
		self.p = None
		self.y = None
		self.zP = None
		self.zR = None
		self.zL = None
		self.theta = None

		# Diagnostics counters
		self.num_oa_cuts = 0
		self.num_cover_cuts = 0
		self.num_split_cuts = 0

	# ------------------------------------------------------------------
	# Quadratic objective utilities
	def Q_value(self, p: np.ndarray) -> float:
		return 0.5 * float(p @ self.S @ p) - float(self.b @ p)

	def Q_grad(self, p: np.ndarray) -> np.ndarray:
		return self.S @ p - self.b

	# ------------------------------------------------------------------
	# Model construction
	def build_model(self):  # -> gp.Model (gurobipy 未インストール環境で型エラー回避)
		if gp is None:
			raise RuntimeError("gurobipy が利用できません。Gurobi がインストールされた環境で実行してください。")

		m = gp.Model("price_opt_disjunctive")
		n = self.n
		l, u, p0, delta = self.l, self.u, self.p0, self.delta

		# Variables
		p = m.addVars(n, lb=l.tolist(), ub=u.tolist(), vtype=GRB.CONTINUOUS, name="p")
		zP = m.addVars(n, vtype=GRB.BINARY, name="zP")  # stay
		zR = m.addVars(n, vtype=GRB.BINARY, name="zR")  # raise
		zL = m.addVars(n, vtype=GRB.BINARY, name="zL")  # lower
		y = m.addVars(n, vtype=GRB.BINARY, name="y")    # change indicator

		# One-hot and change linking
		m.addConstrs((zP[i] + zR[i] + zL[i] == 1 for i in range(n)), name="onehot")
		m.addConstrs((y[i] == zR[i] + zL[i] for i in range(n)), name="link_y")
		m.addConstr(gp.quicksum(y[i] for i in range(n)) <= self.k, name="cardinality")

		# Optional trust region
		if self.opts.trust_radius is not None:
			R = float(self.opts.trust_radius)
			m.addConstrs((p[i] - p0[i] <= R for i in range(n)), name="trust_up")
			m.addConstrs((p0[i] - p[i] <= R for i in range(n)), name="trust_dn")

		base = self.opts.base_formulation.lower()
		if base not in ("convex_hull", "indicators"):
			raise ValueError("Options.base_formulation must be 'convex_hull' or 'indicators'")

		if base == "convex_hull":
			# Ideal convex hull inequalities (tight)
			m.addConstrs(
				(l[i] * zL[i] + p0[i] * zP[i] + (p0[i] + delta[i]) * zR[i] <= p[i] for i in range(n)),
				name="split_lower_ch",
			)
			m.addConstrs(
				((p0[i] - delta[i]) * zL[i] + p0[i] * zP[i] + u[i] * zR[i] >= p[i] for i in range(n)),
				name="split_upper_ch",
			)
		else:
			# Indicator constraints
			m.addConstrs(((zP[i] == 1) >> (p[i] <= p0[i]) for i in range(n)), name="ind_eq_le")
			m.addConstrs(((zP[i] == 1) >> (p[i] >= p0[i]) for i in range(n)), name="ind_eq_ge")
			m.addConstrs(((zR[i] == 1) >> (p[i] >= p0[i] + delta[i]) for i in range(n)), name="ind_up_ge")
			m.addConstrs(((zL[i] == 1) >> (p[i] <= p0[i] - delta[i]) for i in range(n)), name="ind_dn_le")

		# Objective: MIQP direct or MILP + θ
		if self.opts.use_theta_OA:
			theta = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")
			m.setObjective(theta, GRB.MINIMIZE)
			self.theta = theta
		else:
			# build convex quadratic Q(p)
			quad = gp.QuadExpr()
			for i in range(n):
				# diagonal
				if abs(self.S[i, i]) > 0:
					quad.addTerm(0.5 * self.S[i, i], p[i], p[i])
				for j in range(i + 1, n):
					if abs(self.S[i, j]) > 0:
						quad.addTerm(self.S[i, j], p[i], p[j])
			lin = -gp.quicksum(self.b[i] * p[i] for i in range(n))
			m.setObjective(quad + lin, GRB.MINIMIZE)
			self.theta = None

		# Save state
		self.model = m
		self.p, self.y, self.zP, self.zR, self.zL = p, y, zP, zR, zL

		# Gurobi parameters
		m.Params.OutputFlag = 1 if self.opts.verbose else 0
		if self.opts.mip_focus is not None:
			m.Params.MIPFocus = int(self.opts.mip_focus)
		if self.opts.time_limit is not None:
			m.Params.TimeLimit = float(self.opts.time_limit)
		m.Params.Cuts = 2
		return m

	# ------------------------------------------------------------------
	# Callback helpers
	def _get_node_rel(self, model, varlist) -> List[float]:
		return [model.cbGetNodeRel(v) for v in varlist]

	def _oa_tangent_cut(self, model, p_rel: List[float]) -> bool:
		if self.theta is None:
			return False
		pbar = np.array(p_rel, dtype=float)
		f = self.Q_value(pbar)
		g = self.Q_grad(pbar)
		lhs = self.theta - gp.quicksum(float(g[i]) * self.p[i] for i in range(self.n))
		rhs = f - float(g @ pbar)
		model.cbCut(lhs >= rhs)
		self.num_oa_cuts += 1
		return True

	def _cover_cut(self, model, y_rel: List[float]) -> bool:
		if self.k >= self.n:
			return False
		viol = sum(y_rel) - self.k
		if viol <= self.opts.cut_tolerance:
			return False
		idx_sorted = sorted(range(self.n), key=lambda i: y_rel[i], reverse=True)
		C = idx_sorted[: min(self.k + 1, self.n)]
		if len(C) <= self.k:
			return False
		model.cbCut(gp.quicksum(self.y[i] for i in C) <= self.k)
		self.num_cover_cuts += 1
		return True

	def _split_convex_hull_cuts(
		self,
		model,
		p_rel: List[float],
		zP_rel: List[float],
		zR_rel: List[float],
		zL_rel: List[float],
	) -> bool:
		added = 0
		tol = self.opts.cut_tolerance
		for i in range(self.n):
			lower_lhs = self.l[i] * zL_rel[i] + self.p0[i] * zP_rel[i] + (self.p0[i] + self.delta[i]) * zR_rel[i]
			if p_rel[i] < lower_lhs - tol:
				model.cbCut(
					self.p[i] >= self.l[i] * self.zL[i] + self.p0[i] * self.zP[i] + (self.p0[i] + self.delta[i]) * self.zR[i]
				)
				added += 1
			upper_rhs = (self.p0[i] - self.delta[i]) * zL_rel[i] + self.p0[i] * zP_rel[i] + self.u[i] * zR_rel[i]
			if p_rel[i] > upper_rhs + tol:
				model.cbCut(
					self.p[i] <= (self.p0[i] - self.delta[i]) * self.zL[i] + self.p0[i] * self.zP[i] + self.u[i] * self.zR[i]
				)
				added += 1
			if added >= self.opts.max_cuts_per_node:
				break
		self.num_split_cuts += added
		return added > 0

	# ------------------------------------------------------------------
	def _callback(self, model, where):  # pragma: no cover (solver runtime)
		if where == GRB.Callback.MIPNODE:
			status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
			if status != GRB.OPTIMAL:
				return
			p_rel = self._get_node_rel(model, [self.p[i] for i in range(self.n)])
			y_rel = self._get_node_rel(model, [self.y[i] for i in range(self.n)])
			zP_rel = self._get_node_rel(model, [self.zP[i] for i in range(self.n)])
			zR_rel = self._get_node_rel(model, [self.zR[i] for i in range(self.n)])
			zL_rel = self._get_node_rel(model, [self.zL[i] for i in range(self.n)])

			cuts_added = 0
			if self.opts.use_theta_OA and cuts_added < self.opts.max_cuts_per_node:
				if self._oa_tangent_cut(model, p_rel):
					cuts_added += 1
			if self.opts.add_cover_cuts and cuts_added < self.opts.max_cuts_per_node:
				if self._cover_cut(model, y_rel):
					cuts_added += 1
			if self.opts.add_split_cuts and cuts_added < self.opts.max_cuts_per_node:
				self._split_convex_hull_cuts(model, p_rel, zP_rel, zR_rel, zL_rel)
		elif where == GRB.Callback.MIPSOL:
			# integer solution reached — could add lazy constraints if needed
			pass

	# ------------------------------------------------------------------
	def solve(self) -> Tuple[np.ndarray, Dict[str, Any]]:
		m = self.model if self.model is not None else self.build_model()
		m.optimize(self._callback)
		status = m.Status
		if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL):
			raise RuntimeError(f"Gurobi ended with status {status}")
		p_sol = np.array([self.p[i].X for i in range(self.n)], dtype=float)
		info: Dict[str, Any] = {
			"status": status,
			"obj_val": m.ObjVal if m.SolCount > 0 else None,
			"num_oa_cuts": self.num_oa_cuts,
			"num_cover_cuts": self.num_cover_cuts,
			"num_split_cuts": self.num_split_cuts,
			"gap": m.MIPGap if hasattr(m, "MIPGap") else None,
			"runtime": m.Runtime if hasattr(m, "Runtime") else None,
		}
		return p_sol, info


# =============================================================================
# Demo (manual run)
# =============================================================================
def _spd_matrix(n: int, seed: int = 0) -> np.ndarray:
	rng = np.random.default_rng(seed)
	A = rng.normal(size=(n, n))
	S = A.T @ A + 1e-2 * np.eye(n)
	return S


def demo(
	n: int = 20,
	k: int = 3,
	seed: int = 0,
	base: str = "convex_hull",
	use_theta_OA: bool = True,
) -> None:  # pragma: no cover
	S = _spd_matrix(n, seed)
	rng = np.random.default_rng(seed + 1)
	b = rng.uniform(0.5, 1.5, size=n)
	p0 = rng.uniform(5.0, 15.0, size=n)
	delta = rng.uniform(0.5, 2.0, size=n)
	l = np.maximum(0.0, p0 - 3.0)
	u = p0 + 3.0
	data = ModelData(S=S, b=b, p0=p0, delta=delta, l=l, u=u, k=k)
	opts = Options(base_formulation=base, use_theta_OA=use_theta_OA, verbose=True, time_limit=60)
	solver = DisjunctiveSplitCutOptimizer(data, opts)
	solver.build_model()
	p_star, info = solver.solve()
	print("Solution p*:", p_star)
	print("Info:", info)


if __name__ == "__main__":  # pragma: no cover
	if gp is not None:
		demo()
	else:
		print("Gurobi not available; module loaded.")

