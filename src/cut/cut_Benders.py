#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cut_Benders.py
===============
Logic / Combinatorial Benders (Generalized Benders Decomposition style) for the
price optimization problem with 3-way per-item split:

	For each item i:    p_i = p0_i   OR   p_i >= p0_i + δ_i   OR   p_i <= p0_i - δ_i

We solve the convex quadratic minimization of
	Q(p) = 1/2 p^T S p - f^T p
subject to the split disjunction, cardinality |{i: p_i != p0_i}| ≤ k, and box L ≤ p ≤ U.

Approach:
  Master MILP chooses a pattern (z0,z+,z-) and defines t_i = chosen reference value
	(p0_i, p0_i+δ_i, or p0_i-δ_i). Objective variable η underestimates Q(p*).
  Subproblem (convex QP) given the pattern enforces:
		p_i = t_i          if pattern=0
		p_i ≥ t_i          if pattern=+
		p_i ≤ t_i          if pattern=-
		L_i ≤ p_i ≤ U_i
	and returns optimal value v_hat plus KKT multipliers to build an optimality cut.

Optimality Cut (classic Benders / KKT subgradient form):
	η ≥ v_hat + Σ_i g_i (t_i - t_hat_i)
where g is a subgradient of the value function at current pattern (via multipliers).

Returned dict includes LB (master), UB (best subproblem value), gap, pattern, iterations.

Integration notes with existing project:
  - Data can be produced from `create_data.generate_dataset` (S=ds.S.toarray(), f=ds.f, p0=ds.p0,
	delta=ds.delta, L=ds.l, U=ds.u, k=ds.k)
  - Objective sign consistent with other cutting files (minimize Q; revenue = -Q).
  - This file does not depend on other custom modules to stay lightweight.

CLI usage example:
  python -m src.cut.cut_Benders --n 80 --seed 0 --max-iters 150 --tol 1e-4

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import numpy as np

try:
	import gurobipy as gp  # type: ignore
	from gurobipy import GRB, quicksum  # type: ignore
except Exception as e:  # pragma: no cover
	gp = None  # type: ignore
	GRB = None  # type: ignore
	quicksum = None  # type: ignore

__all__ = [
	"PriceOptData",
	"logic_benders_solve",
	"solve_benders_from_dataset",
]


# =============================================================================
# Data container
# =============================================================================
@dataclass
class PriceOptData:
	# Q(p) = 0.5 * p^T S p - f^T p  (S should be SPD)
	S: np.ndarray        # (n,n)
	f: np.ndarray        # (n,)
	p0: np.ndarray       # (n,)
	delta: np.ndarray    # (n,)
	L: np.ndarray        # (n,)
	U: np.ndarray        # (n,)
	k: int               # cardinality budget

	def __post_init__(self) -> None:
		n = self.S.shape[0]
		assert self.S.shape == (n, n)
		assert self.f.shape == (n,)
		assert self.p0.shape == (n,)
		assert self.delta.shape == (n,)
		assert self.L.shape == (n,)
		assert self.U.shape == (n,)
		assert np.all(self.U >= self.L)
		# Symmetrize defensively
		self.S = 0.5 * (self.S + self.S.T)


# =============================================================================
# Master MILP builder
# =============================================================================
def build_master(data: PriceOptData, log: bool = False):  # -> (model, vars...)
	if gp is None:
		raise RuntimeError("gurobipy がインストールされていません。")
	n = data.S.shape[0]
	m = gp.Model("master_logic_benders")
	if not log:
		m.Params.OutputFlag = 0

	z0 = m.addVars(n, vtype=GRB.BINARY, name="z0")  # keep p=p0
	zp = m.addVars(n, vtype=GRB.BINARY, name="zp")  # raise ≥ p0+δ
	zm = m.addVars(n, vtype=GRB.BINARY, name="zm")  # lower ≤ p0-δ
	y  = m.addVars(n, vtype=GRB.BINARY, name="y")   # changed or not
	t  = m.addVars(n, vtype=GRB.CONTINUOUS, name="t")  # chosen reference point
	eta = m.addVar(vtype=GRB.CONTINUOUS, name="eta")   # underestimator

	for i in range(n):
		m.addConstr(z0[i] + zp[i] + zm[i] == 1, name=f"onehot[{i}]")
		m.addConstr(zp[i] + zm[i] <= y[i], name=f"change_link[{i}]")
	m.addConstr(quicksum(y[i] for i in range(n)) <= data.k, name="cardinality")

	for i in range(n):
		p0_i = float(data.p0[i]); d_i = float(data.delta[i])
		m.addConstr(
			t[i] == p0_i * z0[i] + (p0_i + d_i) * zp[i] + (p0_i - d_i) * zm[i],
			name=f"t_def[{i}]",
		)

	# Box feasibility pruning (optional helpful)
	for i in range(n):
		if data.p0[i] + data.delta[i] > data.U[i] + 1e-9:
			m.addConstr(zp[i] == 0, name=f"forbid_raise[{i}]")
		if data.p0[i] - data.delta[i] < data.L[i] - 1e-9:
			m.addConstr(zm[i] == 0, name=f"forbid_lower[{i}]")

	m.setObjective(eta, GRB.MINIMIZE)
	m.update()
	return m, z0, zp, zm, y, t, eta


# =============================================================================
# Subproblem solver (convex QP) & subgradient extraction
# =============================================================================
def solve_sub_and_get_cut(
	data: PriceOptData,
	s_pattern: List[str],
	t_hat: np.ndarray,
	log: bool = False,
) -> Tuple[bool, float, np.ndarray]:
	if gp is None:
		raise RuntimeError("gurobipy がインストールされていません。")
	n = data.S.shape[0]
	sub = gp.Model("sub_qp")
	if not log:
		sub.Params.OutputFlag = 0
	sub.Params.Method = 2  # barrier
	sub.Params.Crossover = 0

	p = sub.addVars(n, lb=data.L.tolist(), ub=data.U.tolist(), vtype=GRB.CONTINUOUS, name="p")
	obj = gp.QuadExpr()
	for i in range(n):
		for j in range(n):
			sij = data.S[i, j]
			if sij != 0.0:
				obj += 0.5 * sij * p[i] * p[j]
	for i in range(n):
		fi = data.f[i]
		if fi != 0.0:
			obj += - float(fi) * p[i]
	sub.setObjective(obj, GRB.MINIMIZE)

	cons_eq: Dict[int, Any] = {}
	cons_lo: Dict[int, Any] = {}
	cons_up: Dict[int, Any] = {}
	for i in range(n):
		if s_pattern[i] == "0":
			cons_eq[i] = sub.addConstr(p[i] == float(t_hat[i]), name=f"fix[{i}]")
		elif s_pattern[i] == "+":
			cons_lo[i] = sub.addConstr(-p[i] <= -float(t_hat[i]), name=f"lo[{i}]")  # p>=t
		elif s_pattern[i] == "-":
			cons_up[i] = sub.addConstr(p[i] <= float(t_hat[i]), name=f"up[{i}]")    # p<=t
		else:
			raise ValueError("pattern must be '0','+','-'")

	sub.optimize()
	if sub.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
		return False, np.inf, np.zeros(n)
	if sub.Status != GRB.OPTIMAL:
		raise RuntimeError(f"Subproblem status={sub.Status}")

	v_hat = sub.ObjVal
	g = np.zeros(n)
	for i, c in cons_eq.items():
		g[i] = -c.Pi
	for i, c in cons_lo.items():
		g[i] = c.Pi
	for i, c in cons_up.items():
		g[i] = -c.Pi
	return True, v_hat, g


# =============================================================================
# Helpers
# =============================================================================
def extract_pattern(z0_vals, zp_vals, zm_vals) -> List[str]:
	n = len(z0_vals)
	patt: List[str] = []
	for i in range(n):
		v0 = round(float(z0_vals[i])); vp = round(float(zp_vals[i])); vm = round(float(zm_vals[i]))
		if v0 + vp + vm != 1:
			mx = max([(v0, '0'), (vp, '+'), (vm, '-')])[1]
			patt.append(mx)
		else:
			patt.append('0' if v0 == 1 else ('+' if vp == 1 else '-'))
	return patt


def no_good_cut(model: Any, z0, zp, zm, patt: List[str]) -> None:
	n = len(patt)
	lhs = quicksum((z0[i] if patt[i] == '0' else (zp[i] if patt[i] == '+' else zm[i])) for i in range(n))
	model.addConstr(lhs <= n - 1, name=f"nogood[{model.NumConstrs}]")
	model.update()


# =============================================================================
# Main Benders loop
# =============================================================================
def logic_benders_solve(
	data: PriceOptData,
	max_iters: int = 200,
	tol: float = 1e-5,
	log: bool = True,
) -> Dict[str, Any]:
	m, z0, zp, zm, y, t, eta = build_master(data, log=log)
	LB = -np.inf
	UB = +np.inf
	best: Dict[str, Any] = dict(UB=UB, patt=None)

	for it in range(1, max_iters + 1):
		if log:
			print(f"\n=== Benders Iteration {it} ===")
		m.optimize()
		if m.Status != GRB.OPTIMAL:
			raise RuntimeError(f"Master optimization failed status={m.Status}")
		LB = m.ObjVal
		z0_vals = [z0[i].X for i in range(data.S.shape[0])]
		zp_vals = [zp[i].X for i in range(data.S.shape[0])]
		zm_vals = [zm[i].X for i in range(data.S.shape[0])]
		patt = extract_pattern(z0_vals, zp_vals, zm_vals)
		t_hat = np.array([t[i].X for i in range(data.S.shape[0])])

		if log:
			changes = sum(1 for s in patt if s != '0')
			print(f"Master LB (eta) = {LB:.6f}, changes={changes}/{data.k}")

		feasible, v_hat, g = solve_sub_and_get_cut(data, patt, t_hat, log=False)
		if not feasible:
			if log:
				print("Subproblem infeasible -> adding no-good cut")
			no_good_cut(m, z0, zp, zm, patt)
			continue

		if v_hat < UB:
			UB = v_hat
			best.update(UB=UB, patt=patt)

		gap = (UB - LB) / max(1.0, abs(UB))
		if log:
			print(f"Sub value={v_hat:.6f}, UB={UB:.6f}, LB={LB:.6f}, gap={100*gap:.3f}%")

		cut_expr = v_hat + quicksum(float(g[i]) * (t[i] - float(t_hat[i])) for i in range(data.S.shape[0]))
		m.addConstr(eta >= cut_expr, name=f"optcut[{m.NumConstrs}]")
		m.update()

		if gap <= tol:
			if log:
				print(f"Converged: gap={gap:.3e}")
			break

	return {
		"LB": LB,
		"UB": UB,
		"gap": (UB - LB) / max(1.0, abs(UB)),
		"pattern": best["patt"],
		"status": "converged" if (UB - LB) / max(1.0, abs(UB)) <= tol else "stopped",
		"iterations": it,
	}


# =============================================================================
# Dataset helper (optional integration with create_data)
# =============================================================================
def solve_benders_from_dataset(
	ds: Any,
	*,
	max_iters: int = 200,
	tol: float = 1e-5,
	log: bool = True,
) -> Dict[str, Any]:
	"""Helper bridging a PriceOptDataset (from create_data) to Benders solver."""
	S = ds.S.toarray() if hasattr(ds.S, 'toarray') else ds.S
	data = PriceOptData(
		S=S,
		f=ds.f,
		p0=ds.p0,
		delta=ds.delta,
		L=ds.l,
		U=ds.u,
		k=int(ds.k),
	)
	return logic_benders_solve(data, max_iters=max_iters, tol=tol, log=log)


# =============================================================================
# CLI
# =============================================================================
def _cli():  # pragma: no cover
	import argparse
	ap = argparse.ArgumentParser(description="Logic/Combinatorial Benders for Price Optimization")
	ap.add_argument("--n", type=int, default=50)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--max-iters", type=int, default=200)
	ap.add_argument("--tol", type=float, default=1e-5)
	ap.add_argument("--quiet", action="store_true")
	args = ap.parse_args()
	if gp is None:
		raise RuntimeError("gurobipy が利用できません。")
	try:
		from create_data.create_data import generate_dataset  # type: ignore
	except Exception:
		raise RuntimeError("create_data.generate_dataset が見つかりません。PYTHONPATH を確認してください。")
	ds = generate_dataset(n=args.n, seed=args.seed)
	res = solve_benders_from_dataset(ds, max_iters=args.max_iters, tol=args.tol, log=not args.quiet)
	print("=== Benders Result ===")
	print(f"Status : {res['status']}")
	print(f"LB     : {res['LB']:.6f}")
	print(f"UB     : {res['UB']:.6f}")
	print(f"Gap    : {100*res['gap']:.3f}%")
	print(f"Iters  : {res['iterations']}")
	print(f"Pattern: {res['pattern']}")


if __name__ == "__main__":  # pragma: no cover
	_cli()

