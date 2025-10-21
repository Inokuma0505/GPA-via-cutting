# -*- coding: utf-8 -*-
"""
Lagrangian Cutting Plane (cardinality-relaxation lower bounds) + OA (Kelley) for price optimization
- Master: MILP with OA cuts and Lagrangian cuts; keeps logic/link constraints, DROPS sum y <= k.
- Subproblem: MIQP (exact) or QP (continuous relaxation) to evaluate ϕ(λ) and its subgradient g(λ) = sum y* - k.
- Feasible projection: builds a k-change, ±δ-feasible price vector from a candidate to update UB quickly.

Requirements:
  pip install numpy gurobipy

Notation:
  Q(p) = 0.5 * p^T S p - f^T p  (convex; S ≻ 0), gradQ(p) = S p - f
  Actions per i: zP=keep, zR=raise, zL=lower; y_i = zR_i + zL_i
  Cardinality: sum_i y_i ≤ k (RELAXED in master; enforced indirectly via Lagrangian cuts)

Author: ChatGPT (切除平面プロジェクト)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum


# -----------------------------
# Data structure and utilities
# -----------------------------

@dataclass
class PriceOptData:
    S: np.ndarray            # (n,n) SPD matrix
    f: np.ndarray            # (n,) vector (linear term)
    p0: np.ndarray           # (n,) baseline price
    delta: np.ndarray        # (n,) minimum absolute move when changed
    k: int                   # max number of changed items
    l: Optional[np.ndarray] = None  # (n,) lower bounds (can be None)
    u: Optional[np.ndarray] = None  # (n,) upper bounds (can be None)

    def __post_init__(self):
        n = self.S.shape[0]
        assert self.S.shape == (n, n)
        assert self.f.shape == (n,)
        assert self.p0.shape == (n,)
        assert self.delta.shape == (n,)
        assert isinstance(self.k, int) and 0 <= self.k <= n
        if self.l is not None:
            assert self.l.shape == (n,)
        if self.u is not None:
            assert self.u.shape == (n,)


def Q_value_grad(p: np.ndarray, data: PriceOptData) -> Tuple[float, np.ndarray]:
    """Q(p) = 0.5 p^T S p - f^T p, grad = S p - f"""
    val = 0.5 * float(p @ (data.S @ p)) - float(data.f @ p)
    grad = data.S @ p - data.f
    return val, grad


def unconstrained_minimizer(data: PriceOptData) -> np.ndarray:
    """Solve S p = f (SPD) for unconstrained minimizer."""
    return np.linalg.solve(data.S, data.f)


def clip_bounds(p: np.ndarray, data: PriceOptData) -> np.ndarray:
    """Clip by optional bounds l,u."""
    q = p.copy()
    if data.l is not None:
        q = np.maximum(q, data.l)
    if data.u is not None:
        q = np.minimum(q, data.u)
    return q


def feasible_projection_k_delta(
    p_cand: np.ndarray, data: PriceOptData
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic projection to Ω_{k, p0, δ}:
      - Start from p0.
      - Choose top-k indices by |descent signal| (use grad at p_cand) and move them by ±δ_i
        in the sign that decreases Q (based on grad sign), respecting [l, u].
      - Others remain at p0.
    Returns (p_feas, y_binary) with y_i ∈ {0,1} as change indicator.
    """
    n = data.S.shape[0]
    _, g = Q_value_grad(p_cand, data)
    # Score: magnitude of gradient times δ (proxy for benefit)
    score = np.abs(g) * np.maximum(data.delta, 1e-12)
    idx = np.argsort(-score)[: data.k]  # top-k
    p = data.p0.copy()
    y = np.zeros(n, dtype=int)
    for i in idx:
        if g[i] < 0:
            # Increasing p_i decreases Q
            pi = data.p0[i] + data.delta[i]
        else:
            # Decreasing p_i decreases Q
            pi = data.p0[i] - data.delta[i]
        # Respect bounds
        if data.u is not None:
            pi = min(pi, data.u[i])
        if data.l is not None:
            pi = max(pi, data.l[i])
        # If move collapsed to p0 due to bounds, keep it as p0 (i.e., no change)
        if np.abs(pi - data.p0[i]) >= data.delta[i] - 1e-12:
            p[i] = pi
            y[i] = 1
    return p, y


# ----------------------------------
# Subproblem φ(λ): exact or relaxed
# ----------------------------------

def _add_quadratic_Q(model: gp.Model, p_vars: List[gp.Var], S: np.ndarray, f: np.ndarray) -> gp.QuadExpr:
    """Build 0.5 p^T S p - f^T p as a QuadExpr. Assumes S symmetric."""
    n = len(p_vars)
    obj = gp.QuadExpr()
    # off-diagonal (i<j): coefficient S_ij
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] != 0.0:
                obj += S[i, j] * p_vars[i] * p_vars[j]
    # diagonal 0.5 * S_ii
    for i in range(n):
        if S[i, i] != 0.0:
            obj += 0.5 * S[i, i] * p_vars[i] * p_vars[i]
    # linear part -f^T p
    obj += gp.LinExpr([-float(fi) for fi in f], p_vars)
    return obj


def evaluate_phi_lambda(
    data: PriceOptData,
    lam: float,
    exact_integers: bool = True,
    time_limit: Optional[float] = 10.0,
    mip_gap: Optional[float] = 1e-4,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute φ(λ) = min_{p,z} Q(p) + λ * sum_i y_i  s.t. logic/link constraints (NO sum y ≤ k).
    Returns (phi, sum_y, p_sol).
    - exact_integers=True: z binaries + indicator constraints (accurate, MIQP).
    - exact_integers=False: z ∈ [0,1] + big-M linking (fast QP lower bound).
    """
    n = data.S.shape[0]
    m = gp.Model("phi_lambda")
    m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap

    # Variables
    lb = data.l if data.l is not None else -np.inf * np.ones(n)
    ub = data.u if data.u is not None else +np.inf * np.ones(n)
    p = m.addVars(n, lb=lb.tolist(), ub=ub.tolist(), vtype=GRB.CONTINUOUS, name="p")

    if exact_integers:
        vtype = GRB.BINARY
    else:
        vtype = GRB.CONTINUOUS

    zP = m.addVars(n, lb=0.0, ub=1.0, vtype=vtype, name="zP")
    zR = m.addVars(n, lb=0.0, ub=1.0, vtype=vtype, name="zR")
    zL = m.addVars(n, lb=0.0, ub=1.0, vtype=vtype, name="zL")
    yv = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")  # y = zR + zL

    # Partition and link
    for i in range(n):
        m.addConstr(zP[i] + zR[i] + zL[i] == 1, name=f"part_{i}")
        m.addConstr(yv[i] == zR[i] + zL[i], name=f"ydef_{i}")

    if exact_integers:
        # Indicator constraints (tighter, need binaries)
        for i in range(n):
            # keep: p_i == p0_i
            m.addGenConstrIndicator(zP[i], True, p[i] - float(data.p0[i]), GRB.EQUAL, 0.0, name=f"indP_{i}")
            # raise: p_i >= p0_i + delta_i
            m.addGenConstrIndicator(zR[i], True, p[i] - float(data.p0[i] + data.delta[i]),
                                    GRB.GREATER_EQUAL, 0.0, name=f"indR_{i}")
            # lower: p_i <= p0_i - delta_i
            m.addGenConstrIndicator(zL[i], True, p[i] - float(data.p0[i] - data.delta[i]),
                                    GRB.LESS_EQUAL, 0.0, name=f"indL_{i}")
    else:
        # Big-M relaxed linking (allows z in [0,1])
        # Build reasonable M_i from bounds if available; otherwise fallback
        if data.l is not None and data.u is not None:
            M = (data.u - data.l).astype(float)
        else:
            span = float(np.max(np.abs(data.p0)) + np.max(data.delta) + 10.0)
            M = np.full(n, max(10.0, span), dtype=float)

        for i in range(n):
            # keep: |p - p0| <= M*(1 - zP)
            m.addConstr(p[i] - float(data.p0[i]) <= M[i] * (1 - zP[i]), name=f"relP_ub_{i}")
            m.addConstr(float(data.p0[i]) - p[i] <= M[i] * (1 - zP[i]), name=f"relP_lb_{i}")
            # raise: p >= p0 + δ - M*(1 - zR)
            m.addConstr(p[i] >= float(data.p0[i] + data.delta[i]) - M[i] * (1 - zR[i]), name=f"relR_{i}")
            # lower: p <= p0 - δ + M*(1 - zL)
            m.addConstr(p[i] <= float(data.p0[i] - data.delta[i]) + M[i] * (1 - zL[i]), name=f"relL_{i}")

    # Objective: Q(p) + λ * sum y
    obj = _add_quadratic_Q(m, [p[i] for i in range(n)], data.S, data.f)
    obj += lam * quicksum(yv[i] for i in range(n))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"phi(λ) solve failed with status {m.Status}")

    p_sol = np.array([p[i].X for i in range(n)], dtype=float)
    y_sum = float(sum(yv[i].X for i in range(n)))
    phi = float(m.ObjVal)
    return phi, y_sum, p_sol


# ---------------------------------
# Master (OA + Lagrangian cuts)
# ---------------------------------

class LagrangianOA_Master:
    def __init__(self, data: PriceOptData):
        self.data = data
        self.n = data.S.shape[0]
        self.m = gp.Model("master_LagrangianOA")
        self.m.Params.OutputFlag = 0

        # Variables
        lb = data.l if data.l is not None else -np.inf * np.ones(self.n)
        ub = data.u if data.u is not None else +np.inf * np.ones(self.n)
        self.p = self.m.addVars(self.n, lb=lb.tolist(), ub=ub.tolist(), vtype=GRB.CONTINUOUS, name="p")
        self.zP = self.m.addVars(self.n, vtype=GRB.BINARY, name="zP")
        self.zR = self.m.addVars(self.n, vtype=GRB.BINARY, name="zR")
        self.zL = self.m.addVars(self.n, vtype=GRB.BINARY, name="zL")
        self.yv = self.m.addVars(self.n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
        self.eta = self.m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="eta")

        # Partition & link (no cardinality constraint here!)
        for i in range(self.n):
            self.m.addConstr(self.zP[i] + self.zR[i] + self.zL[i] == 1, name=f"part_{i}")
            self.m.addConstr(self.yv[i] == self.zR[i] + self.zL[i], name=f"ydef_{i}")
            # Indicator constraints
            self.m.addGenConstrIndicator(self.zP[i], True, self.p[i] - float(self.data.p0[i]), GRB.EQUAL, 0.0, name=f"indP_{i}")
            self.m.addGenConstrIndicator(self.zR[i], True, self.p[i] - float(self.data.p0[i] + self.data.delta[i]),
                                         GRB.GREATER_EQUAL, 0.0, name=f"indR_{i}")
            self.m.addGenConstrIndicator(self.zL[i], True, self.p[i] - float(self.data.p0[i] - self.data.delta[i]),
                                         GRB.LESS_EQUAL, 0.0, name=f"indL_{i}")

        # Objective: min eta
        self.m.setObjective(self.eta, GRB.MINIMIZE)

        # Tracking cuts
        self.oa_cuts: List[gp.Constr] = []
        self.lagr_cuts: List[gp.Constr] = []

    def add_oa_cut(self, p_t: np.ndarray):
        """η ≥ Q(p_t) + ∇Q(p_t)^T (p - p_t)"""
        val, grad = Q_value_grad(p_t, self.data)
        lin = gp.LinExpr()
        for i in range(self.n):
            lin += float(grad[i]) * (self.p[i] - float(p_t[i]))
        c = self.m.addConstr(self.eta >= float(val) + lin, name=f"OA_{len(self.oa_cuts)}")
        self.oa_cuts.append(c)

    def add_lagr_cut(self, phi_val: float, lam: float):
        """η ≥ φ(λ) + λ (sum y - k)"""
        cut_expr = phi_val + lam * (quicksum(self.yv[i] for i in range(self.n)) - float(self.data.k))
        c = self.m.addConstr(self.eta >= cut_expr, name=f"LAGR_{len(self.lagr_cuts)}")
        self.lagr_cuts.append(c)

    def optimize(self, time_limit: Optional[float] = 10.0) -> Tuple[float, np.ndarray, float]:
        if time_limit is not None:
            self.m.Params.TimeLimit = time_limit
        self.m.optimize()
        # Return (lower bound eta, p_solution, sum_y)
        p_sol = np.array([self.p[i].X for i in range(self.n)], dtype=float)
        y_sum = float(sum(self.yv[i].X for i in range(self.n)))
        return float(self.eta.X), p_sol, y_sum


# ---------------------------------
# Main algorithm
# ---------------------------------

@dataclass
class LCPConfig:
    max_iters: int = 30
    tol_gap: float = 1e-4
    lam0: float = 0.0
    beta_polyak: float = 1.0  # (0,2) typically
    sub_time_limit: float = 10.0
    master_time_limit: float = 10.0
    sub_exact_every: int = 1  # use exact MIQP each iteration by default; set >1 to interleave relaxed solves


def lagrangian_cutting_plane(
    data: PriceOptData, cfg: LCPConfig = LCPConfig()
) -> Dict[str, object]:
    """
    Hybrid Lagrangian cuts + OA. Returns dict with best solution and logs.
    """
    n = data.S.shape[0]
    # 1) Init
    p_hat = unconstrained_minimizer(data)
    p_hat = clip_bounds(p_hat, data)

    master = LagrangianOA_Master(data)
    master.add_oa_cut(p_hat)  # Kelley seed

    # Feasible UB via projection from p_hat
    p_feas, _ = feasible_projection_k_delta(p_hat, data)
    UB, _ = Q_value_grad(p_feas, data)
    best_p = p_feas.copy()

    # λ init
    lam = max(0.0, cfg.lam0)

    history = []
    for it in range(cfg.max_iters):
        # 2) Evaluate φ(λ)
        exact = ((it % cfg.sub_exact_every) == 0)
        phi, y_sum_sub, p_sub = evaluate_phi_lambda(
            data, lam, exact_integers=exact,
            time_limit=cfg.sub_time_limit, mip_gap=1e-4
        )

        g = y_sum_sub - data.k  # subgradient

        # Add cuts
        master.add_lagr_cut(phi, lam)
        master.add_oa_cut(p_sub)

        # 3) Solve master
        eta_lb, p_bar, y_sum_bar = master.optimize(time_limit=cfg.master_time_limit)

        # 4) Feasible projection for UB update
        p_proj, _ = feasible_projection_k_delta(p_bar, data)
        UB_candidate, _ = Q_value_grad(p_proj, data)
        if UB_candidate < UB - 1e-12:
            UB = UB_candidate
            best_p = p_proj.copy()

        # 5) Polyak step for λ
        step = cfg.beta_polyak * max(0.0, UB - phi) / (g * g + 1e-12)
        lam = max(0.0, lam + step * g)

        gap = max(0.0, UB - eta_lb)
        history.append({
            "iter": it,
            "phi": phi,
            "y_sum_sub": y_sum_sub,
            "g": g,
            "eta_lb": eta_lb,
            "UB": UB,
            "lam": lam,
            "gap": gap
        })
        print(f"[it {it:02d}] phi={phi:.6g}  sumy*={y_sum_sub:.3f}  g={g:.3f}  "
              f"eta_lb={eta_lb:.6g}  UB={UB:.6g}  lam={lam:.4g}  gap={gap:.6g}")

        if gap <= cfg.tol_gap:
            print("Converged by gap tolerance.")
            break

    return {
        "best_p": best_p,
        "best_obj": UB,
        "lower_bound": eta_lb,
        "gap": UB - eta_lb,
        "lambda": lam,
        "history": history,
    }


# ---------------------------------
# Minimal runnable example
# ---------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    n = 30
    # Build SPD S
    A = np.random.randn(n, n)
    S = A.T @ A + 1e-1 * np.eye(n)
    f = np.random.randn(n) * 2.0  # linear term
    p0 = np.maximum(0.0, np.random.randn(n) * 2.0 + 10.0)
    delta = np.full(n, 0.5)
    k = 6
    l = np.zeros(n)
    u = p0 + 5.0  # example upper bounds

    data = PriceOptData(S=S, f=f, p0=p0, delta=delta, k=k, l=l, u=u)
    cfg = LCPConfig(max_iters=40, tol_gap=1e-4, lam0=0.0, beta_polyak=1.0,
                    sub_time_limit=5.0, master_time_limit=5.0, sub_exact_every=1)

    result = lagrangian_cutting_plane(data, cfg)
    print("\n=== Result ===")
    print(f"best_obj (UB)  = {result['best_obj']:.6f}")
    print(f"lower_bound    = {result['lower_bound']:.6f}")
    print(f"final_gap      = {result['gap']:.6g}")
    print(f"lambda*        = {result['lambda']:.6g}")
    print("best_p (first 10):", np.round(result["best_p"][:10], 4))
