# pip install gurobipy
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_price_optimization_gurobi(
    *,
    p0: np.ndarray,         # baseline price vector (n,)
    delta: np.ndarray,      # minimum absolute change per item (n,), delta_i > 0
    k: int,                 # max number of price changes
    c: np.ndarray,          # unit cost vector (n,)
    a: np.ndarray = None,   # demand intercept (n,)
    D: np.ndarray = None,   # demand slope matrix (n,n), v(p) = a - D p
    S: np.ndarray = None,   # optional: S = D + D.T (symmetric part). If provided, D can be None.
    l: np.ndarray = None,   # optional lower bounds per item (n,)
    u: np.ndarray = None,   # optional upper bounds per item (n,)
    M: float = 1e6,         # Big-M used only when l,u are None（式(36)）
    time_limit: float = 3600,
    mip_gap: float = None,
    threads: int = 1,
    verbose: bool = True,
):
    """
    Gurobi MIQP for:
      maximize  -0.5 p^T S p + (a + D^T c)^T p
      subject to per-item 3-way choice (stay / raise / lower) with minimum jump delta,
                 at most k items can change (|{i: p_i != p0_i}| <= k).

    If l,u are provided (with l_i <= p0_i - delta_i and u_i >= p0_i + delta_i), uses the bounded formulation (式(37)).
    Otherwise uses Big-M formulation (式(36)).

    Returns dict with solution vector p, binaries zP,zR,zL and model object.
    """
    # ---- basic checks / shapes ----
    p0   = np.asarray(p0, dtype=float)
    delta= np.asarray(delta, dtype=float)
    c    = np.asarray(c, dtype=float)
    n    = p0.size
    assert delta.shape == (n,)
    assert c.shape == (n,)
    assert np.all(delta > 0), "delta_i は正にしてください（論文の仮定(A4)）。"

    # build S and f = (a + D^T c)
    if S is None:
        assert (a is not None) and (D is not None), "S を渡さない場合は a, D が必要です。"
        a = np.asarray(a, dtype=float)
        D = np.asarray(D, dtype=float)
        assert a.shape == (n,) and D.shape == (n, n)
        S = D + D.T
    else:
        S = np.asarray(S, dtype=float)
        assert S.shape == (n, n)

    if a is None or D is None:
        # a or D 未提供でも f は必要。D が無い場合は f を別途与える設計にしていないので、a と D の両方必要。
        # Sのみ渡されたケースでも a と D は論文の f = a + D^T c のため必要です。
        # ただし「f を直接渡したい」場合はここを書き換えてください。
        assert (a is not None) and (D is not None), "f = a + D^T c のため a, D を渡してください。"
    f = a + D.T @ c  # (n,)

    # optional bounds
    use_bounds = (l is not None) and (u is not None)
    if use_bounds:
        l = np.asarray(l, dtype=float); u = np.asarray(u, dtype=float)
        assert l.shape == (n,) and u.shape == (n,)
        # 論文の式(37)で仮定している境界関係を軽くチェック
        if not np.all(l <= p0 - delta + 1e-12):
            raise ValueError("各 i について l_i <= p0_i - delta_i を満たす必要があります（式(37)）。")
        if not np.all(u >= p0 + delta - 1e-12):
            raise ValueError("各 i について u_i >= p0_i + delta_i を満たす必要があります（式(37)）。")

    # ---- build model ----
    m = gp.Model("PriceOpt_MIQP")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    if threads is not None:
        m.Params.Threads = threads

    # variables
    if use_bounds:
        p = m.addVars(n, lb=l.tolist(), ub=u.tolist(), vtype=GRB.CONTINUOUS, name="p")
    else:
        p = m.addVars(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p")

    zP = m.addVars(n, vtype=GRB.BINARY, name="zP")  # stay at p0
    zR = m.addVars(n, vtype=GRB.BINARY, name="zR")  # >= p0 + delta
    zL = m.addVars(n, vtype=GRB.BINARY, name="zL")  # <= p0 - delta

    # exactly one of {stay, raise, lower}
    m.addConstrs((zP[i] + zR[i] + zL[i] == 1 for i in range(n)), name="one_of_three")

    # at most k changes (raise or lower)
    m.addConstr(gp.quicksum(zR[i] + zL[i] for i in range(n)) <= k, name="change_budget")

    # link price & regime
    if use_bounds:
        # --- bounded formulation: 式(37) ---
        # lower linking:  p_i >= p0_i*zP + l_i*zL + (p0_i + delta_i)*zR
        m.addConstrs(
            (p[i] >= p0[i]*zP[i] + l[i]*zL[i] + (p0[i]+delta[i])*zR[i] for i in range(n)),
            name="link_lb_bounded"
        )
        # upper linking:  p_i <= p0_i*zP + (p0_i - delta_i)*zL + u_i*zR
        m.addConstrs(
            (p[i] <= p0[i]*zP[i] + (p0[i]-delta[i])*zL[i] + u[i]*zR[i] for i in range(n)),
            name="link_ub_bounded"
        )
    else:
        # --- Big-M formulation: 式(36) ---
        # lower linking:  p_i >= p0_i*zP - M*zL + (p0_i + delta_i)*zR
        m.addConstrs(
            (p[i] >= p0[i]*zP[i] - M*zL[i] + (p0[i]+delta[i])*zR[i] for i in range(n)),
            name="link_lb_bigM"
        )
        # upper linking:  p_i <= p0_i*zP + (p0_i - delta_i)*zL + M*zR
        m.addConstrs(
            (p[i] <= p0[i]*zP[i] + (p0[i]-delta[i])*zL[i] + M*zR[i] for i in range(n)),
            name="link_ub_bigM"
        )

    # ---- objective: maximize -0.5 p^T S p + f^T p ----
    quad = gp.QuadExpr()

    # diagonal terms
    for i in range(n):
        if S[i, i] != 0.0:
            quad += -0.5 * S[i, i] * p[i] * p[i]
    # off-diagonal (i<j) terms（S は対称を想定）
    for i in range(n):
        for j in range(i+1, n):
            sij = S[i, j]
            if sij != 0.0:
                quad += -1.0 * sij * p[i] * p[j]

    lin = gp.LinExpr(gp.quicksum(float(f[i]) * p[i] for i in range(n)))
    m.setObjective(quad + lin, GRB.MAXIMIZE)

    # ---- solve ----
    m.optimize()

    # ---- extract ----
    status = m.Status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi ended with status={status}")

    p_val  = np.array([p[i].X for i in range(n)])
    zP_val = np.array([zP[i].X for i in range(n)], dtype=int)
    zR_val = np.array([zR[i].X for i in range(n)], dtype=int)
    zL_val = np.array([zL[i].X for i in range(n)], dtype=int)

    return {
        "p": p_val,
        "zP": zP_val,
        "zR": zR_val,
        "zL": zL_val,
        "obj": m.ObjVal if m.SolCount > 0 else None,
        "model": m,
    }

