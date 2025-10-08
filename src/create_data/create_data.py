# -*- coding: utf-8 -*-
"""
Extension: (5.3) price bounds l,u and (5.1) 5 initial solutions for GPA

- Bounds (5.3):
    Scheme "absolute" (default):  l_i = max(0, p0_i - δ_i),  u_i = p0_i + δ_i
    Scheme "relative":            l_i = max(0, (1-δ_i) p0_i), u_i = (1+δ_i) p0_i
  （δ は 5.1 で与えた 1.0 または 0.5 をそのまま使用）

- GPA initial candidates (5.1 end; all box-feasible in [l,u]):
    1) p0                         : ベースライン
    2) u                          : 上限に張り付け
    3) l                          : 下限に張り付け
    4) proj_[l,u](S^{-1} f)       : 二次目的の無制約解の投影
    5) proj_[l,u](p0 + α (f − S p0)):
           α ≈ 1/L,  L ≈ λ_max(S) をパワー法で推定（1 ステップPGの良い初期解）
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError as e:
    raise ImportError("Please install scipy for sparse linear algebra.") from e


# ====== ここは前回の PriceOptDataset を拡張 ======
@dataclass
class PriceOptDataset:
    D: sp.csr_matrix        # (n x n)
    S: sp.csr_matrix        # (n x n), SPD
    a: np.ndarray           # (n,)
    c: np.ndarray           # (n,)
    p0: np.ndarray          # (n,)
    delta: np.ndarray       # (n,)
    k: int                  # <= 0.1 n
    f: np.ndarray           # a + D^T c
    # --- 追加 ---
    l: np.ndarray           # 下限 (n,)
    u: np.ndarray           # 上限 (n,)
    initials: Dict[str, np.ndarray]  # GPA初期解 5本
    lipschitz_L: float                  # ≈ λ_max(S)
    meta: Dict[str, Any]


# ====== ここは前回のヘルパ関数（生成/SPD化） ======
def _ensure_spd_via_diag_dominance(D: sp.csr_matrix, tau: float = 0.10):
    n = D.shape[0]
    S = (D + D.T).tocsr()
    absS = abs(S)
    diag = S.diagonal()
    offsum = np.array(absS.sum(axis=1)).ravel() - np.abs(diag)
    need = (1.0 + tau) * offsum - diag
    need = np.maximum(need, 0.0)
    if np.any(need > 0):
        bump = sp.diags(need * 0.5, offsets=0, shape=(n, n), format="csr")
        D = (D + bump).tocsr()
        S = (S + sp.diags(need, offsets=0, shape=(n, n), format="csr")).tocsr()
    return D, S


def _generate_D(
    n: int,
    max_offdiag: int = 5,
    diag_low: float = 1.0,
    diag_high: float = 10.0,
    neg_ratio: float = 0.2,
    rng: np.random.Generator = np.random.default_rng(0),
) -> sp.csr_matrix:
    rows, cols, data = [], [], []
    for i in range(n):
        d_ii = rng.uniform(diag_low, diag_high)
        rows.append(i); cols.append(i); data.append(d_ii)
        m = rng.integers(low=0, high=max_offdiag + 1)
        if m > 0:
            js = rng.choice([j for j in range(n) if j != i], size=m, replace=False)
            mags = rng.uniform(0.0, neg_ratio * d_ii, size=m)
            rows.extend([i] * m); cols.extend(js.tolist()); data.extend((-mags).tolist())
    D = sp.csr_matrix((np.array(data, float), (np.array(rows), np.array(cols))), shape=(n, n))
    return D


# ====== 追加: Bounds 生成と GPA 初期解 ======
def _make_bounds(
    p0: np.ndarray,
    delta: np.ndarray,
    scheme: Literal["absolute", "relative"] = "absolute",
    nonnegativity: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if scheme == "absolute":
        l = p0 - delta
        u = p0 + delta
    elif scheme == "relative":
        l = p0 * (1.0 - delta)
        u = p0 * (1.0 + delta)
    else:
        raise ValueError("scheme must be 'absolute' or 'relative'")

    if nonnegativity:
        l = np.maximum(0.0, l)

    # 安全対策: l <= u を保証（稀な数値誤差対処）
    mask = l > u
    if np.any(mask):
        mid = 0.5 * (l[mask] + u[mask])
        l[mask] = mid
        u[mask] = mid
    return l.astype(float), u.astype(float)


def _estimate_lambda_max_power(S: sp.csr_matrix, iters: int = 30, rng: Optional[np.random.Generator] = None) -> float:
    """パワー法で最大固有値を概算（疎行列対応）。"""
    n = S.shape[0]
    rng = rng or np.random.default_rng(0)
    x = rng.normal(size=n)
    x /= np.linalg.norm(x) + 1e-12
    lam = 0.0
    for _ in range(iters):
        y = S @ x
        ny = np.linalg.norm(y) + 1e-12
        x = y / ny
        lam = float(x @ (S @ x))  # Rayleigh quotient
    return max(lam, 1e-12)


def _solve_unconstrained_minimizer(S: sp.csr_matrix, f: np.ndarray, tol: float = 1e-6, maxiter: Optional[int] = None):
    """
    無制約二次最小化: min 0.5 p^T S p - f^T p
    勾配= S p - f = 0 -> S p = f を CG で解く（SPD前提）。
    """
    n = S.shape[0]
    if maxiter is None:
        # 超大規模向けにやや控えめ（必要なら外から調整）
        maxiter = min(1000, max(200, n // 50))
    p, info = spla.cg(S, f, atol=0.0, tol=tol, maxiter=maxiter)
    # info = 0: 収束, >0: 反復数, <0: breakdown
    return p if p is not None else np.zeros_like(f)


def _project_box(x: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, l), u)


def _build_gpa_initials(
    S: sp.csr_matrix, f: np.ndarray,
    p0: np.ndarray, l: np.ndarray, u: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    L_hint: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    GPA 用の初期解 5 本を返す（すべて box 可行）と、推定 L（≈ λ_max(S)）を返す。
    """
    rng = rng or np.random.default_rng(0)

    # 1) p0
    p_init1 = _project_box(p0, l, u)

    # 2) 上限
    p_init2 = u.copy()

    # 3) 下限
    p_init3 = l.copy()

    # 4) 無制約最適解の投影
    p_star = _solve_unconstrained_minimizer(S, f, tol=1e-6)
    p_init4 = _project_box(p_star, l, u)

    # 5) 1 ステップの射影 PG
    L = float(L_hint) if L_hint is not None else _estimate_lambda_max_power(S, iters=30, rng=rng)
    alpha = 1.0 / max(L, 1e-12)
    grad_p0 = S @ p_init1 - f
    p_init5 = _project_box(p_init1 - alpha * grad_p0, l, u)

    initials = {
        "init_1_p0": p_init1,
        "init_2_upper": p_init2,
        "init_3_lower": p_init3,
        "init_4_proj_unconstrained": p_init4,
        "init_5_one_step_pg": p_init5,
    }
    return initials, L


# ====== データセット生成（前回関数に組み込み） ======
def generate_dataset(
    n: int,
    delta_value: float = 1.0,                 # 5.1: {1.0, 0.5}
    seed: Optional[int] = 0,
    max_offdiag: int = 5,
    diag_range: Tuple[float, float] = (1.0, 10.0),
    p0_range: Tuple[float, float] = (1.0, 10.0),
    cost_range: Tuple[float, float] = (1.0, 10.0),
    r_range: Tuple[float, float] = (1.0, 10.0),
    tau_spd: float = 0.10,
    # --- 追加オプション ---
    bounds_scheme: Literal["absolute", "relative"] = "absolute",  # 5.3 の解釈（既定: 絶対幅）
    nonnegativity: bool = True,
) -> PriceOptDataset:
    rng = np.random.default_rng(seed)

    # 1) D 生成 -> SPD 化
    D = _generate_D(
        n=n,
        max_offdiag=max_offdiag,
        diag_low=diag_range[0],
        diag_high=diag_range[1],
        rng=rng,
    )
    D, S = _ensure_spd_via_diag_dominance(D, tau=tau_spd)

    # 2) ベクトル a, c, δ, p0
    p0 = rng.uniform(p0_range[0], p0_range[1], size=n)
    c  = rng.uniform(cost_range[0], cost_range[1], size=n)
    r  = rng.uniform(r_range[0], r_range[1], size=n)
    Dt_c = D.T @ c
    a  = -(r + Dt_c)
    delta = np.full(n, float(delta_value))
    k = int(np.floor(0.10 * n))
    f = a + Dt_c  # Q(p)=0.5 p^T S p - f^T p

    # 3) (5.3) 下限/上限の生成
    l, u = _make_bounds(p0, delta, scheme=bounds_scheme, nonnegativity=nonnegativity)

    # 4) (5.1) GPA 初期解 5 本
    initials, L = _build_gpa_initials(S, f, p0, l, u, rng=rng)

    return PriceOptDataset(
        D=D.tocsr(),
        S=S.tocsr(),
        a=a.astype(float),
        c=c.astype(float),
        p0=p0.astype(float),
        delta=delta.astype(float),
        k=k,
        f=f.astype(float),
        l=l, u=u,
        initials=initials,
        lipschitz_L=L,
        meta=dict(
            n=n,
            max_offdiag=max_offdiag,
            diag_range=diag_range,
            p0_range=p0_range,
            cost_range=cost_range,
            r_range=r_range,
            delta_value=delta_value,
            tau_spd=tau_spd,
            seed=seed,
            bounds_scheme=bounds_scheme,
            nonnegativity=nonnegativity,
        ),
    )

