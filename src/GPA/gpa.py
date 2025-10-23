# gpa.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np


def _largest_eigval_symmetric(
    S: np.ndarray, use_power: bool = True, power_iters: int = 200, tol: float = 1e-8, seed: int = 42
) -> float:
    """
    最大固有値 λ_max を返す（S は対称, PSD）。
    n が大きいときはパワーイテレーションを既定で使用。
    """
    n = S.shape[0]
    if not use_power or n <= 2000:
        # 厳密（小規模）
        return float(np.linalg.eigvalsh(S)[-1])

    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n,))
    v /= np.linalg.norm(v) + 1e-12
    lam_old = 0.0
    for _ in range(power_iters):
        w = S @ v
        normw = np.linalg.norm(w)
        if normw == 0:
            return 0.0
        v = w / normw
        lam = float(v @ (S @ v))
        if abs(lam - lam_old) <= tol * max(1.0, abs(lam_old)):
            break
        lam_old = lam
    return lam_old


def _proj_1d_unbounded(qi: float, p0i: float, di: float) -> float:
    """
    1次元集合 P_i^{δ_i} = {p0} ∪ [p0+δ, ∞) ∪ (-∞, p0-δ] への射影。
    （論文の場合分け規則に対応）
    """
    if di <= 0:
        # δ_i=0 は {p0} ∪ R（実質 R ）なので qi を返すのが最適
        return qi

    if qi >= p0i + di:
        return qi
    if qi <= p0i - di:
        return qi

    # 中央のギャップ領域は最近接端点 or p0
    half = di / 2.0
    if p0i + half < qi < p0i + di:
        return p0i + di
    if p0i <= qi <= p0i + half:
        return p0i
    if p0i - di <= qi <= p0i - half:
        return p0i - di
    if p0i - half <= qi <= p0i:
        return p0i

    # 理論上ここには来ない
    return p0i


def _proj_1d_bounded(qi: float, p0i: float, di: float, li: float, ui: float) -> float:
    """
    1次元集合 P_i^{δ_i,l_i,u_i} = {p0} ∪ [p0+δ, ui] ∪ [li, p0-δ] への射影。
    （論文 Remark 11 に対応）
    """
    # まず区間外ならクリップ（これ自体が射影）
    if qi < li:
        return li
    if qi > ui:
        return ui

    if di <= 0:
        # δ_i=0 のときは [li, ui] 全域が可行：単純にクリップされた qi（すでに [li,ui]）
        return qi

    half = di / 2.0
    # 右側の許容区間 [p0+δ, ui]
    if qi >= p0i + di:
        return qi
    # 左側の許容区間 [li, p0-δ]
    if qi <= p0i - di:
        return qi

    # ギャップ領域 [p0-δ, p0+δ] の場合分け
    # 右ギャップ
    if p0i + half < qi < p0i + di:
        return min(p0i + di, ui)
    # 中央（p0）に最近接なら p0（p0 が [li,ui] にある前提）
    if p0i - half <= qi <= p0i + half:
        # 念のため p0 が [li,ui] 外なら最近接端点へ
        if p0i < li:
            return li
        if p0i > ui:
            return ui
        return p0i
    # 左ギャップ
    if p0i - di < qi < p0i - half:
        return max(p0i - di, li)

    # 理論上ここには来ない
    return np.clip(qi, li, ui)


def _dist2_to_Pi_unbounded(qi: float, p0i: float, di: float) -> float:
    """d(qi, P_i)^2 for unbounded case."""
    if di <= 0:
        return 0.0
    if qi >= p0i + di or qi <= p0i - di:
        return 0.0
    half = di / 2.0
    if p0i + half <= qi <= p0i + di:
        return (p0i + di - qi) ** 2
    if p0i <= qi <= p0i + half:
        return (qi - p0i) ** 2
    if p0i - half <= qi <= p0i:
        return (p0i - qi) ** 2
    if p0i - di <= qi <= p0i - half:
        return (qi - (p0i - di)) ** 2
    return 0.0


def _dist2_to_Pi_bounded(qi: float, p0i: float, di: float, li: float, ui: float) -> float:
    """d(qi, P_i)^2 for bounded case."""
    # 区間外なら距離は単純なクリップ距離
    if qi < li:
        return (li - qi) ** 2
    if qi > ui:
        return (qi - ui) ** 2

    if di <= 0:
        return 0.0

    # 中央ギャップ：端点 or p0 までの距離（二乗）
    half = di / 2.0
    if qi >= p0i + di or qi <= p0i - di:
        return 0.0
    # 右ギャップ
    if p0i + half <= qi <= p0i + di:
        return (min(p0i + di, ui) - qi) ** 2
    # 中央（p0）
    if p0i - half <= qi <= p0i + half:
        # p0 が [li,ui] 外の可能性を考慮
        t = np.clip(p0i, li, ui)
        return (qi - t) ** 2
    # 左ギャップ
    if p0i - di <= qi <= p0i - half:
        return (qi - max(p0i - di, li)) ** 2
    return 0.0


def _project_onto_omega(
    q: np.ndarray,
    p0: np.ndarray,
    delta: np.ndarray,
    k: int,
    l: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Ω_{k,p0,δ,(l,u)} へのユークリッド射影。
    命題：Δ_i = (p0_i - q_i)^2 - d(q_i, P_i)^2 を計算し、Δ が大きい上位 k のインデックスだけ 1次元射影、残りは p0 に据え置き。
    """
    n = q.size
    p = p0.copy()

    bounded = l is not None and u is not None
    if bounded:
        # 安全のため寸法チェック
        assert l.shape == p0.shape and u.shape == p0.shape

    # Δ と各 i の 1次元射影点を同時に用意
    deltas = np.empty(n, dtype=float)
    proj_vals = np.empty(n, dtype=float)

    for i in range(n):
        qi = float(q[i]); p0i = float(p0[i]); di = float(delta[i])
        if bounded:
            li = float(l[i]); ui = float(u[i])
            d2 = _dist2_to_Pi_bounded(qi, p0i, di, li, ui)
            pi = _proj_1d_bounded(qi, p0i, di, li, ui)
        else:
            d2 = _dist2_to_Pi_unbounded(qi, p0i, di)
            pi = _proj_1d_unbounded(qi, p0i, di)
        deltas[i] = (p0i - qi) ** 2 - d2
        proj_vals[i] = pi

    # Δ が正のものだけ候補に（Δ<=0 は変更益なし）
    idx_pos = np.where(deltas > 0)[0]
    if idx_pos.size == 0:
        # 何も動かさない
        return p

    # 上位 k を選択
    take = idx_pos[np.argsort(-deltas[idx_pos])[: min(k, idx_pos.size)]]

    # 選ばれた成分のみ 1次元射影値、他は p0
    p[take] = proj_vals[take]
    return p


def gpa_optimize(
    a: np.ndarray,
    D: np.ndarray,
    c: np.ndarray,
    p0: np.ndarray,
    delta: np.ndarray,
    k: int,
    L: Optional[float] = None,
    l: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    init: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    compute_lambda_by_power: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Gradient Projection Algorithm (GPA)
    -------------------------
    Minimize Q(p) = 1/2 p^T S p - (a + D^T c)^T p over Ω_{k,p0,δ,(l,u)}.
    Returns:
        p_star, info
    """
    # 入力を numpy 配列へ
    a = np.asarray(a, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float).reshape(-1)
    p0 = np.asarray(p0, dtype=float).reshape(-1)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    n = p0.size
    D = np.asarray(D, dtype=float).reshape(n, n)

    if l is not None:
        l = np.asarray(l, dtype=float).reshape(-1)
    if u is not None:
        u = np.asarray(u, dtype=float).reshape(-1)
    if (l is None) ^ (u is None):
        raise ValueError("Both l and u must be provided together, or neither.")

    # S と f の準備
    S = D + D.T
    f = a + D.T @ c  # 勾配が Sp - f

    # L 設定（L > λ_max(S)）
    if L is None:
        lam_max = _largest_eigval_symmetric(S, use_power=compute_lambda_by_power)
        L = 1.05 * max(lam_max, 1e-12)  # 少しだけ上乗せ
    elif L <= 0:
        raise ValueError("L must be positive.")

    # 初期点（||p - p0||_0 ≤ k を満たすもの。既定は p0）
    p = p0.copy() if init is None else np.asarray(init, dtype=float).reshape(-1)
    if p.shape != p0.shape:
        raise ValueError("init (if provided) must have same shape as p0.")
    # 念のため可行化（k 超過やギャップ内を消す）
    p = _project_onto_omega(p, p0, delta, k, l, u)

    def Q(pv: np.ndarray) -> float:
        return 0.5 * float(pv @ (S @ pv)) - float(f @ pv)

    def gradQ(pv: np.ndarray) -> np.ndarray:
        return S @ pv - f

    hist_Q = [Q(p)]
    converged = False
    iters = 0

    for t in range(1, max_iter + 1):
        g = gradQ(p)
        q = p - (1.0 / L) * g
        p_next = _project_onto_omega(q, p0, delta, k, l, u)
        Qp = hist_Q[-1]
        Qn = Q(p_next)

        # 収束判定（論文の停止条件）
        if Qp - Qn <= tol:
            converged = True
            p = p_next
            hist_Q.append(Qn)
            iters = t
            break

        p = p_next
        hist_Q.append(Qn)
        iters = t

        if verbose and (t % 100 == 0 or t == 1):
            print(f"[iter {t}] Q={Qn:.6e}, ΔQ={Qp - Qn:.3e}")

    # 期待利益（最大化）も返す
    def Z(pv: np.ndarray) -> float:
        return float((pv - c) @ (a - D @ pv))

    info = {
        "converged": converged,
        "iterations": iters,
        "Q": hist_Q[-1],
        "history_Q": np.array(hist_Q),
        "Z": Z(p),
        "L": L,
    }
    return p, info

