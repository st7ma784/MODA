"""
Dynamical Bayesian coupling function estimation (MODA bayes_main + CFprint).

Infers time-varying coupling functions q21(φ1,φ2) and q12(φ1,φ2) from
phase time-series using a sliding-window Fourier-series model:

    dφ1/dt = ω1 + q21(φ1, φ2)     q21 = Σ_k c1_k · basis_k(φ1, φ2)
    dφ2/dt = ω2 + q12(φ1, φ2)

The Fourier basis matches MODA's ``calculateP`` ordering exactly:
  constant, sin/cos(i·φ1), sin/cos(i·φ2), sin/cos(i·φ1 ± j·φ2)  i,j = 1…bn

All operations vectorised — no per-window or per-basis-function loops.
Batched least squares via np.linalg.solve / torch.linalg.solve.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _dev(device=None):
    if not _TORCH:
        return None
    return device or (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("cpu"))


# ── Fourier basis (MODA calculateP ordering) ──────────────────────────────────

def _build_basis(phi1: np.ndarray, phi2: np.ndarray, bn: int) -> np.ndarray:
    """
    Build Fourier basis matrix P matching MODA's ``calculateP`` ordering.

    Basis terms (K = 1 + 4·bn + 4·bn²):
      1  constant
      sin(i·φ1), cos(i·φ1)          for i = 1…bn   (2·bn terms)
      sin(i·φ2), cos(i·φ2)          for i = 1…bn   (2·bn terms)
      sin(i·φ1 + j·φ2), cos(…)      for i,j = 1…bn (2·bn² terms)
      sin(i·φ1 − j·φ2), cos(…)      for i,j = 1…bn (2·bn² terms)

    Fully vectorised via broadcasting — no per-basis loop.

    Parameters
    ----------
    phi1, phi2 : (T,) phase arrays (unwrapped)
    bn         : Fourier basis order

    Returns
    -------
    P : (T, K) float32
    """
    T = len(phi1)
    i_range = np.arange(1, bn + 1, dtype=np.float32)      # (bn,)

    # Single-variable harmonics — broadcasting: (T,1)*(1,bn) → (T,bn)
    phi1_h = phi1[:, np.newaxis] * i_range[np.newaxis, :]
    phi2_h = phi2[:, np.newaxis] * i_range[np.newaxis, :]

    # Cross harmonics: (T,bn,1) ± (T,1,bn) → (T,bn,bn) → (T,bn²)
    cross_p = (phi1_h[:, :, np.newaxis]
               + phi2_h[:, np.newaxis, :]).reshape(T, bn * bn)
    cross_m = (phi1_h[:, :, np.newaxis]
               - phi2_h[:, np.newaxis, :]).reshape(T, bn * bn)

    P = np.concatenate([
        np.ones((T, 1), dtype=np.float32),
        np.sin(phi1_h), np.cos(phi1_h),
        np.sin(phi2_h), np.cos(phi2_h),
        np.sin(cross_p), np.cos(cross_p),
        np.sin(cross_m), np.cos(cross_m),
    ], axis=1)                                              # (T, K)
    return P


def _eval_coupling_grid(c: np.ndarray, bn: int, n_grid: int = 50) -> np.ndarray:
    """
    Evaluate coupling function q(φ1, φ2) = P(grid) · c on a phase grid.

    Vectorised: build P for all n_grid² phase combinations at once.

    Returns
    -------
    q : (n_grid, n_grid) float32
    """
    phi_g  = np.linspace(0, 2 * np.pi, n_grid, dtype=np.float32)
    pg1, pg2 = np.meshgrid(phi_g, phi_g)
    P_flat = _build_basis(pg1.flatten(), pg2.flatten(), bn)  # (n_grid², K)
    return (P_flat @ c).reshape(n_grid, n_grid)


# ── main estimation ───────────────────────────────────────────────────────────

def sync_map(
    c1_mean: np.ndarray,
    c2_mean: np.ndarray,
    bn: int,
    n_grid: int = 500,
) -> dict:
    """
    Detect 1:1 synchronisation via the phase-coupling map (MODA sync_map.m).

    Strategy: evaluate the effective phase-difference velocity
        dψ/dt = F1(ψ/2, ψ/2) − F2(ψ/2, ψ/2)
    on a dense grid of phase differences ψ ∈ [0, 2π].  Stable zero-crossings
    (where dψ/dt passes from + to −) indicate synchronised states.

    This is a vectorised approximation of the Newton-Raphson root-finding
    algorithm in sync_map.m, valid for 1:1 synchronisation detection.

    Fully vectorised: all grid evaluations in one matmul.

    Returns
    -------
    dict with:
        sync_index           : 1 = synchronised, 0 = not
        n_stable_fixed_points: number of stable fixed points detected
        fixed_points_rad     : list of ψ values at stable FPs (radians)
        phi_diff_grid        : phase difference grid
        coupling_profile     : dψ/dt evaluated on grid
    """
    K   = len(c1_mean)
    psi = np.linspace(0, 2 * np.pi, n_grid, dtype=np.float32)  # (n_grid,)

    # Evaluate along diagonal θ1 = θ2 = ψ/2 — one matmul
    P       = _build_basis(psi / 2, psi / 2, bn)               # (n_grid, K) vectorised
    dth1    = P @ c1_mean                                       # (n_grid,)
    dth2    = P @ c2_mean
    dpsi    = dth1 - dth2                                       # (n_grid,) net phase velocity

    # Zero crossings: sign changes — vectorised
    signs    = np.sign(dpsi)
    crossings = np.where(np.diff(signs) != 0)[0]               # indices before crossing

    # Stability: + → − (stable), − → + (unstable)
    grads    = np.diff(dpsi)
    stable   = crossings[grads[crossings] < 0]

    fps      = float(np.mean(psi[stable])) if len(stable) > 0 else float('nan')

    return {
        "sync_index":             int(len(stable) > 0),
        "is_synchronised":        bool(len(stable) > 0),
        "n_stable_fixed_points":  int(len(stable)),
        "fixed_points_rad":       psi[stable].tolist(),
        "mean_fixed_point_deg":   round(float(np.degrees(fps)), 2) if not np.isnan(fps) else None,
        "phi_diff_grid":          psi[::max(1, n_grid // 200)].tolist(),
        "coupling_profile":       dpsi[::max(1, n_grid // 200)].tolist(),
    }


def estimate_coupling_functions(
    ph1: np.ndarray,
    ph2: np.ndarray,
    fs: float,
    bn: int = 3,
    win_s: float = 40.0,
    overlap: float = 0.5,
    n_grid: int = 50,
    device=None,
) -> Dict:
    """
    Estimate coupling functions via sliding-window batched OLS.

    Implements the core of MODA ``bayes_main`` + ``CFprint`` — using ordinary
    least squares (no prior propagation) for efficiency and GPU compatibility.

    All matrix operations are batched:
    • Basis matrices for all windows built in one broadcast pass.
    • Normal equations solved with a single batched ``linalg.solve`` call.
    • Coupling function reconstruction on a phase grid uses a single matmul.

    Parameters
    ----------
    ph1, ph2    : instantaneous phase time series (radians, will be unwrapped)
    fs          : sampling frequency (Hz)
    bn          : Fourier basis order (MODA default 2; higher = finer coupling)
    win_s       : window duration (seconds)
    overlap     : fractional overlap between windows (0 to <1)
    n_grid      : resolution of coupling function phase grid
    device      : torch device or None (auto)

    Returns
    -------
    dict with:
        q21         : (n_grid, n_grid)  mean coupling function 2→1
        q12         : (n_grid, n_grid)  mean coupling function 1→2
        q21_time    : (n_wins, n_grid, n_grid)  time-varying q21
        q12_time    : (n_wins, n_grid, n_grid)  time-varying q12
        cpl1        : (n_wins,)  coupling strength 2→1
        cpl2        : (n_wins,)  coupling strength 1→2
        direction   : (n_wins,)  directionality ∈ [−1, 1]
        times       : (n_wins,)  window-centre time (s)
        phi_grid    : (n_grid,)  phase grid values
        bn, gpu_used
    """
    N     = len(ph1)
    W     = min(int(win_s * fs), N)
    hop   = max(1, int(W * (1 - overlap)))
    n_wins = max(1, (N - W) // hop + 1)

    # Unwrap phases
    p1 = np.unwrap(ph1.astype(np.float64)).astype(np.float32)
    p2 = np.unwrap(ph2.astype(np.float64)).astype(np.float32)

    dev = _dev(device)

    # ── Build all window indices at once — no loop ──────────────────────
    starts   = np.clip(np.arange(n_wins) * hop, 0, N - W)            # (n_wins,)
    win_idx  = starts[:, np.newaxis] + np.arange(W)                   # (n_wins, W)

    p1_win   = p1[win_idx]                                             # (n_wins, W)
    p2_win   = p2[win_idx]

    # Midpoint phases and phase velocities
    p1_mid   = 0.5 * (p1_win[:, :-1] + p1_win[:, 1:])               # (n_wins, W-1)
    p2_mid   = 0.5 * (p2_win[:, :-1] + p2_win[:, 1:])
    dp1      = np.diff(p1_win, axis=1) * fs                           # (n_wins, W-1)
    dp2      = np.diff(p2_win, axis=1) * fs

    L        = W - 1                                                   # samples per window
    K        = 1 + 4 * bn + 4 * bn * bn                              # basis size

    # ── Basis matrices for all windows — vectorised ─────────────────────
    # Flatten all windows, build P, reshape
    P_flat   = _build_basis(p1_mid.reshape(-1), p2_mid.reshape(-1), bn)  # (n_wins*L, K)
    P_all    = P_flat.reshape(n_wins, L, K)                            # (n_wins, L, K)

    if dev is not None and _TORCH:
        P_t   = torch.as_tensor(P_all, dtype=torch.float32, device=dev)
        dp1_t = torch.as_tensor(dp1,   dtype=torch.float32, device=dev)
        dp2_t = torch.as_tensor(dp2,   dtype=torch.float32, device=dev)

        # Normal equations — batch einsum  (n_wins, K, K) and (n_wins, K)
        PtP   = torch.einsum("nwk,nwl->nkl", P_t, P_t)
        Pt1   = torch.einsum("nwk,nw->nk",   P_t, dp1_t)
        Pt2   = torch.einsum("nwk,nw->nk",   P_t, dp2_t)

        reg   = 1e-8 * torch.eye(K, device=dev).unsqueeze(0)
        c1_t  = torch.linalg.solve(PtP + reg, Pt1.unsqueeze(-1)).squeeze(-1)
        c2_t  = torch.linalg.solve(PtP + reg, Pt2.unsqueeze(-1)).squeeze(-1)

        c1    = c1_t.cpu().numpy()                                    # (n_wins, K)
        c2    = c2_t.cpu().numpy()
        gpu_used = True
    else:
        # CPU numpy batched solve
        PtP   = np.einsum("nwk,nwl->nkl", P_all, P_all)              # (n_wins, K, K)
        Pt1   = np.einsum("nwk,nw->nk",   P_all, dp1)
        Pt2   = np.einsum("nwk,nw->nk",   P_all, dp2)
        reg   = 1e-8 * np.eye(K)[np.newaxis, :, :]
        c1    = np.linalg.solve(PtP + reg, Pt1[:, :, np.newaxis]).squeeze(-1)
        c2    = np.linalg.solve(PtP + reg, Pt2[:, :, np.newaxis]).squeeze(-1)
        gpu_used = False

    # ── Coupling function on grid — vectorised ──────────────────────────
    phi_g  = np.linspace(0, 2 * np.pi, n_grid, dtype=np.float32)
    pg1, pg2 = np.meshgrid(phi_g, phi_g)
    P_grid   = _build_basis(pg1.flatten(), pg2.flatten(), bn)         # (n_grid², K)

    # c1: (n_wins, K) → q21_time: (n_wins, n_grid²) → reshape
    q21_flat = c1 @ P_grid.T                                          # (n_wins, n_grid²)
    q12_flat = c2 @ P_grid.T
    q21_time = q21_flat.reshape(n_wins, n_grid, n_grid)
    q12_time = q12_flat.reshape(n_wins, n_grid, n_grid)

    q21_mean = q21_time.mean(axis=0)
    q12_mean = q12_time.mean(axis=0)

    # ── Coupling direction (dirc equivalent) ────────────────────────────
    # Norm of coupling coefficients as proxy for strength
    cpl1      = np.linalg.norm(c1, axis=1)                            # (n_wins,)
    cpl2      = np.linalg.norm(c2, axis=1)
    direction = (cpl2 - cpl1) / (cpl2 + cpl1 + 1e-12)               # ∈ [−1, 1]

    # Standard errors across windows (proxy for uncertainty; zero when n_wins=1)
    cpl1_se  = float(np.std(cpl1))   if n_wins > 1 else 0.0
    cpl2_se  = float(np.std(cpl2))   if n_wins > 1 else 0.0
    dir_se   = float(np.std(direction)) if n_wins > 1 else 0.0

    times   = (starts + W // 2) / fs

    return {
        "q21":       q21_mean.astype(np.float32),
        "q12":       q12_mean.astype(np.float32),
        "q21_time":  q21_time.astype(np.float32),
        "q12_time":  q12_time.astype(np.float32),
        "cpl1":      cpl1.astype(np.float32),
        "cpl2":      cpl2.astype(np.float32),
        "direction": direction.astype(np.float32),
        "cpl1_se":   np.float32(cpl1_se),
        "cpl2_se":   np.float32(cpl2_se),
        "dir_se":    np.float32(dir_se),
        "times":     times.astype(np.float32),
        "phi_grid":  phi_g,
        "c1":        c1.astype(np.float32),   # (n_wins, K) — used by sync_map
        "c2":        c2.astype(np.float32),
        "c1_mean":   c1.mean(axis=0).astype(np.float32),  # (K,)
        "c2_mean":   c2.mean(axis=0).astype(np.float32),
        "bn":        bn,
        "gpu_used":  gpu_used,
    }
