"""Replace WECC GENCLS units with REGCA1(+REECA1) by capacity penetration.

Penetration is capacity-based: sum(Sn_replaced) / sum(Sn_all_GENCLS).
Online GENCLS ``M`` (=2H) sums are recorded separately as inertia evidence.

Scope: public ANDES WECC 179-bus GENCLS case only — not a real regional grid.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def inertia_sum(ss: Any) -> float:
    """Sum of ``M`` for online GENCLS units (``u > 0``)."""
    if not hasattr(ss, "GENCLS") or len(ss.GENCLS) == 0:
        return 0.0
    m = np.asarray(ss.GENCLS.M.v, dtype=float)
    u = np.asarray(ss.GENCLS.u.v, dtype=float)
    return float(np.nansum(m[u > 0]))


def total_gencls_m(ss: Any) -> float:
    """Sum of all GENCLS ``M`` parameters (including u=0 placeholders)."""
    if not hasattr(ss, "GENCLS") or len(ss.GENCLS) == 0:
        return 0.0
    return float(np.nansum(np.asarray(ss.GENCLS.M.v, dtype=float)))


def plan_replacements(ss: Any, target_penetration: float) -> list[int]:
    """Deterministic Sn-descending indices to reach capacity penetration."""
    if target_penetration < 0:
        raise ValueError("target_penetration must be >= 0")
    if target_penetration == 0:
        return []
    sn = np.asarray(ss.GENCLS.Sn.v, dtype=float)
    if sn.size == 0:
        raise RuntimeError("system has no GENCLS units to replace")
    total = float(np.nansum(sn))
    if total <= 0:
        raise RuntimeError("GENCLS Sn sum is non-positive")
    order = np.argsort(-sn)
    chosen: list[int] = []
    cum = 0.0
    for i in order:
        if cum / total >= target_penetration:
            break
        chosen.append(int(i))
        cum += float(sn[int(i)])
    return chosen


def replace_sg_with_ibr(
    ss: Any,
    target_penetration: float,
    *,
    ibr_model: str = "REGCA1",
    tg_s: float = 0.02,
    volim: float = 1.5,
) -> dict[str, Any]:
    """Disable GENCLS by Sn rank and attach REGCA1+REECA1 on the same static gen.

    Must be called after ``andes.load(..., setup=False)`` and before ``ss.setup()``.
    ``Sn`` for planning is valid pre-setup; ``M`` inertia figures are filled by the
    caller after setup via :func:`inertia_sum` / :func:`total_gencls_m`.

    REGCA1 uses ``Lvplsw=0`` and a modest ``Tg`` so WECC TDS can initialize; REECA1
    control flags start at 0 (Q control defaults), matching ieee14_solar practice.
    """
    if ibr_model != "REGCA1":
        raise ValueError(f"unsupported ibr_model: {ibr_model}")

    sn = np.asarray(ss.GENCLS.Sn.v, dtype=float)
    total_sn = float(np.nansum(sn))
    chosen = plan_replacements(ss, target_penetration)

    replaced_idx: list[str] = []
    replaced_sn = 0.0
    reg_indices: list[Any] = []
    for i in chosen:
        bus = ss.GENCLS.bus.v[i]
        gen = ss.GENCLS.gen.v[i]
        unit_sn = float(sn[i])
        unit_idx = str(ss.GENCLS.idx.v[i])
        ss.GENCLS.u.v[i] = 0.0
        reg_idx = ss.add(
            "REGCA1",
            bus=bus,
            gen=gen,
            Sn=unit_sn,
            gammap=1.0,
            gammaq=1.0,
            Lvplsw=0.0,
            Volim=float(volim),
            Tg=float(tg_s),
        )
        ss.add(
            "REECA1",
            reg=reg_idx,
            PFFLAG=0,
            VFLAG=0,
            QFLAG=0,
            PFLAG=0,
            PQFLAG=0,
        )
        replaced_idx.append(unit_idx)
        replaced_sn += unit_sn
        reg_indices.append(reg_idx)

    achieved = (replaced_sn / total_sn) if total_sn > 0 else 0.0
    return {
        "replaced_count": len(replaced_idx),
        "achieved_penetration": float(achieved),
        "target_penetration": float(target_penetration),
        "replaced_idx": replaced_idx,
        "replaced_Sn_sum": float(replaced_sn),
        "total_Sn": float(total_sn),
        "reg_indices": [str(x) for x in reg_indices],
        "ibr_model": ibr_model,
        "tg_s": float(tg_s),
        "inertia_before": None,  # filled after setup
        "inertia_after": None,
    }


def finalize_inertia_meta(ss: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """Attach post-setup inertia totals to a replace_sg_with_ibr meta dict."""
    # Original inventory: all GENCLS M parameters (u=0 units keep their M).
    before = total_gencls_m(ss)
    after = inertia_sum(ss)
    meta = dict(meta)
    meta["inertia_before"] = float(before)
    meta["inertia_after"] = float(after)
    meta["inertia_reduction_frac"] = (
        float((before - after) / before) if before > 0 else 0.0
    )
    return meta


def mean_online_frequency_hz(ss: Any) -> float | None:
    """Nominal-scaled mean omega of online GENCLS at the latest TDS sample."""
    if not hasattr(ss, "GENCLS") or len(ss.GENCLS) == 0:
        return None
    u = np.asarray(ss.GENCLS.u.v, dtype=float)
    online = u > 0
    if not online.any():
        return None
    try:
        frame = ss.TDS.get_timeseries(ss.GENCLS.omega)
    except Exception:
        return None
    omega = np.asarray(frame.values, dtype=float)
    if omega.ndim == 1:
        omega = omega[:, None]
    if omega.shape[0] < 1:
        return None
    last = omega[-1, online]
    last = last[np.isfinite(last)]
    if last.size == 0:
        return None
    freq = float(getattr(ss.config, "freq", 60.0))
    return float(np.mean(last) * freq)
