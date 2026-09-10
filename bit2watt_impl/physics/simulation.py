"""Robust ANDES time-domain execution for public test systems."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dataset.attack_profiles import AttackProfile
from bit2watt_impl.physics.waveforms import build_event_schedule


@dataclass
class StepRecord:
    attempt: int
    target_s: float
    reached_s: float
    ok: bool
    busted: bool
    chunk_s: float
    exit_code: int | None = None
    message: str = ""


@dataclass
class SimulationResult:
    row: dict[str, Any]
    convergence: list[dict[str, Any]]


def _import_andes():
    try:
        import andes
    except ImportError as exc:
        raise RuntimeError(
            "ANDES is unavailable. Activate .venv-physics or run setup_physics_env.sh."
        ) from exc
    return andes


def _configure_constant_power(ss: Any) -> None:
    ss.config.warn_abnormal = 0
    ss.PQ.config.pq2z = 0
    ss.PQ.config.p2p, ss.PQ.config.p2i, ss.PQ.config.p2z = 1.0, 0.0, 0.0
    ss.TDS.config.no_tqdm = 1


def build_kundur_system() -> Any:
    andes = _import_andes()
    ss = andes.load(
        andes.get_case("kundur/kundur_ieeest.xlsx"),
        setup=False,
        no_output=True,
    )
    # This workbook contains a line-trip Toggle at t=2 s. It belongs to the
    # case's original disturbance study and must be disabled so the measured
    # response is attributable only to our PQ-load waveform.
    disabled_events: list[str] = []
    if len(ss.Toggle):
        ss.Toggle.u.v[:] = [0] * len(ss.Toggle)
        disabled_events.append("Toggle")
    ss.setup()
    ss._gridpulse_disabled_events = disabled_events
    _configure_constant_power(ss)
    if not ss.PFlow.run():
        raise RuntimeError("Kundur power flow did not converge")
    return ss


def build_wecc_system() -> Any:
    andes = _import_andes()
    raw = andes.get_case("wecc/wecc.raw")
    dyr = andes.get_case("wecc/wecc_gencls.dyr")
    ss = andes.load(raw, addfile=dyr, setup=False, no_output=True)
    _configure_constant_power(ss)
    ss.setup()
    if not ss.PFlow.run():
        raise RuntimeError("WECC 179 power flow did not converge")
    return ss


def current_time(ss: Any) -> float:
    return float(getattr(ss.dae, "t", 0.0) or 0.0)


def select_pq_idx(
    ss: Any,
    requested: Any | None = None,
    *,
    strategy: str = "first",
) -> Any:
    indices = list(ss.PQ.idx.v)
    if not indices:
        raise RuntimeError("test system has no PQ loads")
    if requested is not None:
        requested_text = str(requested)
        for idx in indices:
            if str(idx) == requested_text:
                return idx
        raise ValueError(f"unknown PQ idx: {requested}")
    if strategy == "first":
        return indices[0]
    if strategy == "largest_p0":
        powers = np.asarray(ss.PQ.p0.v, dtype=float)
        status = np.asarray(ss.PQ.u.v, dtype=float)
        eligible = np.where((status > 0) & np.isfinite(powers) & (powers > 0), powers, np.nan)
        if not np.isfinite(eligible).any():
            raise RuntimeError("test system has no online positive-P PQ load")
        return indices[int(np.nanargmax(eligible))]
    raise ValueError(f"unknown PQ selection strategy: {strategy}")


def run_to(
    ss: Any,
    target_s: float,
    *,
    max_chunk_s: float = 0.5,
    min_chunk_s: float = 0.01,
    max_retries: int = 3,
) -> tuple[bool, list[StepRecord], str]:
    """Advance to target; never retry a busted System in place."""
    records: list[StepRecord] = []
    chunk = max_chunk_s
    retries = 0
    tolerance = 1e-7

    while current_time(ss) < target_s - tolerance:
        before = current_time(ss)
        segment_target = min(before + chunk, target_s)
        ss.TDS.config.tf = segment_target
        try:
            returned = ss.TDS.run()
            ok = bool(returned)
            message = "" if ok else "TDS.run returned false"
        except Exception as exc:  # ANDES can raise on singular Jacobians
            ok = False
            message = f"{type(exc).__name__}: {exc}"
        reached = current_time(ss)
        busted = bool(getattr(ss.TDS, "busted", False))
        progressed = reached > before + tolerance
        reached_target = reached >= segment_target - tolerance
        records.append(
            StepRecord(
                attempt=len(records) + 1,
                target_s=float(segment_target),
                reached_s=float(reached),
                ok=bool(ok and reached_target),
                busted=busted,
                chunk_s=float(chunk),
                exit_code=(
                    int(ss.exit_code)
                    if getattr(ss, "exit_code", None) is not None
                    else None
                ),
                message=message,
            )
        )

        if busted:
            return False, records, "TDS entered busted state; fresh replay required"
        if ok and reached_target:
            retries = 0
            chunk = max_chunk_s
            continue

        retries += 1
        chunk /= 2.0
        if retries >= max_retries or chunk < min_chunk_s:
            reason = "TDS failed to reach target after adaptive retries"
            if not progressed:
                reason += " (no time progress)"
            return False, records, reason

    return True, records, ""


def _extract_omega(
    ss: Any,
    *,
    model_name: str,
    t_start_s: float,
    t_end_s: float,
    forcing_frequency_hz: float | None,
) -> dict[str, Any]:
    model = getattr(ss, model_name)
    frame = ss.TDS.get_timeseries(model.omega)
    times = np.asarray(frame.index, dtype=float)
    omega = np.asarray(frame.values, dtype=float)
    if omega.ndim == 1:
        omega = omega[:, None]
    mask = np.isfinite(times) & (times >= t_start_s) & (times <= t_end_s)
    times, omega = times[mask], omega[mask]
    if len(times) < 8:
        raise RuntimeError("insufficient omega samples in analysis interval")

    unique = np.concatenate(([True], np.diff(times) > 1e-10))
    times, omega = times[unique], omega[unique]
    if len(times) < 8:
        raise RuntimeError("insufficient unique omega samples")

    std_by_generator = np.nanstd(omega, axis=0)
    response_col = int(np.nanargmax(std_by_generator))
    selected = omega[:, response_col]
    nominal_hz = float(getattr(ss.config, "freq", 60.0))
    gradient = np.gradient(omega, times, axis=0)
    rocof = float(np.nanmax(np.abs(gradient)) * nominal_hz)

    dt = float(np.median(np.diff(times)))
    uniform_times = np.arange(times[0], times[-1], dt)
    uniform_signal = np.interp(uniform_times, times, selected)
    centered = uniform_signal - np.mean(uniform_signal)
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    spectrum = np.abs(np.fft.rfft(centered))
    dominant = float(freqs[1 + np.argmax(spectrum[1:])]) if len(freqs) > 2 else 0.0
    forcing_amplitude = None
    if forcing_frequency_hz is not None and len(freqs):
        forcing_bin = int(np.argmin(np.abs(freqs - forcing_frequency_hz)))
        forcing_amplitude = float(2.0 * spectrum[forcing_bin] / len(centered))
    generator_indices = list(model.idx.v)

    return {
        "rocof_hz_s": rocof,
        "osc_std": float(np.nanmax(std_by_generator)),
        "osc_ptp": float(np.nanmax(np.ptp(omega, axis=0))),
        "dominant_freq_hz": dominant,
        "forcing_response_amp": forcing_amplitude,
        "response_model": model_name,
        "response_device": str(generator_indices[response_col]),
        "omega_samples": int(len(times)),
    }


def _run_once(
    builder: Callable[[], Any],
    profile: AttackProfile,
    *,
    test_system: str,
    model_name: str,
    pq_idx: Any | None,
    max_chunk_s: float,
    criteria: int,
    pq_selection: str,
    tstep_s: float,
) -> SimulationResult:
    ss = builder()
    ss.TDS.config.criteria = criteria
    ss.TDS.config.tstep = tstep_s
    ss.TDS.config.shrinkt = 1
    ss.TDS.config.max_iter = max(30, int(ss.TDS.config.max_iter))
    selected_pq = select_pq_idx(ss, pq_idx, strategy=pq_selection)
    p0 = float(ss.PQ.get(src="p0", idx=selected_pq, attr="v"))
    pq_bus = str(ss.PQ.get(src="bus", idx=selected_pq, attr="v"))
    all_records: list[dict[str, Any]] = []

    for segment_index, event in enumerate(build_event_schedule(profile)):
        ok, records, reason = run_to(ss, event.time_s, max_chunk_s=max_chunk_s)
        for record in records:
            item = asdict(record)
            item.update({"segment_index": segment_index, "attack_id": profile.attack_id})
            all_records.append(item)
        if not ok:
            return SimulationResult(
                row={
                    **profile.to_dict(),
                    "test_system": test_system,
                    "case_version": "ANDES-2.0.0",
                    "pq_idx": str(selected_pq),
                    "pq_bus": pq_bus,
                    "pq_base_pu": p0,
                    "pq_selection": pq_selection,
                    "criteria": criteria,
                    "tstep_s": tstep_s,
                    "disabled_builtin_events": ",".join(
                        getattr(ss, "_gridpulse_disabled_events", [])
                    ),
                    "converged": False,
                    "failure_reason": reason,
                    "t_reached_s": current_time(ss),
                },
                convergence=all_records,
            )
        # p0(attr='v') and Ppf are both system-base values.
        ss.PQ.set(src="Ppf", idx=selected_pq, value=p0 * (1.0 + event.relative_delta))

    final_target = profile.t_end_s + 2.0
    ok, records, reason = run_to(ss, final_target, max_chunk_s=max_chunk_s)
    for record in records:
        item = asdict(record)
        item.update({"segment_index": len(build_event_schedule(profile)), "attack_id": profile.attack_id})
        all_records.append(item)
    base = {
        **profile.to_dict(),
        "test_system": test_system,
        "case_version": "ANDES-2.0.0",
        "pq_idx": str(selected_pq),
        "pq_bus": pq_bus,
        "pq_base_pu": p0,
        "pq_selection": pq_selection,
        "criteria": criteria,
        "tstep_s": tstep_s,
        "disabled_builtin_events": ",".join(
            getattr(ss, "_gridpulse_disabled_events", [])
        ),
        "converged": bool(ok),
        "failure_reason": reason,
        "t_reached_s": current_time(ss),
    }
    if ok:
        base.update(
            _extract_omega(
                ss,
                model_name=model_name,
                t_start_s=max(profile.t_start_s + 1.0, 2.0),
                t_end_s=final_target,
                forcing_frequency_hz=profile.frequency_hz,
            )
        )
    return SimulationResult(row=base, convergence=all_records)


def simulate_profile(
    builder: Callable[[], Any],
    profile: AttackProfile,
    *,
    test_system: str,
    model_name: str,
    pq_idx: Any | None = None,
    max_chunk_s: float = 0.5,
    max_rebuilds: int = 2,
    criteria: int = 1,
    pq_selection: str = "first",
    base_tstep_s: float = 1.0 / 30.0,
) -> SimulationResult:
    """Run a profile, rebuilding and replaying from t=0 after busted failures."""
    combined_logs: list[dict[str, Any]] = []
    last: SimulationResult | None = None
    for rebuild in range(max_rebuilds + 1):
        result = _run_once(
            builder,
            profile,
            test_system=test_system,
            model_name=model_name,
            pq_idx=pq_idx,
            max_chunk_s=max_chunk_s / (2**rebuild),
            criteria=criteria,
            pq_selection=pq_selection,
            tstep_s=base_tstep_s / (2**rebuild),
        )
        for item in result.convergence:
            item["rebuild"] = rebuild
        combined_logs.extend(result.convergence)
        result.row["rebuild_count"] = rebuild
        last = result
        if result.row["converged"]:
            result.convergence = combined_logs
            return result
        if not any(bool(item.get("busted")) for item in result.convergence):
            break
    assert last is not None
    last.convergence = combined_logs
    return last


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def power_series_to_relative_events(
    power_series: np.ndarray,
    sample_hz: float,
    *,
    t_start_s: float = 1.0,
    max_amp_frac: float = 0.15,
    min_interval_s: float = 0.1,
    change_eps: float = 0.01,
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """Convert observed GPU watts into sparse PQ relative-delta events.

    Mapping is shape-preserving relative deviation around the window mean,
    not a conversion from watts to grid MW. The relative deviation is clipped
    to ``max_amp_frac`` so public test-system TDS remains numerically stable.
    """
    power = np.asarray(power_series, dtype=float)
    power = power[np.isfinite(power)]
    if len(power) < 8:
        raise ValueError("power_series needs at least 8 finite samples")
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")

    mean = float(np.mean(power))
    if abs(mean) < 1e-9:
        raise ValueError("power_series mean is near zero; cannot normalize")
    raw_delta = (power - mean) / mean
    peak_raw = float(np.max(np.abs(raw_delta)))
    scale = 1.0
    if peak_raw > max_amp_frac and peak_raw > 0:
        scale = max_amp_frac / peak_raw
    clipped = np.clip(raw_delta * scale, -max_amp_frac, max_amp_frac)

    events: list[tuple[float, float]] = []
    last_t = -1e9
    last_delta: float | None = None
    for i, delta in enumerate(clipped):
        t = t_start_s + i / sample_hz
        delta_f = float(delta)
        if last_delta is None:
            events.append((t, delta_f))
            last_t, last_delta = t, delta_f
            continue
        if (t - last_t) < min_interval_s and abs(delta_f - last_delta) < change_eps:
            continue
        if abs(delta_f - last_delta) < change_eps and (t - last_t) < max(min_interval_s * 5, 0.5):
            continue
        events.append((t, delta_f))
        last_t, last_delta = t, delta_f

    t_end = t_start_s + (len(clipped) - 1) / sample_hz
    if not events or abs(events[-1][0] - t_end) > 1e-9:
        events.append((t_end, float(clipped[-1])))
    events.append((t_end + 2.0, 0.0))

    meta = {
        "mean_w": mean,
        "peak_raw_delta": peak_raw,
        "amp_scale": float(scale),
        "max_amp_frac": float(max_amp_frac),
        "n_events": float(len(events)),
        "t_start_s": float(t_start_s),
        "t_end_s": float(t_end),
        "duration_s": float(t_end - t_start_s),
    }
    return events, meta


def inject_observed_waveform(
    power_series: np.ndarray,
    sample_hz: float,
    *,
    test_system: str = "kundur_ieeest",
    max_amp_frac: float = 0.15,
    min_interval_s: float = 0.1,
    max_chunk_s: float = 0.5,
    criteria: int = 1,
) -> dict[str, Any]:
    """Replay an observed power(t) window onto PQ.Ppf of a public test system.

    Unlike fixed AttackProfile square/ramp generators, this path does not invent
    a synthetic frequency/amplitude scenario. It replays the candidate window
    shape as a relative load modulation on Kundur/WECC.
    """
    if test_system == "kundur_ieeest":
        builder, model_name, pq_selection = build_kundur_system, "GENROU", "first"
    elif test_system in {"wecc_179_gencls", "wecc"}:
        builder, model_name, pq_selection = build_wecc_system, "GENCLS", "largest_p0"
    else:
        raise ValueError(f"unsupported test_system: {test_system}")

    events, meta = power_series_to_relative_events(
        power_series,
        sample_hz,
        max_amp_frac=max_amp_frac,
        min_interval_s=min_interval_s,
    )

    ss = builder()
    ss.TDS.config.criteria = criteria
    ss.TDS.config.tstep = 1.0 / 30.0
    ss.TDS.config.shrinkt = 1
    ss.TDS.config.max_iter = max(30, int(ss.TDS.config.max_iter))
    selected_pq = select_pq_idx(ss, strategy=pq_selection)
    p0 = float(ss.PQ.get(src="p0", idx=selected_pq, attr="v"))
    pq_bus = str(ss.PQ.get(src="bus", idx=selected_pq, attr="v"))

    for t_s, relative_delta in events:
        ok, _, reason = run_to(ss, t_s, max_chunk_s=max_chunk_s)
        if not ok:
            return {
                "test_system": test_system,
                "case_version": "ANDES-2.0.0",
                "mode": "observed_waveform_replay",
                "pq_idx": str(selected_pq),
                "pq_bus": pq_bus,
                "pq_base_pu": p0,
                "converged": False,
                "failure_reason": reason,
                "t_reached_s": current_time(ss),
                "osc_std": None,
                "rocof_hz_s": None,
                "dominant_freq_hz": None,
                "osc_ptp": None,
                **{f"waveform_{k}": v for k, v in meta.items()},
            }
        ss.PQ.set(src="Ppf", idx=selected_pq, value=p0 * (1.0 + relative_delta))

    features = _extract_omega(
        ss,
        model_name=model_name,
        t_start_s=meta["t_start_s"],
        t_end_s=meta["t_end_s"] + 2.0,
        forcing_frequency_hz=None,
    )
    return {
        "test_system": test_system,
        "case_version": "ANDES-2.0.0",
        "mode": "observed_waveform_replay",
        "pq_idx": str(selected_pq),
        "pq_bus": pq_bus,
        "pq_base_pu": p0,
        "converged": True,
        "failure_reason": "",
        "t_reached_s": current_time(ss),
        **features,
        **{f"waveform_{k}": v for k, v in meta.items()},
    }


def inject_observed_waveform_with_timeout(
    power_series: np.ndarray,
    sample_hz: float,
    *,
    timeout_s: float = 60.0,
    test_system: str = "kundur_ieeest",
    max_amp_frac: float = 0.15,
) -> dict[str, Any]:
    """Run observed-waveform injection with a wall-clock timeout.

    Uses SIGALRM so a hung TDS does not block the cyber pipeline forever.
    """
    import signal

    class _PhysicsTimeout(Exception):
        pass

    def _handle(_signum, _frame) -> None:
        raise _PhysicsTimeout(f"physics validation exceeded {timeout_s:.1f}s")

    previous = signal.signal(signal.SIGALRM, _handle)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        return inject_observed_waveform(
            power_series,
            sample_hz,
            test_system=test_system,
            max_amp_frac=max_amp_frac,
        )
    except _PhysicsTimeout as exc:
        return {
            "test_system": test_system,
            "mode": "observed_waveform_replay",
            "converged": False,
            "failure_reason": str(exc),
            "osc_std": None,
            "rocof_hz_s": None,
            "dominant_freq_hz": None,
            "osc_ptp": None,
        }
    except Exception as exc:
        return {
            "test_system": test_system,
            "mode": "observed_waveform_replay",
            "converged": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "osc_std": None,
            "rocof_hz_s": None,
            "dominant_freq_hz": None,
            "osc_ptp": None,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
