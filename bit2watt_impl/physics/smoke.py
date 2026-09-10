"""Fast environment and case-availability smoke checks for Physics Tier B."""
from __future__ import annotations

import json

from bit2watt_impl.physics.simulation import (
    build_kundur_system,
    build_wecc_system,
    run_to,
)


def main() -> None:
    rows = []
    for name, builder in (
        ("kundur_ieeest", build_kundur_system),
        ("wecc_179_gencls", build_wecc_system),
    ):
        try:
            ss = builder()
            ok, _, reason = run_to(ss, 0.2, max_chunk_s=0.2)
            rows.append(
                {
                    "test_system": name,
                    "ok": ok,
                    "reason": reason,
                    "pq_count": int(len(ss.PQ)),
                    "generator_count": int(len(ss.GENROU) + len(ss.GENCLS)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "test_system": name,
                    "ok": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if not all(row["ok"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
