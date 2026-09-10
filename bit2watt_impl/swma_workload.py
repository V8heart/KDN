"""PyTorch 연산/sleep 토글로 규칙적 SWMA-like 부하를 재현한다.

논문의 persistent CUDA kernel을 동일 재현하는 코드가 아니라, NVML/DCGM에서
관측 가능한 규칙적 전력 변동을 검증하기 위한 통제된 근사 워크로드다.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def run(args) -> None:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise RuntimeError("CUDA 지원 PyTorch가 필요합니다.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")
    if not 0 < args.duration <= 120:
        raise ValueError("--duration은 0초 초과 120초 이하여야 합니다.")
    if not 0.1 <= args.duty_cycle <= 0.9:
        raise ValueError("--duty-cycle은 0.1~0.9 범위여야 합니다.")
    if args.period < 0.05:
        raise ValueError("--period는 최소 50ms입니다. kHz 물리 재현용 도구가 아닙니다.")

    rank = 0
    if args.distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        args.gpu_id = rank
    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)
    a = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=torch.float16)
    b = torch.randn_like(a)
    deadline = time.monotonic() + args.duration
    cycles = 0
    if args.distributed:
        dist.barrier()
        deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        cycle_start = time.monotonic()
        active_deadline = cycle_start + args.period * args.duty_cycle
        while time.monotonic() < active_deadline:
            result = a @ b
            a, b = b, result
        torch.cuda.synchronize()
        passive_deadline = cycle_start + args.period
        time.sleep(max(0.0, passive_deadline - time.monotonic()))
        cycles += 1
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({
            "workload": "swma-like",
            "gpu_id": args.gpu_id,
            "distributed": args.distributed,
            "duration_s": args.duration,
            "period_s": args.period,
            "duty_cycle": args.duty_cycle,
            "cycles": cycles,
            "approximation": True,
        }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--duty-cycle", type=float, default=0.5)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--distributed", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

