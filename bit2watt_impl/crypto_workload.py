"""로컬 GPU 지속 고부하 벤치 (cryptojacking-like).

실제 채굴 풀/네트워크에 접속하지 않는다. NVML에서 '높고 평평한' 전력·사용률
패턴을 재현하기 위한 통제된 스모크/검증용 워크로드다.
"""
from __future__ import annotations

import argparse
import json
import time


def run(args) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA 지원 PyTorch가 필요합니다.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")
    if not 0 < args.duration <= 120:
        raise ValueError("--duration은 0초 초과 120초 이하여야 합니다.")

    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)
    a = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=torch.float16)
    b = torch.randn_like(a)
    deadline = time.monotonic() + args.duration
    steps = 0
    while time.monotonic() < deadline:
        a = torch.tanh(a @ b)
        steps += 1
        if steps % 8 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    print(json.dumps({
        "workload": "cryptojacking-like",
        "gpu_id": args.gpu_id,
        "duration_s": args.duration,
        "matrix_size": args.matrix_size,
        "steps": steps,
        "network": False,
        "approximation": True,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--matrix-size", type=int, default=4096)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
