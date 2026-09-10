"""정상 학습 루프에 불규칙 보조 연산을 삽입하는 LTMA-like wrapper."""
from __future__ import annotations

import argparse
import json
import random
import time


def run(args) -> None:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("CUDA 지원 PyTorch가 필요합니다.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")
    if not 0 < args.duration <= 120:
        raise ValueError("--duration은 0초 초과 120초 이하여야 합니다.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    model = nn.Sequential(
        nn.Linear(args.width, args.width),
        nn.GELU(),
        nn.Linear(args.width, args.width),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    aux_a = torch.randn(args.width, args.width, device=device, dtype=torch.float16)
    aux_b = torch.randn_like(aux_a)
    deadline = time.monotonic() + args.duration
    step = inserted = 0
    next_injection = random.randint(7, 23)

    while time.monotonic() < deadline:
        batch = random.choice([16, 24, 32, 40])
        x = torch.randn(batch, args.width, device=device)
        target = torch.randn_like(x)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        step += 1

        if step >= next_injection:
            repeats = random.randint(1, args.max_aux_repeats)
            for _ in range(repeats):
                aux_a = torch.tanh(aux_a @ aux_b)
            torch.cuda.synchronize()
            if random.random() < 0.5:
                time.sleep(random.uniform(0.01, 0.15))
            inserted += 1
            next_injection = step + random.randint(5, 31)

    torch.cuda.synchronize()
    print(json.dumps({
        "workload": "ltma-like",
        "gpu_id": args.gpu_id,
        "duration_s": args.duration,
        "training_steps": step,
        "injection_events": inserted,
        "approximation": True,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-aux-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

