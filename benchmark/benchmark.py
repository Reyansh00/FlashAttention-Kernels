import time
import torch
import pandas as pd
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
sys.path.append(str(ROOT))
sys.path.append(str(BIN))

import naive_attention
import tiled_attention
import tiled_attention_2
import online_softmax
import online_warp

device = "cuda"

B = 2
D = 64

seq_lengths = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
]

warmup = 10
iters = 50

results = []

# ----------------------------------------
# timing helper
# ----------------------------------------
def benchmark(fn):

    torch.cuda.synchronize()

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = time.time()

    for _ in range(iters):
        fn()

    torch.cuda.synchronize()

    end = time.time()

    return (end - start) * 1000 / iters


# ----------------------------------------
# benchmark loop
# ----------------------------------------
for N in seq_lengths:

    print(f"\n===== N={N} =====")

    # fp32 for custom kernels
    q32 = torch.randn(B, N, D, device=device, dtype=torch.float32)
    k32 = torch.randn(B, N, D, device=device, dtype=torch.float32)
    v32 = torch.randn(B, N, D, device=device, dtype=torch.float32)

    # fp16 for flash attention
    q16 = q32.half()
    k16 = k32.half()
    v16 = v32.half()

    q16_sdpa = q16.unsqueeze(1)
    k16_sdpa = k16.unsqueeze(1)
    v16_sdpa = v16.unsqueeze(1)

    q32_sdpa = q32.unsqueeze(1)
    k32_sdpa = k32.unsqueeze(1)
    v32_sdpa = v32.unsqueeze(1)

    kernels = [
        ("naive_attention",
         lambda: naive_attention.forward(q32, k32, v32)),

        ("tiled_attention",
         lambda: tiled_attention.forward(q32, k32, v32)),

        ("tiled_attention_2",
         lambda: tiled_attention_2.forward(q32, k32, v32)),

        ("online_softmax",
         lambda: online_softmax.forward(q32, k32, v32)),

        ("online_warp",
         lambda: online_warp.forward(q32, k32, v32)),
    ]

    for name, fn in kernels:

        ms = benchmark(fn)

        print(f"{name:25s}: {ms:.3f} ms")

        results.append({
            "kernel": name,
            "N": N,
            "time_ms": ms,
        })

    # ----------------------------------------
    # SDPA math backend (fp32)
    # ----------------------------------------
    with sdpa_kernel(SDPBackend.MATH):

        ms = benchmark(
            lambda: F.scaled_dot_product_attention(
                q32_sdpa,
                k32_sdpa,
                v32_sdpa,
                dropout_p=0.0,
                is_causal=False
            )
        )

        print(f"{'sdpa_math':25s}: {ms:.3f} ms")

        results.append({
            "kernel": "sdpa_math",
            "N": N,
            "time_ms": ms,
        })

    # ----------------------------------------
    # FlashAttention backend (fp16)
    # ----------------------------------------
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):

        ms = benchmark(
            lambda: F.scaled_dot_product_attention(
                q16_sdpa,
                k16_sdpa,
                v16_sdpa,
                dropout_p=0.0,
                is_causal=False
            )
        )

        print(f"{'sdpa_flash':25s}: {ms:.3f} ms")

        results.append({
            "kernel": "sdpa_flash",
            "N": N,
            "time_ms": ms,
        })

# ----------------------------------------
# save results
# ----------------------------------------
df = pd.DataFrame(results)

print("\n")
print(df)

df.to_csv("attention_benchmarks.csv", index=False)

print("\nSaved to attention_benchmarks.csv")