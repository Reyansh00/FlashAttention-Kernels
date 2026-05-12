import time
import math
import sys
import torch

import naive_attention
import tiled_attention
import tiled_attention_2
import online_softmax
import online_warp

def attention_cpu(q, k, v):
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def attention_cuda_naive(q, k, v):
    return naive_attention.forward(q, k, v)


def attention_cuda_tiled(q, k, v):
    return tiled_attention.forward(q, k, v)

def attention_cuda_tiled_2(q, k, v):
    return tiled_attention_2.forward(q, k, v)

def attention_cuda_online_softmax(q, k, v):
    return online_softmax.forward(q, k, v)

def attention_cuda_online_warp(q, k, v):
    return online_warp.forward(q, k, v)

# -----------------------------
# kernel selection
# -----------------------------
KERNELS = {
    0: ("naive", attention_cuda_naive),
    1: ("tiled", attention_cuda_tiled),
    2: ("tiled_2", attention_cuda_tiled_2),
    3: ("online_softmax", attention_cuda_online_softmax),
    4: ("online_warp", attention_cuda_online_warp),
}

if len(sys.argv) < 2:
    print("usage: python test.py <kernel_id>")
    print("0 -> naive")
    print("1 -> tiled")
    print("2 -> tiled_2")
    sys.exit(1)

kernel_id = int(sys.argv[1])

if kernel_id not in KERNELS:
    print("invalid kernel id")
    sys.exit(1)

kernel_name, attention_cuda = KERNELS[kernel_id]

print(f"running kernel: {kernel_name}")


# -----------------------------
# test config
# -----------------------------
B, N, D = 2, 4096, 64

q = torch.randn(B, N, D, dtype=torch.float32)
k = torch.randn(B, N, D, dtype=torch.float32)
v = torch.randn(B, N, D, dtype=torch.float32)

ref = attention_cpu(q, k, v)

qg = q.cuda()
kg = k.cuda()
vg = v.cuda()


# -----------------------------
# warmup
# -----------------------------
for _ in range(10):
    attention_cuda(qg, kg, vg)

torch.cuda.synchronize()


# -----------------------------
# correctness
# -----------------------------
out = attention_cuda(qg, kg, vg).cpu()

print("max error:", (ref - out).abs().max().item())


# -----------------------------
# cpu timing
# -----------------------------
t0 = time.time()

for _ in range(20):
    out = attention_cpu(q, k, v)

print("cpu time:", time.time() - t0)


# -----------------------------
# gpu timing + profiling
# -----------------------------
torch.cuda.cudart().cudaProfilerStart()

torch.cuda.synchronize()

t0 = time.time()

for _ in range(20):
    out = attention_cuda(qg, kg, vg)

torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStop()

print("cuda time:", time.time() - t0)