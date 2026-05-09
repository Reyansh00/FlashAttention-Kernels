import time
import math
import torch
import naive_attention

def attention_cpu(q, k, v):
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v

def attention_cuda(q, k, v):
    return naive_attention.forward(q, k, v)

# test
B, N, D = 2, 128, 64
q = torch.randn(B, N, D, dtype=torch.float32)
k = torch.randn(B, N, D, dtype=torch.float32)
v = torch.randn(B, N, D, dtype=torch.float32)

ref = attention_cpu(q, k, v)

qg = q.cuda()
kg = k.cuda()
vg = v.cuda()

out = attention_cuda(qg, kg, vg).cpu()

print("max error:", (ref - out).abs().max().item())

# simple timing
tz = time.time()
for _ in range(20):
    out = attention_cpu(q, k, v)
print("cpu time:", time.time() - tz)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    out = attention_cuda(qg, kg, vg)
torch.cuda.synchronize()
print("cuda time:", time.time() - t0)