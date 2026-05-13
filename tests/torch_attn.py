import math
import torch

def attention_cpu(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    q, k, v: [B, N, D], float32, on CPU
    returns: [B, N, D]
    """
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)   # [B, N, N]
    probs = torch.softmax(scores, dim=-1)             # [B, N, N]
    out = probs @ v                                   # [B, N, D]
    return out