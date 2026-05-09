import torch
import naive_attention

q = torch.randn(2, 128, 64, device='cuda')
k = torch.randn(2, 128, 64, device='cuda')
v = torch.randn(2, 128, 64, device='cuda')

out = naive_attention.forward(q, k, v)

print(out.shape)
print(out.device)
