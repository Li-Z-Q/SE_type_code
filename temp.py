import torch

a = torch.randn(7, 300)
print(a.shape)

b = torch.randn(7)
print(b.shape)

c = torch.einsum("ij, i->ij", a, b)
print(c.shape)