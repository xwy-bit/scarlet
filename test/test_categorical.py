from torch.distributions import Categorical
import torch
a = torch.Tensor([1,2,3])
b = Categorical(a)
c = [0,0,0]
for i in range(100):
    c[b.sample()] += 1

print(c)