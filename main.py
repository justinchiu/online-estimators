import torch
import runningstats
from welford_torch import Welford

x = torch.randn(100, 1000)

v = runningstats.Variance()
w = Welford()
for y in x:
    v.add(y[None])
    w.add(y[None])

true = x.var(0).sum()
runningstats_var = v.variance().sum()
welford_var = w.var_s.sum()

print(true)
print(runningstats_var)
print(welford_var)
