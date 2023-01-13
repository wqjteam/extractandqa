import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

pred = torch.tensor([[0.0,10.0,0.0],[0.0,0.0,10.0]])

target = torch.tensor([1, 2], dtype=torch.long)

# 函数形式：
out = F.cross_entropy(pred, target)

# 类的形式：
func = CrossEntropyLoss()  # 括号不要掉，定义一个对象
out = func(pred, target)
print(out)

