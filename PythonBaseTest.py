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

a=[1,1,1,1,5,10,1,2,3]
b=[1,1]
index=0
find_all_index=[]
while(index<len(a)):
    if a[index]==b[0] and index+len(b)<=len(a) and a[index:index+len(b)]==b[:] :
        find_all_index.append((index,index+len(b)))
        index+=len(b)
    else:
        index+=1

print(find_all_index)






