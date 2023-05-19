# 导包，需要torch和torch.nn
import torch
from torch import nn

# 构建一个embedding module，其包含5个size为3的tensor
# 这里的embedding相当于是一个Embedding层，写于forward()方法中，还没有真实值的传入，只能够得到其size为(5,4):表示含有5个tensor，且每个tensor=[num1,num2,num3,num4]
embedding = nn.Embedding(5, 4)

# 接下来构建一个参数tensor并传入Embedding层中
# 在构建test时，需要注意，当你设置5个tensor时，表示test中的数值的范围在[0,1,2,3,4]，输入数值不可不在这个范围内，否则会报错：index out of range in self
test = [[1, 2, 3],
        [2, 3, 4]]
embed = embedding(torch.LongTensor(test))
# 将test传入Embedding层，即可构建一个look-up embedding查询表
print(embed)
print(embed.size())