import torch

from PraticeOfTransformers.CustomModel import BertForUnionNspAndQA

model = BertForUnionNspAndQA()
model.load_state_dict(torch.load("model/path1"))
model.eval() #==  self.train(False) ，使用评估模式（model.train()训练模式） 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。