import torch

from PraticeOfTransformers.CustomModel import BertForUnionNspAndQA

model = BertForUnionNspAndQA()
model.load_state_dict(torch.load("model/path1"))
model.eval()