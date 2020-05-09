#coding=utf-8
import torch.nn as nn

# 返回归一化层
def tensor_layer_norm(num_features):
	return nn.BatchNorm2d(num_features)
