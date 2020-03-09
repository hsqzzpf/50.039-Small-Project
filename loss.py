import torch.nn as nn


class CustomLoss(nn.Module):

	def __init__(self):
		super(CustomLoss, self).__init__()
		

	def forward(self, inputs, targets):
		return torch.mean((inputs - targets)**2)