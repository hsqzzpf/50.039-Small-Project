import torch.nn as nn
from torch.nn import LogSigmoid
from sklearn.metrics import log_loss


log_sigmoid = LogSigmoid()


class CustomLoss1(nn.Module):

	def __init__(self):
		super(CustomLoss, self).__init__()
		

	def forward(self, inputs, targets):

		inputs = inputs.flatten()
		targets = target.flatten()

		loss = -(targets * log_sigmoid(inputs) + (1 - targets) * log_sigmoid(-inputs))
		return torch.sum(loss)


class CustomLoss2(nn.Module):

	def __init__(self):
		super(CustomLoss, self).__init__()
		

	def forward(self, inputs, targets):

		loss = -(targets * log_sigmoid(inputs) + (1 - targets) * log_sigmoid(-inputs))
		return torch.sum(loss)


class CustomLoss3(nn.Module):

	def __init__(self):
		super(CustomLoss, self).__init__()
		

	def forward(self, inputs, targets):
		return torch.mean((inputs - targets)**2)
