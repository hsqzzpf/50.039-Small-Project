import torch
import torch.nn as nn
from torch.nn import LogSigmoid
from sklearn.metrics import log_loss


log_sigmoid = LogSigmoid()


class CustomLoss1(nn.Module):

	def __init__(self):
		super(CustomLoss1, self).__init__()
		

	def forward(self, inputs, targets):

		inputs = inputs.flatten()
		targets = targets.flatten()

		loss = -(targets * log_sigmoid(inputs) + (1 - targets) * log_sigmoid(1-inputs))
		# print("l1 loss average: {}".format(torch.mean(loss)))
		return torch.mean(loss)


class CustomLoss2(nn.Module):

	def __init__(self):
		super(CustomLoss2, self).__init__()
		

	def forward(self, inputs, targets):

		loss = -(targets * log_sigmoid(inputs) + (1 - targets) * log_sigmoid(1-inputs))
		# print("l2 loss average: {}".format(torch.mean(loss)))
		return torch.mean(loss)


class CustomLoss3(nn.Module):

	def __init__(self):
		super(CustomLoss3, self).__init__()
		

	def forward(self, inputs, targets):
		return torch.mean((inputs - targets)**2)

if __name__ == "__main__":
	loss_fn1 = CustomLoss1()
	loss_fn2 = CustomLoss2()

	y_true = torch.tensor([[0., 0., 1., 1.],
							[0., 0., 1., 1.]])

	y_pred = torch.tensor([[0.1, 0.1, 0.9, 0.9],
							[0.1, 0.1, 0.9, 0.8]])

	print(loss_fn1(y_pred, y_true))
	print(loss_fn2(y_pred, y_true))
	print(log_sigmoid(torch.tensor(0.9)))

