import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR

from dataset import CustomDataset
from model import CustomModel
#from loss import CustomLoss1, CustomLoss2, CustomLoss3
from sklearn.metrics import average_precision_score
import torchvision.models as models
from torchvision import transforms

def train(model, device, train_loader, optimizer, epoch, loss_function, log_interval=10):

	model.train()
	losses = []
	for batch_idx, batch in enumerate(train_loader):
		data, target = batch
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = loss_function(output, target)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		
		# if batch_idx % log_interval == 0:
		# 	print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
		# 		epoch, batch_idx * len(data), len(train_loader.dataset),
		# 		100. * batch_idx / len(train_loader), loss.item()))

	loss_average = torch.mean(torch.tensor(losses)).item()
	print("Epoch: {}".format(epoch))
	print("Training set -> Average Loss: {}".format(loss_average))
	return loss_average

def average_precision_measure(pred, target):
	return average_precision_score(target.cpu(), pred.cpu())

def evaluate(model, device, data_loader, loss_function):
	model.eval()
	eval_loss = 0
	correct = 0
	with torch.no_grad():
		for batch_idx, batch in enumerate(data_loader):
			data, target = batch
			data, target = data.to(device), target.to(device)
			output = model(data)
			eval_loss += loss_function(output, target).item()
			#print("output: ", output)
			pred = torch.sigmoid(output)
			#print("pred: ", pred)
			if batch_idx == 0:
				predictions = pred
				targets = target
			else:
				predictions = torch.cat((predictions, pred))
				targets = torch.cat((targets, target))

	eval_loss /= len(data_loader)
	ap_score = average_precision_measure(predictions.reshape(-1, 20), targets.reshape(-1,20))
	print('\nValidation set: Average loss: {:.5f}, Accuracy: {:.5f})\n'.format(
		eval_loss, ap_score))

	return eval_loss, ap_score


def main(num_epochs, batch_size_train, batch_size_val, lr, gamma, device, seed=None):
	
	
	if seed is not None:
		torch.manual_seed(seed)
	
	tr = transforms.Compose([transforms.RandomResizedCrop(300),
							transforms.ToTensor(),
							transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])])
	
	augs = transforms.Compose([transforms.RandomResizedCrop(300),
							transforms.RandomRotation(20),
							transforms.ToTensor(),
							transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])])

	dir_img = ""
	train_set = CustomDataset("train.csv", dir_img, transforms=augs)
	train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)

	val_set = CustomDataset("val.csv", dir_img, transforms=tr)
	val_loader = DataLoader(val_set, batch_size=batch_size_val, shuffle=False)

	model = models.resnet34(pretrained=True)
	model.fc = nn.Linear(512, 20)

	loss_function = nn.BCEWithLogitsLoss()

	model.to(device)
	print('Starting optimizer with LR={}'.format(lr))
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
   
	scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
	for epoch in range(1, num_epochs + 1):
		train(model, device, train_loader, optimizer, epoch, loss_function)
		evaluate(model, device, val_loader, loss_function)
		scheduler.step()

	torch.save(model.state_dict(), "well_trained model.pt")


if __name__ == "__main__":
	num_epochs = 10
	batch_size_train = 32
	batch_size_val = 32
	ls = 0.01
	gamma = 0.9
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
	main(num_epochs, batch_size_train, batch_size_val, ls, gamma, device)

