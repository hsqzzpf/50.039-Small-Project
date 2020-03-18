import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR

from dataset import CustomDataset
from model import CustomModel
from loss import CustomLoss1, CustomLoss2, CustomLoss3
from sklearn.metrics import average_precision_score
import torchvision.models as models
from torchvision import transforms
from rank import ranking
import matplotlib.pyplot as plt
import numpy as np

def plot(name, ls, lr, criterion_name, epoch):
	if criterion_name is not None:
		
		for i in range(len(ls)):
			plt.plot(np.arange(1,epoch+1), ls[i])

		plt.title("model " + name + " at learning rate "+str(lr))
		plt.xlabel("epoch")
		plt.xticks(np.arange(1,epoch+1))
		plt.ylabel(name)
		if name == "loss":
			plt.legend(["Train", "Val"], loc = "upper right")
		else:
			plt.legend(["Train", "Val"], loc = "upper left")
		
		plt.savefig(criterion_name+"_"+name+"_01")
		plt.clf()

	else:
		for i in range(len(ls)):
			plt.plot(np.arange(1,epoch+1), ls[i])
		plt.title("model " + name + " at learning rate "+str(lr))
		plt.xlabel("epoch")
		plt.xticks(np.arange(1,epoch+1))
		plt.ylabel(name)
		

		if name == "loss":
			plt.legend(["Train1", "Val1", "Train2", "Val2", "Train3", "Val3"], loc = "upper right")
		else:
			plt.legend(["Train1", "Val1", "Train2", "Val2", "Train3", "Val3"], loc = "upper left")
		
		plt.savefig("total_"+name+"_01")
		plt.clf()

def train_epoch(model, device, train_loader, optimizer, loss_function, log_interval=10):

	model.train()
	losses = []
	
	for batch_idx, batch in enumerate(train_loader):
		data, target = batch[0], batch[1]
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
		# 		100.0 * batch_idx * len(data)/ len(train_loader.dataset), loss.item()))

	loss_average = sum(losses)/len(losses)
	
	print("Average Training Loss: {:.5f}".format(loss_average))

	return loss_average


def average_precision_measure(pred, target):
	return average_precision_score(target.cpu(), pred.cpu())

def evaluate(model, device, data_loader, loss_function):
	model.eval()
	eval_loss = 0
	losses = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(data_loader):
			data, target = batch[0], batch[1]
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = loss_function(output, target)
			losses.append(loss.item())
			#print("output: ", output)
			pred = torch.sigmoid(output)
			#print("pred: ", pred)
			if batch_idx == 0:
				predictions = pred
				targets = target
			else:
				predictions = torch.cat((predictions, pred))
				targets = torch.cat((targets, target))

	eval_loss =sum(losses)/len(losses)
	ap_score = average_precision_measure(predictions.reshape(-1, 20), targets.reshape(-1,20))
	

	return eval_loss, ap_score



def train(train_loader, val_loader, model, num_epochs, batch_size_train, batch_size_val, criterion, lr, gamma, device,):
	


	#loss_function = nn.BCEWithLogitsLoss()
	loss_function = criterion

	model.to(device)
	print('Starting optimizer with LR={}'.format(lr))
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
   
	#scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
	train_losses = []
	train_scores = []
	eval_losses = []
	ap_scores = []
	best_measure = 0
	best_epoch = -1
	for epoch in range(1, num_epochs + 1):
		print("############### Epoch {} ###############".format(epoch))
		model.train(True)
		train_loss = train_epoch(model, device, train_loader, optimizer, loss_function)
		train_losses.append(train_loss)
		model.train(False)
		_, train_score = evaluate(model, device, train_loader, loss_function)
		print("Training Accuracy: {:.5f}".format(train_score))
		eval_loss, ap_score = evaluate(model, device, val_loader, loss_function)
		#scheduler.step()
		print("Average Validation Loss: {:.5f}".format(eval_loss))
		print("Validation Accuracy: {:.5f}".format(ap_score))

		train_scores.append(train_score)
		eval_losses.append(eval_loss)
		ap_scores.append(ap_score)

		if ap_score > best_measure:
			bestweights = model.state_dict()
			best_measure = ap_score
			best_epoch = epoch
		print('current best: {:.5f} at epoch {}'.format(best_measure, best_epoch))
	#torch.save(model.state_dict(), "well_trained model.pt")
		
		
	return train_scores, train_losses, ap_scores, eval_losses, bestweights

def run():
	num_epochs = 20
	batch_size_train = 64
	batch_size_val = 32
	lr = 0.01
	gamma = 0.9
	
	seed = None
	dir_img = ""
	customeLoss = [CustomLoss1(), CustomLoss2(), CustomLoss3()]
	customeLoss_name = ["CustomLoss1", "CustomLoss2", "CustomLoss3"]

	if seed is not None:
		torch.manual_seed(seed)
	
	tr = transforms.Compose([transforms.RandomResizedCrop(300),
							transforms.ToTensor(),
							transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])])
	
	augs = transforms.Compose([transforms.RandomResizedCrop(300),
							transforms.RandomRotation(20),
							transforms.ToTensor(),
							transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])])

	
	train_set = CustomDataset("train.csv", dir_img, transforms=augs)
	train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)

	val_set = CustomDataset("val.csv", dir_img, transforms=tr)
	val_loader = DataLoader(val_set, batch_size=batch_size_val, shuffle=False)

	# model = models.resnet34(pretrained=True)
	# model.fc = nn.Linear(512, 20)
	device = torch.device("cuda: 1") if torch.cuda.is_available() else torch.device('cpu')


	#criterion = CustomLoss1()
	total_loss = []
	total_score = []
	for i in range(len(customeLoss)):
		model = models.resnet34(pretrained=True)
		model.fc = nn.Linear(512, 20)
		

		print("Testing with {}...".format(customeLoss_name[i]))
		criterion = customeLoss[i]
		train_scores, train_losses, ap_scores, eval_losses, bestweights = train(train_loader, val_loader, model, num_epochs, 
																				batch_size_train, batch_size_val, criterion, 
																				lr, gamma, device)
		
		losses = [train_losses, eval_losses]
		scores = [train_scores, ap_scores]

		total_loss += losses
		total_score += scores

		total_loss.append
		plot("loss", losses, lr, customeLoss_name[i], num_epochs)
		plot("Score", scores, lr, customeLoss_name[i], num_epochs)
		print("Display the top-50 highest scoring images and lowest scoring images for 5 random classes...")
		model.load_state_dict(bestweights)
		ranking(model=model, device=device, data_loader=val_loader, choice=5, top_k=50)

	plot("loss", total_loss, lr, None, num_epochs)
	plot("Score", total_score, lr, None, num_epochs)



if __name__ == "__main__":
	run()

