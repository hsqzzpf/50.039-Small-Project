import torch
import numpy as np
from dataset import CustomDataset
import torchvision.models as models
from torchvision import transforms

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR

from dataset import CustomDataset
from model import CustomModel
from loss import CustomLoss1, CustomLoss2, CustomLoss3
from sklearn.metrics import average_precision_score
import os
from shutil import copyfile
import csv
import random
import matplotlib.pyplot as plt

def get_label_dict():
    return {
            0 : 'aeroplane',
            1 : 'bicycle',
            2 : 'bird',
            3 : 'boat',
            4 : 'bottle',
            5 : 'bus',
            6 : 'car',
            7 : 'cat',
            8 : 'chair',
            9 : 'cow',
            10 : 'diningtable',
            11 : 'dog',
            12 : 'horse',
            13 : 'motorbike',
            14 : 'person',
            15 : 'pottedplant',
            16 : 'sheep',
            17 : 'sofa',
            18 : 'train',
            19 : 'tvmonitor'
        }

def plot(accuracies, thresholds):
	plt.plot(thresholds, accuracies, 'b', thresholds, accuracies, 'ro')
	plt.title("Average tailaccs over of each threshold")
	plt.xlabel("Thresholds")
	plt.xticks(thresholds.round(2))
	plt.ylabel("Tailaccs")
	plt.savefig("tailaccs")
	plt.clf()
	
	

def get_tail_acc(pred, gt, t):
	tp = 0
	fp = 0 
	for (p, g) in zip(pred, gt):
		pred = 0
		if p > t:
			pred = 1
			if pred == g:
				tp += 1
			else:
				fp += 1
	tailacc = 0
	if fp+tp > 0:
		tailacc = tp/(tp+fp)
	return tailacc

def class_wise_acc(total_tail):
	accuracies = []
	for tail in total_tail:
		acc = sum(tail)/len(tail)
		accuracies.append(acc)
	return accuracies

def tail_accuracy(model, device, data_loader, num_t = 15, low_t = 0.5, top_k=50):
	model.eval()
	img_ls = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(data_loader):
			data, target, img_name = batch[0].to(device), batch[1].to(device), batch[2]
			output = model(data)
			pred = torch.sigmoid(output)
			if batch_idx == 0:
				predictions = pred
				targets = target
			else:
				predictions = torch.cat((predictions, pred),0)
				targets = torch.cat((targets, target),0)
			img_ls += img_name
	num_img = len(img_ls)		

	#print(predictions.shape)
	num_classes = predictions.shape[1]
	classes = list(np.arange(num_classes))
	#print(predictions[:,0])
	#print(len(predictions[:,0]))
	#print(random_choice)
	max_pred = torch.max(predictions).item()
	print("Maximum prediction: ", max_pred)
	thresholds = np.linspace(low_t, max_pred, num_t, endpoint=False)
	class_pred = {}
	img_pred = []
	for c in classes:
		tmp = []
		for i in range(num_img):
			tmp.append((predictions[i][c].item(), targets[i][c].item()))
		class_pred[c] = tmp
		img_tmp = {}
		for j in range(num_img):
			img_tmp[img_ls[j]] = class_pred[c][j]

		img_pred.append(img_tmp)
	#print(len(random_choice))
	labels_dict = get_label_dict()
	total_tail = []
	for t in thresholds:
		tail = []
		class_count = 0
		for dic in img_pred:
			curr_class = labels_dict[class_count]
			top_img = []
			bottom_img = []
			values_ls = []
			for tup in dic.values():
				values_ls.append(tup[0])
			values_ls.sort()
			top_value = values_ls[len(dic)-top_k:]
			top_value.reverse()
			for i in range(len(top_value)):
				for key in dic:
					if dic[key][0] == top_value[i]:
						top_img.append(key)

			pred = []
			gt = []

			for i in range(len(top_img)):
				image = dic[top_img[i]]
				pred.append(image[0])
				gt.append(image[1])

			tailacc = get_tail_acc(pred, gt, t)
			print("Tail Accuracy for threshold {} of class {}: {}".format(t, curr_class, tailacc))
			class_count += 1	
			tail.append(tailacc)
		total_tail.append(tail)
	accuracies = class_wise_acc(total_tail)
	plot(accuracies, thresholds)
	print("Done")

		

if __name__ == "__main__":
	tr = transforms.Compose([transforms.CenterCrop(224),
							transforms.ToTensor(),
							transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	device = torch.device("cuda")
	model = models.resnet34(pretrained=True)
	model.fc = nn.Linear(512, 20)
	model.load_state_dict(torch.load("weights/01_CustomLoss1resnet34"))
	model = model.to(device)
	val_set = CustomDataset("val.csv", "", transforms=tr)
	val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
	
	tail_accuracy(model, device, val_loader, num_t = 15)