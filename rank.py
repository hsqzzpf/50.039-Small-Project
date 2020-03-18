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


def ranking(model, device, data_loader, choice, top_k):
	model.eval()
	img_ls = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(data_loader):
			data, target, img_name = batch[0].to(device), batch[1].to(device), batch[2]
			output = model(data)
			pred = torch.sigmoid(output)
			if batch_idx == 0:
				predictions = pred
				
			else:
				predictions = torch.cat((predictions, pred))
			img_ls += img_name

	num_img = len(img_ls)		
	if len(predictions) != num_img:
		return "Wrong Length"

	#print(predictions.shape)
	num_classes = predictions.shape[1]
	classes = list(np.arange(num_classes))
	
	random_choice = random.choices(classes, k=choice)
	#print(random_choice)
	class_pred = {}
	img_pred = []
	for c in random_choice:
		tmp = []
		for i in range(num_img):
			tmp.append(predictions[i][c].item())
		class_pred[c] = tmp
		img_tmp = {}
		for j in range(num_img):
			img_tmp[img_ls[j]] = class_pred[c][j]

		img_pred.append(img_tmp)
	#print(len(random_choice))
	labels_dict = get_label_dict()
	class_count = 0
	for dic in img_pred:
		
		curr_class = labels_dict[random_choice[class_count]]
		#print(curr_class)
		top_img = []
		bottom_img = []
		values_ls = list(dic.values())
		values_ls.sort()
		top_value = values_ls[len(dic)-top_k:]
		bottom_value = values_ls[:top_k]
		#print(len(top_value), len(low_value))
		top_value.reverse()
		for i in range(len(top_value)):
			for key in dic:
				if dic[key] == top_value[i]:
					top_img.append(key)

		for j in range(len(bottom_value)):
			for key in dic:
				if dic[key] == bottom_value[j]:
					bottom_img.append(key)
		
		
		if not os.path.exists('class_'+curr_class):
			os.makedirs('class_'+curr_class)
			
		if not os.path.exists('class_'+curr_class+'/'+"top"):
			os.makedirs('class_'+curr_class+'/'+"top")
		
		if not os.path.exists('class_'+curr_class+'/'+"bottom"):
			os.makedirs('class_'+curr_class+'/'+"bottom")
		
		
		out = open('class_'+curr_class+'/'+"top/images_name.csv", 'a')
		wr = csv.writer(out, dialect='excel')
		for i in range(len(top_img)):
			name = top_img[i].split('/')[-1]
			#print(name)
			copyfile(top_img[i], 'class_'+curr_class+'/'+"top/"+name)
			wr.writerow([top_img[i], i+1])
			os.rename('class_'+curr_class+'/'+"top/"+name, 'class_'+curr_class+'/'+"top/"+str(i+1)+"_"+name)
		out.close()

		outb = open('class_'+curr_class+'/'+"bottom/images_name.csv", 'a')
		wrb = csv.writer(outb, dialect='excel')
		for i in range(len(bottom_img)):
			name = bottom_img[i].split('/')[-1]
			copyfile(bottom_img[i], 'class_'+curr_class+'/'+"bottom/"+name)
			wrb.writerow([bottom_img[i], i+1])
			os.rename('class_'+curr_class+'/'+"bottom/"+name, 'class_'+curr_class+'/'+"bottom/"+str(i+1)+"_"+name)
		outb.close()
		class_count += 1
	print("Done")
		



if __name__ == "__main__":
	tr = transforms.Compose([transforms.RandomResizedCrop(300),
						transforms.ToTensor(),
						transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])])
	device = torch.device("cuda")
	model = models.resnet34(pretrained=True)
	model.fc = nn.Linear(512, 20)
	model.load_state_dict(torch.load("test.pt"))
	model = model.to(device)
	val_set = CustomDataset("val.csv", "", transforms=tr)
	val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
	
	ranking(model, device, val_loader, 5, 50)