import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR

from dataset import CustomDataset
from model import CustomModel
from loss import CustomLoss


def train(model, device, train_loader, optimizer, epoch, loss_function, log_interval=10):

	model.train()
	losses = []
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = loss_function(output, target)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

	loss_average = torch.mean(torch.tensor(losses)).item()
	print("Epoch: {}".format(epoch))
	print("Training set -> Average Loss: {}".format(loss_average))
	return loss_average



def test(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(num_epochs, batch_size_train, batch_size_test, lr, gamma, device, seed=None):

	if not seed:
		torch.manual_seed(seed)

	tr = transforms.Compose([transforms.RandomResizedCrop(300),
                             transforms.ToTensor(),
                             transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])])

    augs = transforms.Compose([transforms.RandomResizedCrop(300),
                               transforms.RandomRotation(20),
                               transforms.ToTensor(),
                               transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])])

    train_set = CustomDataset(dir_csv, dir_img, transforms=augs)
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)

    val_set = CustomDataset(dir_csv, dir_img, transforms=tr)
    val_loader = DataLoader(val_set, batch_size=batch_size_test, shuffle=False)

    model = CustomModel()
    loss_function = CustomLoss()

    model.to(device)
    print('Starting optimizer with LR={}'.format(lr))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
   
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_function)
        test(model, device, test_loader, loss_function)
        scheduler.step()

    torch.save(model.state_dict(), "well_trained model.pt")


if __name__ == "__main__":
	num_epochs = 5
	batch_size = 64
	lr = 0.01
	gamma = 0.9
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	main(num_epochs, batch_size, lr, gamma, device)

