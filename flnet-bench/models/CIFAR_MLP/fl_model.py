import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# Training settings
lr = 0.01
momentum = 0.9
log_interval = 10
rou = 1
loss_thres = 0.01

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for CIFAR-10 dataset."""
    # Extract CIFAR-10 data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.CIFAR10(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.testset = datasets.CIFAR10(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.labels = list(self.trainset.classes)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_input = nn.Linear(3072, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         #x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
# class Net(nn.Module): 
#     def __init__(self):
#         super(Net, self).__init__()  
#         self.fc1 = nn.Linear(32 * 32 * 3,64) 
#         self.fc2 = nn.Linear(64, 64) 
#         self.fc3 = nn.Linear(64, 64)
#         self.fc4 = nn.Linear(64, 10) 
     
#     def forward(self, x):
#         x = x.view(-1,32 * 32 * 3)    
#         x = F.relu(self.fc1(x))  
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.softmax(self.fc4(x), dim = 1) 
#         return x


def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def extract_weights(model):
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights

def extract_weights_cuda(model):
    weights = []
    for name, weight in model.named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights

def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)

def flatten_weights(weights):
    # Flatten weights into vectors
    weight_vecs = []
    for _, weight in weights:
        weight_vecs.extend(weight.flatten().tolist())

    return np.array(weight_vecs)


def train(model, trainloader, optimizer, epochs, reg=None):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    reg = None

    # Get the snapshot of weights when training starts, if regularization is on
    if reg is not None:
        old_weights = flatten_weights(extract_weights_cuda(model))
        old_weights = torch.from_numpy(old_weights)

    for epoch in range(1, epochs + 1):
        for batch_id, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add regularization
            if reg is not None:
                new_weights = flatten_weights(extract_weights_cuda(model))
                new_weights = torch.from_numpy(new_weights)
                mse_loss = nn.MSELoss(reduction='sum')
                l2_loss = rou/2 * mse_loss(new_weights, old_weights)
                l2_loss = l2_loss.to(torch.float32)
                loss += l2_loss

            loss.backward()
            optimizer.step()

            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))

    if reg is not None:
        logging.info(
            'loss: {} l2_loss: {}'.format(loss.item(), l2_loss.item()))
    else:
        logging.info(
            'loss: {}'.format(loss.item()))
    return loss.item()

def test(model, testloader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(  # pylint: disable=no-member
                outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
