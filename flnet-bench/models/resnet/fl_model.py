import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy

# Training settings
lr = 0.001
momentum = 0.9
log_interval = 10
rou = 0.01
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
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
            ]))
        self.testset = datasets.CIFAR10(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
            ]))
        self.labels = list(self.trainset.classes)

def Net():
    # print("ResNet loaded!")
    cifar_cnn = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
                                  width_per_group=64, replace_stride_with_dilation=None)
    return cifar_cnn

class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_optimizer(model):
    # return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    return optim.Adam(model.parameters(), lr=lr)


def get_trainloader(trainset, batch_size, idxs):
    # return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return DataLoader(CustomDataset(trainset, idxs), batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    # return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return DataLoader(testset, batch_size=batch_size)


def extract_weights(model):
    # w = model.state_dict()
    w = model.to(torch.device('cpu')).state_dict()
    return w
    
    # weights = []
    # for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
    #     if weight.requires_grad:
    #         weights.append((name, weight.data))
    # return weights

def extract_model(model):
    return copy.deepcopy(model)
    
def extract_weights_old(model):
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
    # updated_state_dict = {}
    # for name, weight in weights:
    #     updated_state_dict[name] = weight

    model.load_state_dict(weights)
    
def load_model(model, weights):
    cur_idx = 0
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data.copy_(
            weights[cur_idx : cur_idx + numel].view(size)
        )
        cur_idx += numel
    
def load_weights_old(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)

    # model.load_state_dict(weights)

def flatten_weights(weights):
    # Flatten weights into vectors
    weight_vecs = []
    for _, weight in weights:
        weight_vecs.extend(weight.flatten().tolist())

    return np.array(weight_vecs)

def cos_sim_test(model, trainloader, _model):
    model.to(eval)
    _model.to(eval)
    
    cosine_sim_sum = 0.0
    num_samples = 0
    
    for batch_id, data in enumerate(trainloader):
        inputs, labels = data
        
        output1 = model(inputs)
        output2 = _model(inputs)
        
        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)
        
        cosine_sim = F.cosine_similarity(output1, output2, dim=1)
        cosine_sim_sum += cosine_sim.sum().item()
        num_samples += cosine_sim.size(0)
        
    mean_cosine_sim = cosine_sim_sum / num_samples
    
    return mean_cosine_sim
        

def train(model, trainloader, optimizer, epochs, reg=None):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # Get the snapshot of weights when training starts, if regularization is on
    if reg is not None:
        model_ = copy.deepcopy(model)
        model_.to(device)
        
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    e_loss = []

    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        for batch_id, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data
            
            if inputs.size()[0] < 2:
                continue
            
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add regularization
            if reg is not None:
                l2_reg = 0.0
                for p1, p2 in zip(model_.parameters(), model.parameters()):
                    l2_reg += (p1 - p2).norm(2)
                loss = loss + l2_reg * (rou / 2)
                # new_weights = flatten_weights(extract_weights_cuda(model))
                # new_weights = torch.from_numpy(new_weights)
                # mse_loss = nn.MSELoss(reduction='sum')
                # l2_loss = rou/2 * mse_loss(new_weights, old_weights)
                # l2_loss = l2_loss.to(torch.float32)
                # loss += l2_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))
                
        train_loss = train_loss / len(trainloader.dataset)
        e_loss.append(train_loss)
    # if reg is not None:
    #     logging.info(
    #         'loss: {} l2_loss: {}'.format(loss.item(), l2_loss.item()))
    # else:
    logging.info('loss: {}'.format(loss.item()))

    total_loss = sum(e_loss) / len(e_loss)
    
    return total_loss

def test(model, testloader):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
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


# import load_data
# import logging
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# import numpy as np

# # Training settings
# lr = 0.01
# momentum = 0.9
# log_interval = 10
# rou = 1
# loss_thres = 0.01

# # Cuda settings
# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')


# class Generator(load_data.Generator):
#     """Generator for CIFAR-10 dataset."""

#     def read(self, path):
#         self.trainset = datasets.CIFAR10(
#             path, train=True, download=True, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ]))
#         self.testset = datasets.CIFAR10(
#             path, train=False, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ]))
#         self.labels = list(self.trainset.classes)


# # Replace the `Net` class with ResNet
# def Net(num_classes=10):
#     model = models.resnet18(pretrained=False)  # Use ResNet18
#     model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer for CIFAR-10
#     return model


# def get_optimizer(model):
#     return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# def get_trainloader(trainset, batch_size):
#     return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


# def get_testloader(testset, batch_size):
#     return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# def extract_weights(model):
#     weights = []
#     for name, weight in model.to(torch.device('cpu')).named_parameters():
#         if weight.requires_grad:
#             weights.append((name, weight.data))

#     return weights


# def extract_weights_cuda(model):
#     weights = []
#     for name, weight in model.named_parameters():
#         if weight.requires_grad:
#             weights.append((name, weight.data))

#     return weights


# def load_weights(model, weights):
#     updated_state_dict = {}
#     for name, weight in weights:
#         updated_state_dict[name] = weight

#     model.load_state_dict(updated_state_dict, strict=False)


# def flatten_weights(weights):
#     weight_vecs = []
#     for _, weight in weights:
#         weight_vecs.extend(weight.flatten().tolist())

#     return np.array(weight_vecs)


# def train(model, trainloader, optimizer, epochs, reg=None):
#     model.to(device)
#     model.train()
#     criterion = nn.CrossEntropyLoss()

#     if reg is not None:
#         old_weights = flatten_weights(extract_weights_cuda(model))
#         old_weights = torch.from_numpy(old_weights)

#     for epoch in range(1, epochs + 1):
#         for batch_id, data in enumerate(trainloader):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             if reg is not None:
#                 new_weights = flatten_weights(extract_weights_cuda(model))
#                 new_weights = torch.from_numpy(new_weights)
#                 mse_loss = nn.MSELoss(reduction='sum')
#                 l2_loss = rou / 2 * mse_loss(new_weights, old_weights)
#                 l2_loss = l2_loss.to(torch.float32)
#                 loss += l2_loss

#             loss.backward()
#             optimizer.step()

#             if batch_id % log_interval == 0:
#                 logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(epoch, epochs, loss.item()))

#     if reg is not None:
#         logging.info('loss: {} l2_loss: {}'.format(loss.item(), l2_loss.item()))
#     else:
#         logging.info('loss: {}'.format(loss.item()))
#     return loss.item()


# def test(model, testloader):
#     model.to(device)
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = correct / total
#     logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

#     return accuracy