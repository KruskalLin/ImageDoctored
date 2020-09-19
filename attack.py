import json
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
from torchvision import datasets
import torch.utils.data
from cnn import CNN
import os

def fgsm(net, criterion, x, y, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
    x_adv = Variable(x.data, requires_grad=True)
    h_adv = net(x_adv)
    if targeted:
        # target label
        cost = criterion(h_adv, y)
    else:
        # no target label, true label
        cost = -criterion(h_adv, y)

    net.zero_grad()
    if x_adv.grad is not None:
        x_adv.grad.data.fill_(0)
    cost.backward()

    x_adv.grad.sign_()
    x_adv = x_adv - eps * x_adv.grad
    x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

    h = net(x)
    h_adv = net(x_adv)

    return x_adv, h_adv, h


DATA_DIR = 'testing'
img_fnames = os.listdir(DATA_DIR + '/authentic')
img_fnames.sort()

transform = transforms.Compose([transforms.ToTensor()])

data = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) == "cuda:0":
    print("cuda")
    cnn = CNN().cuda()
else:
    print("cpu")
    cnn = CNN()

cnn.load_state_dict(torch.load('pretrained/cnn.pt'))
cnn.train()

for index, (inputs, labels) in enumerate(train_loader):
    # get json file
    if index < len(img_fnames):
        continue
    i = index - len(img_fnames)
    configs = json.load(open('configs/' + img_fnames[i].split('.')[0][1:] + '.json', 'r'))

    x = inputs.clone()
    # get the inputs
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda().long())
    else:
        inputs = Variable(inputs)
        labels = Variable(labels)


    for config in configs:
        point = config['point']
        if point[1] + 128 > inputs.size(2) or point[0] + 128 > inputs.size(3):
            continue
        x_adv, h_adv, h = fgsm(cnn, nn.CrossEntropyLoss(), inputs[:, :, point[1]:point[1] + 128, point[0]:point[0] + 128], labels)
        x[:, :, point[1]:point[1] + 128, point[0]:point[0] + 128] = x_adv

    Image.fromarray((x[0, :].detach().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))\
        .save('images/' + img_fnames[i], quality=97, subsampling=0)
