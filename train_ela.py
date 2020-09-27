from torch.utils.data import Dataset

import jpeg
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision import datasets

from PIL import Image
import torch.utils.data
import os


class ELA(nn.Module):
    def forward(self, doc, aut):
        b = doc.size(0)
        aut_jpeg = jpeg.jpeg_compress_decompress(aut, factor=0.9)
        ela_aut = (aut - aut_jpeg).reshape((b, -1))

        doc_jpeg = jpeg.jpeg_compress_decompress(doc, factor=0.9)
        ela_doc = (doc - doc_jpeg).reshape((b, -1))
        return nn.CosineSimilarity()(ela_aut, ela_doc)


class PairDataset(Dataset):
    def __init__(self, image1_paths, image2_paths, transform=None):
        self.image1_paths = image1_paths
        self.image2_paths = image2_paths
        self.transform = transform

    def __getitem__(self, index):
        img1 = Image.open(self.image1_paths[index])
        img2 = Image.open(self.image2_paths[index])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, self.image1_paths[index].split('/')[-1]

    def __len__(self):
        return len(self.image1_paths)


def fgsm(net, criterion, x, y, targeted=True, eps=7.65, x_val_min=0., x_val_max=255.):
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


DATA_DIR = 'patches'
aut_fnames = os.listdir(DATA_DIR + '/authentic')
aut_fnames.sort()
auts = list(map(lambda x: DATA_DIR + '/authentic/' + x, aut_fnames))
doc_fnames = os.listdir(DATA_DIR + '/doctored')
doc_fnames.sort()
docs = list(map(lambda x: DATA_DIR + '/doctored/' + x, doc_fnames))

transform = transforms.Compose([transforms.ToTensor()])

data = PairDataset(auts, docs, transform=transform)
train_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

device = torch.device("cpu")

ela = ELA().to(device)

for index, (img1, img2, fname) in enumerate(train_loader):
    # get the inputs
    img1 = Variable(img1 * 255.)
    img2 = Variable(img2 * 255.)

    x_adv, h_adv, h = fgsm(nn.Identity(), ela, img2, img1)
    Image.fromarray(x_adv.detach()[0].permute((1, 2, 0)).numpy().astype(np.uint8))\
        .save(DATA_DIR + '/attack/' + fname[0], 'JPEG', quality=100, subsampling=0)