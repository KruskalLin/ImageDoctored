import json

import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets
import torch.utils.data
from cnn import CNN
import numpy as np
import os

DATA_DIR = 'testing'
img_fnames = os.listdir(DATA_DIR + '/authentic')
img_fnames.sort()
transform = transforms.Compose([transforms.ToTensor()])

data = datasets.ImageFolder(root=DATA_DIR, transform=transform)
eval_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) == "cuda:0":
    print("cuda")
    cnn = CNN().cuda()
else:
    print("cpu")
    cnn = CNN()

cnn.load_state_dict(torch.load('pretrained/cnn.pt'))
cnn.eval()
total_predicted = []
total_labels = []
for i, (inputs, labels) in enumerate(eval_loader):
    # get json file
    if i >= 20:
        break
    configs = json.load(open('configs/' + img_fnames[i].split('.')[0][1:] + '.json', 'r'))

    # get the inputs
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda().long())
    else:
        inputs = Variable(inputs)
        labels = Variable(labels)

    xs = []
    predicteds = []
    for config in configs:
        point = config['point']
        xs.append(inputs[:, :, point[1]:point[1] + 128, point[0]:point[0] + 128])

    for x in xs:
        outputs = cnn(x)
        _, predicted = torch.max(outputs.data, 1)
        predicteds.append(predicted.max())

    total_labels.extend(labels)
    total_predicted.append(max(predicteds))
predictions = (np.array(total_predicted) == np.array(total_labels)).sum().item()
print(predictions / len(total_predicted))