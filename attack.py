import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torch.utils.data
from cnn import CNN


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
for i, (inputs, labels) in enumerate(train_loader):
    # get the inputs
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda().long())
    else:
        inputs = Variable(inputs)
        labels = Variable(labels)
    inputs = F.unfold(inputs, (128, 128), stride=256).permute(2, 0, 1).reshape(-1, 3, 128, 128) # window
    labels = labels.repeat(inputs.size(0))
    fgsm(cnn, nn.CrossEntropyLoss(), inputs, labels)