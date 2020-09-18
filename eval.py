import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets
import torch.utils.data
from cnn import CNN
import numpy as np

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
cnn.eval()
total_predicted = []
total_labels = []
for i, (inputs, labels) in enumerate(train_loader):
    # get the inputs
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda().long())
    else:
        inputs = Variable(inputs)
        labels = Variable(labels)
    inputs = F.unfold(inputs, (128, 128), stride=256).permute(2, 0, 1).reshape(-1, 3, 128, 128) # window
    outputs = cnn(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total_labels.extend(labels)
    total_predicted.extend([predicted.max()])
predictions = (np.array(total_predicted) == np.array(total_labels)).sum().item()
print(predictions / len(total_predicted))