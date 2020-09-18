import torchvision.transforms as transforms
from torchvision import datasets
import torch
from cnn import CNN
from cnn import train_net
from patch_extractor import PatchExtractor

# pe = PatchExtractor(input_path='dataset_slicing_all', output_path='patches_with_rot',
#                     patches_per_image=2, stride=32, rotations=4, mode='rot')
# pe.extract_patches()


torch.manual_seed(0)
DATA_DIR = "patches_with_rot"
transform = transforms.Compose([transforms.ToTensor()])

data = datasets.ImageFolder(root=DATA_DIR, transform=transform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) == "cuda:0":
    print("cuda")
    cnn = CNN().cuda()
else:
    print("cpu")
    cnn = CNN()

epoch_loss, epoch_accuracy = train_net(cnn, data, n_epochs=400, learning_rate=0.0001, batch_size=128)
torch.save(cnn.state_dict(), 'pretrained/cnn.pt')
