import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import transforms
from PIL import Image

#write a dataset class for cifar10c
class Cifar10cDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, list_corruptions, list_severity):
        self.root_dir = root_dir
        self.list_corruptions = list_corruptions
        self.imgs = []
        for corruption in list_corruptions:
            for severity in list_severity:
                for img in sorted(os.listdir(os.path.join(root_dir, corruption, str(severity), 'x'))):
                    img_path = os.path.join(root_dir, corruption, str(severity), 'x',img)
                    self.imgs.append((img_path))
        self.tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.normalize_transform(self.tensor_transform(img))
        #get path after root
        img_path = img_path.split(self.root_dir)[1]
        return img, img_path