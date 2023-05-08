import os
import json
import torch
import logging
from glob import glob
from typing import Optional, Sequence

from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c
from robustbench.loaders import CustomImageFolder, CustomCifarDataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

def read_images(severity = 5, corruption_seq = ["gaussian_noise"]):
    '''Check if images are read correctly, 
    and the ordering of the dimension
    and the number of images read'''

    path = '../../DDA/dataset/generated/'           # path to the folder containing the images
    images = []
    for cor in corruption_seq:
        path_temp = os.path.join(path, cor, str(severity), 'x')
        filenames = os.listdir(path_temp)
        filenames = [int(filename.split('.')[0]) for filename in filenames]
        filenames = sorted(filenames)
        mask = np.array(filenames) 
        filenames = [str(filename) + '.png' for filename in filenames]
        for filename in filenames:
            img = cv2.imread(os.path.join(path_temp,filename))
            if img is not None:
                images.append(img)
    images = np.array(images)
    return images, mask


def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    domain = []
    x_test = torch.tensor([])
    x_test_diffu = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
            x_tmp_diffu, mask = read_images(severity, [cor])
            y_tmp = y_tmp[mask]
            x_tmp = x_tmp[mask]
            
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")
        
        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        x_test_diffu = torch.cat([x_test_diffu, torch.from_numpy(x_tmp_diffu)], dim=0)
        domain += [cor] * x_tmp.shape[0]
    x_test = x_test.permute(0, 2, 3, 1)
    x_test = x_test.unsqueeze(0)
    x_test_diffu = x_test_diffu/255
    x_test_diffu = x_test_diffu.unsqueeze(0)
    x_test_final = torch.cat([x_test, x_test_diffu], dim=0)
    # f, ax = plt.subplots(1, 2)
    # ind = -1
    # ax[0].imshow(x_test_final[0, ind].numpy())
    # ax[1].imshow(x_test_final[1, ind].numpy())
    # plt.savefig('test.png')
    # exit()
    x_test_final = x_test_final.numpy()
    y_test = y_test.numpy()
    x_test_final = x_test_final.transpose(1, 0, 2, 3, 4)
    # plot the cifar10c images and diffusion images side-by-side and save the sequence as a gif
    for i in range(100):
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(x_test_final[i, 0])
        ax[1].imshow(x_test_final[i, 1])
        plt.savefig('test.png')
        plt.close()
        img = cv2.imread('test.png')
        img = cv2.resize(img, (256, 256))
        if i == 0:
            out = img.copy()
        else:
            out = np.concatenate((out, img), axis=1)
    # save as .gif using PIL images
    out = Image.fromarray(out)
    out.save('test.gif')
    exit()
    


if __name__ == "__main__":
    create_cifarc_dataset()