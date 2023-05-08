import os
import json
import torch
import logging
from glob import glob
from typing import Optional, Sequence

from robustbench.data import CORRUPTIONS, load_cifar10c, load_cifar100c, load_cifar10
from robustbench.loaders import CustomImageFolder, CustomCifarDataset
import matplotlib.pyplot as plt
import numpy as np
import cv2


logger = logging.getLogger(__name__)


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
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = x_test

    return samples

def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    # load imagenet class to id mapping from robustbench
    with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
        class_to_idx = json.load(f)

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = os.path.join(data_dir, corruptions_seq[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting:
        files = []
        for cor in corruptions_seq:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            file_paths = glob(os.path.join(corruption_dir_path, "*", "*.JPEG"))
            files += [(fp, class_to_idx[fp[len(str(corruption_dir_path))+1:].split(os.sep)[0]]) for fp in file_paths]
        dataset_test.samples = files
    elif setting == "correlated" or n_examples != -1:
        # get all test samples of the specified corruption
        file_paths = glob(os.path.join(str(corruption_dir_path), "*", "*.JPEG"))
        files = [(fp, class_to_idx[fp[len(str(corruption_dir_path))+1:].split(os.sep)[0]]) for fp in file_paths]
        dataset_test.samples = files

    return dataset_test

def create_cifar10c_folder():
    os.makedirs("./samples", exist_ok=True)
    for corruption in CORRUPTIONS:
        if not os.path.exists(f"./samples/{corruption}"):
            os.mkdir(f"./samples/{corruption}")
        for severity in range(1, 6):
            if not os.path.exists(f"./samples/{corruption}/{severity}"):
                os.mkdir(f"./samples/{corruption}/{severity}")
                os.mkdir(f"./samples/{corruption}/{severity}/x")
            # save samples in dir with name of corruption and folder inside as the severity
            samples = create_cifarc_dataset(corruption=corruption, severity=severity)
            print(samples.shape)
            for i in range(len(samples)):
                cv2.imwrite(f"./samples/{corruption}/{severity}/x/{i}.png", samples[i]*255)
            
    
    # for i in range(10):
    #     print(samples[0].shape)
    #     cv2.imwrite(f"./samples/corruptions_{i}.png", samples[1][i]*255)

        # plt.imshow(samples[1][i])
        # plt.savefig(f"./samples/clean_{i}.png")
        # plt.close()
    
    # for i in range(10):
    #     f, ax = plt.subplots(4, 4, figsize=(10, 10))
    #     for j in range(16):
    #         ax[j//4, j%4].imshow(samples[j][i])
    #         ax[j//4, j%4].axis("off")
    #         ax[j//4, j%4].set_title(cors[j])
        # plt.savefig(f"./samples/sample_{i}.png")
        # plt.close()

if __name__ == "__main__":
    create_cifar10c_folder()