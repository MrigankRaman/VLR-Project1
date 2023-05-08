
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

    path = '../../DDA/dataset/generate_corrected_cifar10c/'           # path to the folder containing the images
    images = []
    for cor in corruption_seq:
        path_temp = os.path.join(path, cor, str(severity), 'x')
        filenames = os.listdir(path_temp)
        filenames = [int(filename.split('.')[0]) for filename in filenames]
        filenames = sorted(filenames)
        mask = np.array(filenames) 
        filenames = [str(filename) + '.png' for filename in filenames]
        for filename in filenames:
            img_cv = cv2.imread(os.path.join(path_temp,filename))
            #read image using PIL and convert to np array 
            # img = np.array(Image.open(os.path.join(path_temp,filename)))
            # import ipdb; ipdb.set_trace()
            img = img_cv
            # import ipdb; ipdb.set_trace()
            if img is not None:
                images.append(img)
    images = np.array(images)
    return images, mask


def create_diffu_cifarc_dataset(
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
        if dataset_name == 'cifar10_cd':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
            x_tmp_diffu, mask = read_images(severity, [cor])
            # import ipdb; ipdb.set_trace()

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
    # import ipdb; ipdb.set_trace()
    # f, ax = plt.subplots(1, 2)
    # ind = -1
    # ax[0].imshow(x_test_final[0, ind].numpy())
    # ax[1].imshow(x_test_final[1, ind].numpy())
    # plt.savefig('test.png')
    # exit()
    x_test_final = x_test_final.numpy()
    y_test = y_test.numpy()
    x_test_final = x_test_final.transpose(1, 0, 2, 3, 4)
    samples = [[x_test_final[i], y_test[i], domain[i]] for i in range(len(x_test_final))]
    return CustomCifarDataset(samples=samples, transform=transform, isDiffusion=True)


def create_diffu_imagenetc_dataset(
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
    dataset_test = CustomImageFolder(corruption_dir_path, transform, isDiffusion=True)

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

    # get the folders that are present in the specified corruption directory
    folder_paths = glob(os.path.join(str(corruption_dir_path), "*"))
    # get the filenames of the images in the specified corruption directory
    filenames = []
    for folder_path in folder_paths:
        filenames += glob(os.path.join(folder_path, "*.JPEG"))
    dic = {f:1 for f in filenames}
    print(f"Number of test samples: {len(dataset_test.samples)}")
    print(f"Number of test samples in {corruption_dir_path}: {len(dic)}")
    # in dataset_test.samples retain only the samples that are present in the specified corruption directory
    dataset_test.samples = [s for s in dataset_test.samples if s[0] in dic]
    print(f"Number of test samples: {len(dataset_test.samples)}")

    return dataset_test

if __name__ == "__main__":
    create_cifarc_dataset()