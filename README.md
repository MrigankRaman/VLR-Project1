# Diffusion based Online Test Time Adaptation
This is the implementation of our VLR project on online test-time adaptation repository based on PyTorch. It is joint work by Mrigank Raman, Akash Kannan, Pranit Chawla and Rohan Shah. It is based on the repository of [Robust Mean Teacher for Continual and Gradual Test-Time Adaptation](https://github.com/mariodoebler/test-time-adaptation) (CVPR2023)


## Prerequisites
To use the repository, we provide a conda environment.
```bash
cd test-time-adaptation
conda env create -f environment.yml
conda activate tta 
pip install diffusers
```

## Classification

<details open>
<summary>Features</summary>

This repository allows to study a wide range of different datasets, models, settings, and methods. A quick overview is given below:

- **Datasets**
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)

- **Models**
  - For adapting to ImageNet variations, all pre-trained models available in [Torchvision](https://pytorch.org/vision/0.14/models.html) or [timm](https://github.com/huggingface/pytorch-image-models/tree/v0.6.13) can be used.
  - For the corruption benchmarks, pre-trained models from [RobustBench](https://github.com/RobustBench/robustbench) can be used.

- **Settings**
  - `reset_each_shift` Reset the model state after the adaptation to a domain.
  - `continual` Train the model on a sequence of domains without knowing when a domain shift occurs.
  - `gradual` Train the model on a sequence of gradually increasing/decreasing domain shifts without knowing when a domain shift occurs.

- **Methods**
  - The repository currently supports the following methods: [TENT](https://openreview.net/pdf?id=uXl3bZLkr3c),
  [CoTTA](https://arxiv.org/abs/2203.13591), [AdaContrast](https://arxiv.org/abs/2204.10377)


- **Modular Design**
  - Adding new methods should be rather simple, thanks to the modular design.

</details>

### Get Started
To run one of the following benchmarks, the corresponding datasets need to be downloaded.
- *CIFAR10-to-CIFAR10-C*: the data is automatically downloaded.
- *ImageNet-to-ImageNet-C*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF).

Next, specify the root folder for all datasets `_C.DATA_DIR = "./data"` in the file `conf.py`. For the individual datasets, the directory names are specified in `conf.py` as a dictionary (see function `complete_data_dir_path`). In case your directory names deviate from the ones specified in the mapping dictionary, you can simply modify them.

### Run Experiments

We provide config files for all experiments and methods. Simply run the following Python file with the corresponding config file.
```bash
python test_time.py --cfg cfgs/[cifar10_c/cifar100_c/imagenet_c/imagenet_others/domainnet126]/[source/norm_test/norm_alpha/tent/memo/eta/eata/cotta/adacontrast/lame/sar/rotta/gtta/rmt].yaml
```

### Changing Configurations
Changing the evaluation configuration is extremely easy. For example, to run TENT on ImageNet-to-ImageNet-C in the `reset_each_shift` setting with a ResNet-50 and the `IMAGENET1K_V1` initialization, the arguments below have to be passed. 
Further models and initializations can be found [here (torchvision)](https://pytorch.org/vision/0.14/models.html) or [here (timm)](https://github.com/huggingface/pytorch-image-models/tree/v0.6.13).
```bash
python test_time.py --cfg cfgs/imagenet_c/tent.yaml MODEL.ARCH resnet50 MODEL.WEIGHTS IMAGENET1K_V1 SETTING reset_each_shift
```


### Acknowledgements
+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ LAME [official](https://github.com/fiveai/LAME)
+ MEMO [official](https://github.com/zhangmarvin/memo)
+ RoTTA [official](https://github.com/BIT-DA/RoTTA)
+ SAR [official](https://github.com/mr-eggplant/SAR)
