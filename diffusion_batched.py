from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import ipdb
from tqdm import tqdm
from image_adapt.resize_right import resize
import copy
from dataset import Cifar10cDataset
#import dataloader 
import os

D = 1
scale = 1
shape_u = (1, 3, 32, 32)
shape_d = (1, 3, 32//D, 32//D)
w_diffusion = 0.9

preprocess = transforms.Compose(
    [
        # transforms.Resize((32, 32)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
tensor_transform = transforms.ToTensor()
normalize_transform = transforms.Normalize([0.5], [0.5])

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to("cuda")
# scheduler.set_timesteps(200)

sample_size = model.config.sample_size


# noise = transforms.ToTensor()(img).unsqueeze(0).to("cuda")

def get_diffuson_output(img, timesteps, img_path, save_path):
    noise = torch.randn((img.shape[0], 3, sample_size, sample_size)).to("cuda")
    noised_img = scheduler.add_noise(img, noise, timesteps)
    input = noised_img
    # print(scheduler.timesteps)
    # scheduler.set_timesteps(9)
    for t in (scheduler.timesteps):
        # with torch.no_grad():
        # import ipdb
        # ipdb.set_trace()
        input = input.requires_grad_()

        noisy_residual = model(input, t).sample
        output = scheduler.step(noisy_residual, t, input)
        prev_noisy_sample = output.prev_sample
        pred_original_sample = output.pred_original_sample
        prev_noisy_sample_copy = prev_noisy_sample.clone()

        difference = resize(resize(img, scale_factors=1.0/D, out_shape=shape_d), scale_factors=D, out_shape=shape_u) - resize(resize(pred_original_sample, scale_factors=1.0/D, out_shape=shape_d), scale_factors=D, out_shape=shape_u)
        norm = torch.linalg.norm(difference)
        norm = torch.linalg.norm(difference.reshape(-1,3*32*32), dim = 1)
        norm_grad = torch.autograd.grad(outputs=norm.sum(), inputs=input)[0]
        prev_noisy_sample -= norm_grad * scale

        input = w_diffusion * prev_noisy_sample_copy + (1 - w_diffusion) * prev_noisy_sample
        # input = prev_noisy_sample
        input = input.detach_()
    # exit()

    image_cpu = ((input.cpu().permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()
    for i in range(image_cpu.shape[0]):
        image = Image.fromarray(image_cpu[i])
        cur_path = img_path[i]
        cur_path = os.path.join(save_path,cur_path[1:])
        if not os.path.exists(os.path.dirname(cur_path)):
            os.makedirs(os.path.dirname(cur_path))
        image.save(cur_path)



#write dataloader 
list_corruptions = ['brightness', 'fog', 'contrast', 'frost']
# list_corruptions = ['defocus_blur', 'elastic_transform', 'glass_blur', 'impulse_noise']
# list_corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform']
# list_corruptions = ['pixelate', 'shot_noise', 'snow', 'zoom_blur']
# list_corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform']
# list_corruptions = ['fog', 'frost', 'gaussian_noise', 'glass_blur']
#list_corruptions = ['impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate']
# list_corruptions = ["brightness", "contrast"]
# list_corruptions = ["defocus_blur", "elastic_transform"]
# list_corruptions = ["glass_blur", "impulse_noise"]
list_severity = [1]
root_dir = "/home/scratch/rohans2/vlr_project/cifar10c/samples/"
dataset = Cifar10cDataset(root_dir, list_corruptions, list_severity)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
save_path = "/home/scratch/rohans2/vlr_project/cifar10c/generated/"
#iterate over dataloader
# noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
timesteps = torch.LongTensor([999])
for i, (img, img_path) in enumerate(tqdm(dataloader)):
    # print(img_path)
    # import ipdb
    # ipdb.set_trace()
    img = img.to("cuda")
    get_diffuson_output(img, timesteps, img_path, save_path)
