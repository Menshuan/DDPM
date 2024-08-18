import ForwardProcess as FP
import torch.nn.functional as F
import torch
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights  # 引入VGG16的权重

device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建感知损失类
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=2):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:feature_layer]
        self.vgg = vgg.eval()  # 使用VGG16的前几层特征
        for param in self.vgg.parameters():
            param.requires_grad = False  # 冻结VGG的参数，不进行训练

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return F.l1_loss(x_vgg, y_vgg)

perceptual_loss = PerceptualLoss().to(device)

# 损失函数
def p_losses(denoise_model,
             x_start,
             t,
             sqrt_alphas_cumprod,
             sqrt_one_minus_alphas_cumprod,
             noise=None,
             loss_type="l1",
             use_perceptual_loss=False):

    # 如果没有提供噪声，就创建一个与x_start形状相同的随机噪声；
    if noise is None:
        noise = torch.randn_like(x_start)

    # 通过前向过程获得某一时间步的时候的加噪图像；
    x_noisy = FP.q_sample(x_start=x_start,
                          t=t,
                          sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                          sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                          noise=noise)
    # 让模型输出其结果；
    predicted_noise = denoise_model(x_noisy, t)

    # 根据不同type设计loss下降；
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    if use_perceptual_loss:
        perceptual_loss_value = perceptual_loss(predicted_noise, noise)
        loss += perceptual_loss_value

    return loss

# DataLoader的创立
class CIFARdataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        super(CIFARdataset, self).__init__()
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.image_files = \
            [os.path.join(self.dataset_folder, file) for file in os.listdir(self.dataset_folder)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(str(image_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 创建dataloader
def makeDataLoader(datapath="G:/Python/Miyazaki/Miyazaki_Stable Diffusion/dataset"):    # 数据集路径
    dataset = CIFARdataset(datapath, transform=FP.image2tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

