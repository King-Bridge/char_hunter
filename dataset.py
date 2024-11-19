import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


class CharDataset(Dataset):
    def __init__(self, root_dir, transform=None, preprocess=True, train=True):
        '''
        :param root_dir: 数据集根目录
        :param transform: 转化为tensor并归一化
        :param preprocess: 是否预处理
        :param train: 是否为训练集
        '''
        random.seed(22)
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        self.train = train
        l = 20 if train else 21  # 0-19为训练集，20为测试集

        for label in range(l):
            folder_path = os.path.join(root_dir, f"{label:02}")
            for filename in os.listdir(folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):  
                    self.image_paths.append(os.path.join(folder_path, filename))
                    self.labels.append(label)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.preprocess:
            image = self._preprocess(image)
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def _preprocess(self, image):
        # 2. 去除灰色边框
        if self.train:
            image = image.crop((random.randint(3, 5), random.randint(3, 5), 
                                image.width - random.randint(3, 5), image.height - random.randint(3, 5)))
        else:
            image = image.crop((3, 3, image.width - 3, image.height - 3))
        # 3， 添加随机宽度的白色边框，达到移动字符位置的效果
        if self.train:
            border_width = random.randint(1, 100)  # 随机生成1到5之间的整数作为边框宽度
            border_height = random.randint(1, 100)  # 随机生成1到5之间的整数作为边框宽度
            # 创建一个与原图像大小相同的白色图像
            border_image = Image.new("RGB", (image.width + border_width, image.height + border_height), (255, 255, 255))
            # 将原图像粘贴到白色图像的中心
            center = (random.randint(0, border_width), 
                      random.randint(0, border_height))
            border_image.paste(image, center)
            image = border_image
        # 1. 大小转为224*224
        image = image.resize((224, 224))
        # 3. 锐化 (二值化)
        image_np = np.array(image)
        image_cp = image_np.copy()
        threshold = 128  

        image_np[image_cp >= threshold] = 255
        if self.train:
            #image_np[image_np >= threshold] = random.randint(128, 255)
            mask = np.random.rand(image_np.shape[0], image_np.shape[1]) < 0.2
            image_np[mask] = random.randint(128, 255)
              
        image_np[image_cp < threshold] = 0
        image = Image.fromarray(image_np.astype(np.uint8))        
        return image


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
    ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # 转换为Tensor，并归一化到[0, 1]
    ])

    dataset = CharDataset("round0_train", transform=transform, preprocess=True, train=True)
    print(len(dataset))
    image, label = dataset[0]
    image_np = (image.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

    # 如果是灰度图像，squeeze掉通道维度
    if image_np.shape[2] == 1:
        image_np = image_np.squeeze()

    # 显示图像
    plt.imshow(image_np, cmap='gray') # 使用灰度颜色映射
    plt.title(f"Label: {label}")
    plt.axis('off') # 关闭坐标轴
    plt.show()


    # 保存图像
    plt.imsave("preprocessed_image.png", image_np, cmap='gray')
    
