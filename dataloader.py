import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np
from dataset import CharDataset
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, label_index, batch_size, drop_last=False, rondom_seed=22):
        '''
        数据均衡化的batch采样器，保证每个batch中正负样本数量均衡
        :param dataset: 数据集
        :param label_index: 标签索引，即需要针对哪个标签的二分类器
        :param batch_size: batch大小
        '''
        np.random.seed(rondom_seed)
        self.dataset = dataset
        self.label_index = label_index
        self.batch_size = batch_size
        self.drop_last = drop_last  # 是否丢弃最后一个不完整的batch

        self.positive_indices = np.where(np.array(dataset.labels) == label_index)[0]
        self.negative_indices = np.where(np.array(dataset.labels) != label_index)[0]
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

    def __iter__(self):
        positive_iter = iter(self.positive_indices)
        negative_iter = iter(self.negative_indices)

        batch = []
        while True:
            try:
                for _ in range(self.batch_size // 2): #一半正样本
                    batch.append(next(positive_iter))
                for _ in range(self.batch_size // 2): #一半负样本
                    batch.append(next(negative_iter))


                yield batch
                batch = []
            except StopIteration:
                if batch and not self.drop_last: #处理最后一个batch
                   yield batch
                break


    def __len__(self):
        num_positive = len(self.positive_indices)
        num_negative = len(self.negative_indices)
        num_samples = min(num_positive, num_negative)*2
        num_batches = num_samples // self.batch_size

        if not self.drop_last and num_samples % self.batch_size > 0:
            num_batches +=1


        return num_batches


def binary_dataloaders(dataset, batch_size, num_binary_classifiers=20, drop_last=False):
    '''
    返回20个二分类器的dataloader的list
    :param dataset: 数据集
    :param batch_size: batch大小
    :param num_binary_classifiers: 二分类器的数量，默认为20
    :return: train_loader_20class, train_loaders_binary 其中train_loaders_binary是二分类的DataLoader列表    
    '''


    train_loaders_binary = []
    for i in range(num_binary_classifiers):
        sampler = BalancedBatchSampler(dataset, label_index=i, batch_size=batch_size, drop_last=drop_last)
        train_loader_binary = DataLoader(dataset, batch_sampler=sampler) # 注意这里用batch_sampler
        train_loaders_binary.append(train_loader_binary)

    return train_loaders_binary




if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据模型调整大小
        transforms.ToTensor(),
    ])

    train_dataset = CharDataset("round0_train", transform=transform, preprocess=True)
    train_loader_20class = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 20分类，随机打乱
    train_loaders_binary = binary_dataloaders(train_dataset, batch_size=32)

    # 测试一下二分类 DataLoader：
    for i in range(3):
        images, labels = next(iter(train_loaders_binary[i])) #测试label=3的dataloader
        print(f"Binary Classifier {i}: ", images.shape, labels.shape)
        print(f"Binary Classifier {i}: ", labels)

    for i in range(3): #测试20分类的dataloader
        images, labels = next(iter(train_loader_20class ))
        print(f"20 Classifier : ", images.shape, labels.shape)
