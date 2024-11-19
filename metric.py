from dataset import CharDataset
from model import CharModel
from dataloader import binary_dataloaders
from loss import loss_function
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR,ExponentialLR
from torchvision import transforms
from tqdm import tqdm
import os
import time
from sklearn.metrics import accuracy_score
import numpy as np
import copy
import torch
import json

def evaluate_multiclass(model, eval_loader, loss_function, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in tqdm(eval_loader):
            if labels[0] == 20:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, head_type="multiclass")
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(eval_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy

def evaluate_binary(model, eval_loaders, loss_function, device, binary_index=None):
    model.eval()
    total_loss = []
    accuracies = []
    with torch.no_grad():
        #for ind, eval_loader in tqdm(enumerate(eval_loaders)):
        for ind in range(20):
            loader_loss = 0
            all_preds = []
            all_targets = []
            acc = []
            FP_list = []
            FP_list_label = []
            FN_list = []
            if binary_index is None or ind == binary_index:
                loop = tqdm(enumerate(eval_loaders[ind]), total=len(eval_loaders[ind]))
                for i, (images, labels) in loop:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images, head_type="binary")[:, ind].unsqueeze(1) #取第i个二分类器的输出
                    targets = (labels==ind).float().unsqueeze(1).to(device) #构造target
                    loss = loss_function(outputs, targets)
                    loader_loss += loss.item()
                    #print(outputs) if i==1 else None
                    predicted = (outputs>0.5)
                    #print(predicted) if i==1 else None
                    #print(targets)
                    for j, pred in enumerate(predicted):
                        if pred == 1 and targets[j] == 0:
                            FP_list.append(outputs[j].cpu().numpy())
                            FP_list_label.append(labels[j].cpu().numpy())
                            #print("pred: ", pred, "target: ", labels[j])
                        if pred == 0 and targets[j] == 1:
                            FN_list.append(outputs[j].cpu().numpy())
                    acc.append(accuracy_score(targets.cpu().numpy(), predicted.cpu().numpy()))
                    #print(accuracy_score(targets.cpu().numpy(), predicted.cpu().numpy()))
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    # print(all_preds) if i==10 else None
                    # print(all_targets) if i==10 else None
                FP_list = np.array(FP_list)
                FN_list = np.array(FN_list)
                FP_list_label = np.array(FP_list_label)
                print("FP_list: ", FP_list)
                print("FP_list_label: ", FP_list_label)
                print("FN_list: ", FN_list)
            avg_loader_loss = loader_loss / len(eval_loaders[ind])
            total_loss.append(avg_loader_loss)
            # accuracy = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0
            accuracy = np.mean(acc) if len(acc) > 0 else 0
            #print(accuracy)
            accuracies.append(accuracy)



    return total_loss, accuracies  # 返回所有二分类器平均准确率