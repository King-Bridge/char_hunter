from dataset import CharDataset
from model import CharModel
from dataloader import binary_dataloaders
from loss import loss_function
from metric import evaluate_multiclass, evaluate_binary

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR,ExponentialLR
from torchvision import transforms
from tqdm import tqdm
import os
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
import json



multi_learning_rate = 1e-4
binary_learning_rate = 1e-4
multi_gamma, multi_alpha = 1, 0.5
binary_gamma, binary_alpha = 1, 0.6
num_epochs_pretrain = 20
num_epochs_binary = 5
eval_every = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone_name = "resnet50"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 数据集
train_dataset = CharDataset("round0_train", transform=transform, preprocess=True, train=True)
eval_dataset = CharDataset("round0_eval", transform=transform, preprocess=True, train=False)

# 创建 DataLoader
train_loader_20class = DataLoader(train_dataset, batch_size=40, shuffle=True)
eval_loader_20class = DataLoader(eval_dataset, batch_size=100, shuffle=False)  # 验证集不需要 shuffle
train_loaders_binary = binary_dataloaders(train_dataset, batch_size=40)
eval_loaders_binary = binary_dataloaders(eval_dataset, batch_size=100)

model = CharModel(backbone_name = backbone_name, pretrained=True) #初始时不冻结backbone
model.to(device)


###########################################################################
# TODO：调整损失函数的参数。调节gamma，如果gamma=0，alpha=0。5，则损失函数退化为交叉熵损失函数. gamma越大，代表对于难样本的惩罚越大。
######################################################

loss_fn_multiclass = loss_function("focal_loss", multi_gamma, multi_alpha)
loss_fn_binary = loss_function("focal_loss", binary_gamma, binary_alpha)




# 训练历史记录
history = {
    'multi_learning_rate' : multi_learning_rate,
    'binary_learning_rate' : binary_learning_rate,
    'multi' : [multi_gamma, multi_alpha],
    'binary' : [binary_gamma, binary_alpha],
    "pretrain_train_loss": [],
    "pretrain_eval_loss": [],
    "pretrain_eval_acc": [],
    "binary_train_loss": [],
    "binary_eval_loss": [],
    "binary_eval_acc": [],
    'pretrain_lr':[],
    'binary_lr':[]
}

# --- 预训练 20 分类头 ---
print("Pretraining 20-class classifier...")
# 创建优化器
optimizer = Adam(model.parameters(), lr=multi_learning_rate, weight_decay=1e-4)

#学习率衰减
scheduler = ExponentialLR(optimizer, gamma=0.9) 
for epoch in range(num_epochs_pretrain):
    model.train()
    loop = tqdm(enumerate(train_loader_20class), total=len(train_loader_20class))
    train_loss = 0
    for i, (images, labels) in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, head_type="multiclass")
        loss = loss_fn_multiclass(outputs, labels)

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs_pretrain}]")
            loop.set_postfix(loss=loss.item())
        train_loss += loss.item()
    history['pretrain_lr'].append(optimizer.param_groups[0]['lr'])
    scheduler.step()
    history["pretrain_train_loss"].append(loss.item()/len(train_loader_20class))

    if (epoch + 1) % eval_every == 0:
        eval_loss, eval_acc = evaluate_multiclass(model, eval_loader_20class, loss_fn_multiclass, device)
        print(f"Evaluation (Epoch {epoch + 1}): Train Loss = {train_loss/len(train_loader_20class):.4f}  Eval Loss = {eval_loss:.4f}, Accuracy = {eval_acc:.4f}")
        history["pretrain_eval_acc"].append(eval_acc)
        history["pretrain_eval_loss"].append(eval_loss)
        # if epoch // eval_every >= 2 and eval_acc < history["pretrain_eval_acc"][-2]:
        #     # 早停策略
        #     print("Pretrain Model saved!")
        #     break

if num_epochs_pretrain > 0:
    torch.save(model.state_dict(), f"pretrained_{backbone_name}_{multi_gamma}_{multi_alpha}.pth")


if num_epochs_binary + num_epochs_pretrain != 0:
    with open(f"train_logs_{multi_gamma}_{multi_alpha}_{binary_gamma}_{binary_alpha}.json", "w") as f:
        json.dump(history, f, indent=4)


# --- 训练 20 个二分类头 ---
print("Training binary classifiers...")
model.load_state_dict(torch.load(f"pretrained_{backbone_name}_{multi_gamma}_{multi_alpha}.pth"))
#冻结backbone参数
for param in model.backbone.parameters():
    param.requires_grad = False


for i in range(20):
    loss_logs = []
    acc_logs = []
    eval_loss_logs = []
    print(f"Training binary classifier {i+1}/20...")
    optimizer_binary = Adam(model.classifiers_binary[i].parameters(), lr=binary_learning_rate) #每个二分类器单独优化
    scheduler_binary = ExponentialLR(optimizer_binary, gamma=0.85) 
    for epoch in range(num_epochs_binary):
        model.train()
        loop = tqdm(enumerate(train_loaders_binary[i]), total=len(train_loaders_binary[i]))
        train_loss = 0
        for k, (images, labels) in loop:
            images, labels = images.to(device), labels.to(device)
            targets = (labels==i).float().unsqueeze(1).to(device)
            #print(targets)
            optimizer_binary.zero_grad()
            outputs = model(images, head_type="binary", binary_index=i)
            # if k == 0:
            #     print(model(images, head_type="binary", binary_index=i))
            loss = loss_fn_binary(outputs, targets) 
            loss.backward()
            if i == 0:
                history['binary_lr'].append(optimizer.param_groups[0]['lr'])
            optimizer_binary.step()
            if k % 10 == 0:
                loop.set_description(f"Epoch [{epoch + 1}/{num_epochs_binary}]")
                loop.set_postfix(loss=loss.item())
            train_loss += loss.item()
        loss_logs.append(train_loss/len(train_loaders_binary[i]))
        
        scheduler_binary.step()
        if (epoch + 1) % eval_every == 0:
            eval_loss, eval_acc = evaluate_binary(model, eval_loaders_binary, loss_fn_binary, device, i) 
            print(f"The {i}th binary Evaluation (Epoch {epoch + 1}): Train Loss = {train_loss/len(train_loaders_binary[i]):.4f}, Eval Loss = {eval_loss[i]:.4f}, Accuracy = {eval_acc[i]:.4f}, mean accuracy = {np.mean(eval_acc):.4f}")
            acc_logs.append(eval_acc[i])
            eval_loss_logs.append(eval_loss[i])
            # if epoch // eval_every >= 2 and eval_acc[i] < acc_logs[-2]:
            #     print(f"The {i}th Binary Model saved!")
            #     break
    
    history["binary_eval_acc"].append(acc_logs)   
    history["binary_train_loss"].append(loss_logs)
    history["binary_eval_loss"].append(eval_loss_logs)



# 保存训练历史记录和最终模型
if num_epochs_binary + num_epochs_pretrain != 0:
    torch.save(model.state_dict(), f"final_model_{multi_gamma}_{multi_alpha}_{binary_gamma}_{binary_alpha}.pth")
    with open(f"train_logs_{multi_gamma}_{multi_alpha}_{binary_gamma}_{binary_alpha}.json", "w") as f:
        json.dump(history, f, indent=4)


# 加载最终模型
model.load_state_dict(torch.load(f"final_model_{multi_gamma}_{multi_alpha}_{binary_gamma}_{binary_alpha}.pth"))
# 验证最终结果
eval_loss, eval_acc = evaluate_binary(model, eval_loaders_binary, loss_fn_binary, device)
for i in range(20): 
    print(f"Final Evaluation of {i}th Binary Classifier: Loss = {eval_loss[i]:.4f}, Accuracy = {eval_acc[i]:.4f}")







# # 绘制 loss 曲线
# plt.plot(history["pretrain_loss"])
# plt.title("Pretraining Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.savefig("pretrain_loss.png")
# plt.close()

# plt.plot(history["pretrain_eval_acc"])
# plt.title("Pretraining evaluation accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.savefig("pretrain_acc.png")
# plt.close()

# plt.plot(history["pretrain_lr"])
# plt.title("Pretraining learning rate")
# plt.xlabel("Epoch")
# plt.ylabel("Learning rate")
# plt.savefig("pretrain_lr.png")
# plt.close()


# plt.plot(history["binary_loss"])
# plt.title("Binary Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.savefig("binary_loss.png")
# plt.close()

# plt.plot(history["binary_eval_acc"])
# plt.title("Binary evaluation accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.savefig("binary_acc.png")
# plt.close()

# plt.plot(history["binary_lr"])
# plt.title("Binary learning rate")
# plt.xlabel("Epoch")
# plt.ylabel("Learning rate")
# plt.savefig("binary_lr.png")
# plt.close()
