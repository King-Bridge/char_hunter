import torch
import torch.nn as nn
import torch.nn.functional as F


# Todo: 试一下不同gamma值的focal loss，以及试一试23行我添加的那一项会不会对模型有更好的效果
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, input, target):
        num_classes = input.shape[1]
        # 执行label smooth操作
        if input.shape[1] > 1: #多分类情况
            one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)
            smoothed_target = (1.0 - self.smoothing) * one_hot + self.smoothing / num_classes
            log_probs = F.log_softmax(input, dim=1)
            # print(torch.exp(log_probs))
            # print(torch.exp(log_probs).sum(dim=1))
            # loss = - (smoothed_target * log_probs).sum(dim=1) # 计算交叉熵

            pt = torch.exp(log_probs)

            focal_loss = -((1 - pt) ** self.gamma * (self.alpha * smoothed_target * log_probs))  # 应用 Focal Loss

        else:#二分类情况
            # Label Smoothing for Binary Classification
            smoothed_target = (1.0 - self.smoothing) * target + self.smoothing * (1-target)  # 由于是sigmoid输出，平滑方法略有不同
            pt = input
            focal_loss = -((1 - pt) ** self.gamma * (self.alpha * smoothed_target * torch.log(pt)) * target) - \
                (pt ** self.gamma * (1 - self.alpha) * torch.log(1 - pt)) * (1-target) # 添加了负样本的项。
            
        if input.shape[1] > 1: #多分类情况，使用softmax计算概率
            ce_loss = F.cross_entropy(input, target.long(), reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = -(1 - pt) ** self.gamma * (self.alpha * target * torch.log(pt)) 
            #print(pt)
        else: #二分类情况，已经是概率
            pt = input
            # print(pt)
            # print(target)
            focal_loss = -(1 - pt) ** self.gamma * (self.alpha * target * torch.log(pt)) * target - \
                (pt ** self.gamma * (1 - self.alpha) * torch.log(1 - pt)) * (1-target) # 添加了负样本的项。



        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # "none"
            return focal_loss


def loss_function(loss_type, gamma=2.0, alpha=0.25, num_classes=20, reduction="mean"):
    """
    创建一个损失函数。

    Args:
        loss_type: 损失函数类型，可以是 "focal_loss", "cross_entropy", 或 "bce"。
        gamma: Focal Loss 的 gamma 参数。表示对难样本的惩罚程度的“幂次”，若为0则相当于CE
        alpha: Focal Loss 的 alpha 参数。表示BCE权重，若为0.5则相当于CE
        num_classes: 类别数量（仅用于交叉熵损失）。
        reduction: 损失函数的 reduction 方法，可以是 "mean", "sum" 或 "none"。

    Returns:
        一个损失函数对象。
    """

    if loss_type == "focal_loss":
        return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(reduction=reduction)  # 注意：CrossEntropyLoss 包含了 Softmax
    elif loss_type == "bce":
        return nn.BCELoss(reduction=reduction) #注意：BCELoss需要输入概率值
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    

if __name__ == "__main__":

    # 多分类示例
    logits = torch.randn(2, 20)  # 2 个样本，20 个类别
    targets = torch.randint(0, 20, (2,))
    focal_loss_fn = loss_function("focal_loss", gamma=2, alpha=0.25, reduction="mean")
    focal_loss = focal_loss_fn(logits, targets)
    print("Focal Loss (Multi-class):", focal_loss)



    # 二分类示例
    binary_output = torch.randn(5, 1).sigmoid()  # 5 个样本，1 个输出 (概率)
    binary_targets = torch.randint(0, 2, (5,1)).float()
    binary_targets = torch.tensor([[0], [0], [0], [0], [0]]).float()
    print("Binary Targets:", binary_targets)
    print("Binary Output:", binary_output)
    binary_focal_loss_value = focal_loss_fn(binary_output, binary_targets)
    print("Focal Loss (Binary):", binary_focal_loss_value)
    bce_loss = loss_function("bce")
    binary_bce_loss_value = bce_loss(binary_output, binary_targets)
    print("BCE Loss (Binary):", binary_bce_loss_value)