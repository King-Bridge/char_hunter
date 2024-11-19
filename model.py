import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, googlenet # 引入 ResNet50 和 GoogLeNet

# 其他backbone可以在这里导入，例如:
from efficientnet_pytorch import EfficientNet  # EfficientNet
import timm  #timm库，包含vit等SOTA模型


class CharModel(nn.Module):
    def __init__(self, backbone_name, freeze=False, pretrained=False, fc_layer=False, num_classes=20):  
        '''
        返回多分类器的logits，或者所有二分类器的概率concat之后的结果
        backbone_name: 预训练的backbone模型
        freeze: 是否冻结backbone参数
        num_classes: 二分类任务类别数
        '''
        super(CharModel, self).__init__()
        self.backbone_name = backbone_name
        self.fc_l = fc_layer

        # 冻结backbone参数（可选）,在预训练20分类头时不冻结，在训练二分类头时冻结
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False


        if self.backbone_name == "resnet50":
            self.backbone = resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features #获取backbone输出的特征维度
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1])) #去除resnet最后的fc层
            hidden_dim1 = 1024
            hidden_dim2 = 512
        elif self.backbone_name == "resnet18":
            self.backbone = resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
            hidden_dim1 = 512
            hidden_dim2 = 256
        # elif self.backbone_name == "googlenet":
        #     self.backbone = googlenet(pretrained=pretrained)
        #     in_features = self.backbone.fc.in_features
        #     self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))

        # elif backbone_name == "efficientnet-b0": #示例，按需添加
        #     self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        #     in_features = self.backbone.fc.in_features
        #     self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        # elif backbone_name == "vit_base_patch16_224":  # 使用timm
        #     self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        #     in_features = self.backbone.head.in_features
        #     self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        # # ... 其他 backbone ...
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        #print(in_features)
        # 20分类头
        self.classifier_20 = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),  # 可以根据需要调整隐藏层大小
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim1, hidden_dim2),  
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim2, num_classes),           
        )

        if self.fc_l:
            in_features = 1024
        # 20个二分类头 (用 ModuleList 更方便管理)
        self.classifiers_binary = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),  # 可以根据需要调整或移除dropout
                nn.Linear(256, 1),
                nn.Sigmoid()  # 二分类使用 Sigmoid
            ) for _ in range(num_classes)
        ])


    def forward(self, x, head_type="multiclass", binary_index=None):
        features = self.backbone(x).flatten(start_dim=1)
        #print(features.shape)
        if head_type == "multiclass":
            logits = self.classifier_20(features)
            return logits  # 注意返回值是logits，而不是概率，所以后续需要用softmax
        elif head_type == "binary":
            if self.fc_l:
                # 输入变为20分类器的第一层MLP之后的结果
                features = self.classifier_20[0](features)               
            if binary_index is not None:
                binary_result = self.classifiers_binary[binary_index](features)
                return binary_result
            else:
                binary_result = [head(features) for head in self.classifiers_binary]
                #print(binary_result)
                return torch.cat(binary_result, dim=1)  # 合并所有二分类器的输出
        else:
            raise ValueError("Invalid head_type. Choose 'multiclass' or 'binary'.")



if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import time
    model = CharModel(backbone_name="resnet18", pretrained=True)

    path = r'round0_eval\00\label00_200.png'
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = torch.from_numpy(img).float() / 255.0

    dummy_input = img.unsqueeze(0)
    print(time.time())
    multiclass_output = model(dummy_input, head_type="multiclass")
    print(time.time())
    binary_output = model(dummy_input, head_type="binary")
    print(time.time())
    print("Multiclass output shape:", multiclass_output)
    print("Binary output shape:", binary_output)