from model import CharModel
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # 转换为Tensor，并归一化到[0, 1]
])

def preprocess(image):
    image = image.crop((3, 3, image.width - 3, image.height - 3))
    # 1. 大小转为224*224
    image = image.resize((224, 224))
    # 3. 锐化 (二值化)
    image_np = np.array(image)
    image_cp = image_np.copy()
    threshold = 128  
    image_np[image_cp >= threshold] = 255
    image_np[image_cp < threshold] = 0
    image = Image.fromarray(image_np.astype(np.uint8))        
    return image

def recognition(image, model, model_path):
    image_list = []
    for i in range(12):
        for j in range(12):
            image_list.append(image[i*50:(i+1)*50, j*50:(j+1)*50])
    backbone_name = 'resnet50'
    model = CharModel(backbone_name = backbone_name, pretrained=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    result = []
    pred = []
    for image in image_list:
        image = preprocess(image)
        image = transform(image)
        output = model(image, head_type="binary")
        pred.append(output)
        # _, predicted = torch.max(output, 1)
        # result.append(predicted.item())
    
    return result
