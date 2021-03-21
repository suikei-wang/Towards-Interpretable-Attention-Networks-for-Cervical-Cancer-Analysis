import torch
import torch.nn as nn
from torchvision import models

def initialize_model(num_classes):
    input_size = 224 
    use_pretrained = True
    
    model_ft = models.resnet50(pretrained=use_pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 5)
    )

    model_ft.load_state_dict(torch.load('weights/cropped-resnet50'))

    model_ft.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model_ft, input_size