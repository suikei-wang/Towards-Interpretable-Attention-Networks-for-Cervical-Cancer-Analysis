import torch.nn as nn
from torchvision import models

def initialize_model(num_classes):
    '''
    Initialize the pre-trained ResNet50 model
    '''
    model_ft = None
    input_size = 224
    use_pretrained = False
    model_ft = models.resnet50(pretrained=use_pretrained)
    num_ftrs = model_ft.fc.in_features

    # modify the output feature dimension
    model_ft.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model_ft, input_size