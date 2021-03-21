from densenet_pytorch import DenseNet
import torch.nn as nn

def initialize_model(num_classes):
    input_size = 224 
    
    model_ft = DenseNet.from_pretrained("densenet121")
    num_ftrs = model_ft.classifier.in_features

    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model_ft, input_size