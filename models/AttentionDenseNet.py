from densenet_pytorch import DenseNet
import torch.nn as nn
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def initialize_model(num_classes):
    input_size = 224 
    
    model_ft = DenseNet.from_pretrained("densenet121")
    num_ftrs = model_ft.classifier.in_features

    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    model_ft.features.denseblock4.denselayer16 = nn.Sequential(
        model_ft.features.denseblock4.denselayer16,
        CALayer(channel=32)
)
    return model_ft, input_size

