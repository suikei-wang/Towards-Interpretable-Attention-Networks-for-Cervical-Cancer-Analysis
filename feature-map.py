import numpy as np
import cv2
import torch
from torchvision import transforms

from models.AttentionDenseNet import *

# load the original model but remove the fc layer
model, input_size = initialize_model(num_classes=5)
model.load_state_dict(torch.load('weights/model5', map_location=torch.device('cpu')))
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# test an image
test = cv2.imread('separated-data/test/im_Koilocytotic/001.bmp')
dim = (input_size, input_size)
test = cv2.resize(test, dim)
test = torch.Tensor(test)
def feature_visualization(model, img):
    '''prints the cat as a 2d array'''
    img_batch = np.expand_dims(img,axis=0)
    img_batch = torch.reshape(torch.Tensor(img_batch), (1, 3, input_size, input_size))

    conv_img = model(torch.Tensor(img_batch))
    conv_img = np.squeeze(conv_img, axis=0)
    conv_vector = conv_img.detach().numpy()
    return conv_vector


feature_vector = feature_visualization(model, test)
