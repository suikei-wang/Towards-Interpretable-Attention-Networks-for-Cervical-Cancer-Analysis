import numpy as np
import cv2
import torch
import os
from torchvision import transforms

from models.AttentionDenseNet import *

def feature_extractor(model, img):
    dim = (input_size, input_size)
    img = torch.Tensor(cv2.resize(img, dim))

    img_batch = np.expand_dims(img,axis=0)
    img_batch = torch.reshape(torch.Tensor(img_batch), (1, 3, input_size, input_size))

    conv_img = model(torch.Tensor(img_batch))        
    conv_img = np.squeeze(conv_img, axis=0)
    conv_vector = conv_img.detach().numpy()
    return conv_vector


if __name__ == '__main__':
    # load the original model but remove the fc layer
    model, input_size = initialize_model(num_classes=5)
    model.load_state_dict(torch.load('weights/model5', map_location=torch.device('cpu')))
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    dir_path = "separated-data"
    data_type = ["train", "val", "test"]
    classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
    
    for data in data_type:
        for cell in classes:
            cell_path = os.path.join(dir_path, data, cell)
            files = os.listdir(cell_path)
            files = [os.path.join(cell_path, f) for f in files if f.endswith('.bmp')]
            for file in files:
                img = cv2.imread(file)
                feature_vector = feature_extractor(model, img)
                print(feature_vector)