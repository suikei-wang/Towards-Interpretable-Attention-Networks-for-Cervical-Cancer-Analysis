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

    train_dys = []
    val_dys = []
    test_dys = []

    train_koi = []
    val_koi = []
    test_koi = []

    train_met = []
    val_met = []
    test_met = []

    train_par = []
    val_par = []
    test_par = []

    train_sup = []
    val_sup = []
    test_sup = []
    
    for data in data_type:
        for cell in classes:
            cell_path = os.path.join(dir_path, data, cell)
            files = os.listdir(cell_path)
            files = [os.path.join(cell_path, f) for f in files if f.endswith('.bmp')]
            for file in files:
                img = cv2.imread(file)
                feature_vector = feature_extractor(model, img)
                if data == data_type[0]: 
                    if cell == classes[0]:
                        train_dys.append(feature_vector)
                    elif cell == classes[1]:
                        train_koi.append(feature_vector)
                    elif cell == classes[2]:
                        train_met.append(feature_vector)
                    elif cell == classes[3]:
                        train_par.append(feature_vector)
                    elif cell == classes[4]:
                        train_sup.append(feature_vector)
                print("finish type 1")
                if data == data_type[1]: 
                    if cell == classes[0]:
                        val_dys.append(feature_vector)
                    elif cell == classes[1]:
                        val_koi.append(feature_vector)
                    elif cell == classes[2]:
                        val_met.append(feature_vector)
                    elif cell == classes[3]:
                        val_par.append(feature_vector)
                    elif cell == classes[4]:
                        val_sup.append(feature_vector)
                print("finish type 2")
                if data == data_type[2]: 
                    if cell == classes[0]:
                        test_dys.append(feature_vector)
                    elif cell == classes[1]:
                        test_koi.append(feature_vector)
                    elif cell == classes[2]:
                        test_met.append(feature_vector)
                    elif cell == classes[3]:
                        test_par.append(feature_vector)
                    elif cell == classes[4]:
                        test_sup.append(feature_vector)
                print("finish type 3")
    
    print(train_dys)