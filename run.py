import torch
import torch.nn as nn
import torch.optim as optim
import os 
import numpy as np
import matplotlib.pyplot as plt

from models.ResNet50 import initialize_model as ResNet50
from models.Pretrained import initialize_model as Pretrained 
from models.AttentionResnet import *
from models.DenseNet import initialize_model as DenseNet

from dataLoader import get_dataloaders
from train_evaluate import train_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = 'separated-data'                     # baseline1, model1, model2
# data_dir = 'separated-cropped-data'           # baseline2, model3
# data_dir = 'seg-separated-data'               # Baseline4, model4
# data_dir = "separated-smear"                  # generalization test


num_classes = 5
batch_size = 16
shuffle_datasets = True
num_epochs = 50
save_dir = "weights"
os.makedirs(save_dir, exist_ok=True)
save_all_epochs = True


model, input_size = ResNet50(num_classes = num_classes)    # baseline1 & baseline2 & baseline4
# model, input_size = Pretrained(num_classes = num_classes)    # generalization test
# model, input_size = ResidualAttentionModel_92(num_classes = num_classes)    # model1
# model, input_size = DenseNet(num_classes = num_classes)    # model2
# model, input_size = DenseNet(num_classes = num_classes)    # model3
# model, input_size = DenseNet(num_classes = num_classes)    # model4


model = model.to(device)
dataloaders, class_name = get_dataloaders(input_size, batch_size, shuffle_datasets)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
print("Training progress")
print("=" * 20)
trained_model, train_losses, train_acc, val_losses, val_acc = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, save_dir=save_dir, save_all_epochs=save_all_epochs, num_epochs=num_epochs)

# save the model
torch.save(trained_model.state_dict(), "weights/baseline1")
# torch.save(trained_model.state_dict(), "weights/baseline2")
# torch.save(trained_model.state_dict(), "weights/baseline3")
# torch.save(trained_model.state_dict(), "weights/generlization")
# torch.save(trained_model.state_dict(), "weights/model1")
# torch.save(trained_model.state_dict(), "weights/model2")
# torch.save(trained_model.state_dict(), "weights/model3")
# torch.save(trained_model.state_dict(), "weights/model4")

# plot loss and accuracy
print()
print("Plots of loss and accuracy during training")
print("=" * 20)

x = np.arange(0,50,1)
plt.plot(x, train_losses, label='Training loss')
plt.plot(x, val_losses, label='Validation loss')
plt.legend(frameon=False)
plt.title("Pre-trained Resnet50")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(x, train_acc, label='Training accuracy')
plt.plot(x, val_acc, label='Validation accuracy')
plt.legend(frameon=False)
plt.title("Pre-trained Resnet50")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()