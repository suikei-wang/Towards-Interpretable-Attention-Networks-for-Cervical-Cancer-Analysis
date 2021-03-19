import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import os
import copy
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
from sklearn.metrics import precision_recall_fscore_support as score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def get_dataloaders(input_size, batch_size, shuffle = True):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(25),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.RandomRotation(25),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_transforms.keys()}

    class_names = image_datasets['train'].classes
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle, num_workers=4) for x in data_transforms.keys()}
    return dataloaders_dict, class_names

def train_model(model, dataloaders, criterion, optimizer, save_dir = None, save_all_epochs=False, num_epochs=25):
    '''
    model: The NN to train
    dataloaders: A dictionary containing at least the keys 
                 'train','val' that maps to Pytorch data loaders for the dataset
    criterion: The Loss function
    optimizer: The algorithm to update weights 
               (Variations on gradient descent)
    num_epochs: How many epochs to train for
    save_dir: Where to save the best model weights that are found, 
              as they are found. Will save to save_dir/weights_best.pt
              Using None will not write anything to disk
    save_all_epochs: Whether to save weights for ALL epochs, not just the best
                     validation error epoch. Will save to save_dir/weights_e{#}.pt
    '''
    since = time.time()
    
    train_losses = []
    train_acc = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
            
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, train_losses, train_acc

def evaluate(model, dataloader, criterion, is_labelled = False, generate_labels = True, k = 5):
    model.eval()
    running_loss = 0
    running_top1_correct = 0
    running_top5_correct = 0
    predicted_labels = []
    
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        tiled_labels = torch.stack([labels.data for i in range(k)], dim=1) 

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            if is_labelled:
                loss = criterion(outputs, labels)
            _, preds = torch.topk(outputs, k=5, dim=1)
            if generate_labels:
                nparr = preds.cpu().detach().numpy()
                predicted_labels.extend([list(nparr[i]) for i in range(len(nparr))])

        if is_labelled:
            running_loss += loss.item() * inputs.size(0)
            running_top1_correct += torch.sum(preds[:, 0] == labels.data)
            running_top5_correct += torch.sum(preds == tiled_labels)
        else:
            pass

    if is_labelled:
        epoch_loss = float(running_loss / len(dataloader.dataset))
        epoch_top1_acc = float(running_top1_correct.double() / len(dataloader.dataset))
        epoch_top5_acc = float(running_top5_correct.double() / len(dataloader.dataset))
    else:
        epoch_loss = None
        epoch_top1_acc = None
        epoch_top5_acc = None
    
    return epoch_loss, epoch_top1_acc, epoch_top5_acc, predicted_labels


if __name__ == '__main__':
    data_dir = "separated-smear"
    num_classes = 7
    batch_size = 16
    shuffle_datasets = True
    num_epochs = 50

    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    save_all_epochs = True
    save_all_epochs = True

    model, input_size = initialize_model(num_classes)
    dataloaders, class_name = get_dataloaders(input_size, batch_size, shuffle_datasets)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model!
    trained_model, train_losses, train_acc = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, save_dir=save_dir, save_all_epochs=save_all_epochs, num_epochs=num_epochs)

    torch.save(trained_model.state_dict(), "weights/baseline-smear")

    generate_validation_labels = True
    data_dir = 'separated-smear'
    epoch_loss, top1_acc, top5_acc, test_labels = evaluate(model, dataloaders['test'], criterion, is_labelled = True, generate_labels = True, k = 5)
    
    print("Top1 ACC:", top1_acc)

    # Get the confusion matrix from test
    confusion_matrix = {x: [0,0,0,0,0] for x in class_name}
    running_top1_correct = 0
    loader = dataloaders['test']
    labels_array = []
    pred_array = []
    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        
        
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.topk(outputs, k=1, dim=1)
        
        
        for i in range(len(labels)):
            original_label = int(labels[i])
            labels_array.append(original_label)
            pred_array.append(int(preds[i]))
            confusion_matrix[class_name[original_label]][int(preds[i])] += 1
            
        running_top1_correct += torch.sum(preds[:, 0] == labels.data)

    precision, recall, fscore, support = score(labels_array, pred_array)

    epoch_top1_acc = float(running_top1_correct.double() / len(loader.dataset))
    percentage = {x: [y /sum(confusion_matrix[x]) for y in confusion_matrix[x]] for x in confusion_matrix.keys()}
    print()
    print("Confusion matrix")
    print("=" * 20)
    print(percentage)
    print()
    print("Precision:", precision)
    print("Recall", recall)
    print("F1-Score", fscore)
    print("Support:", support)