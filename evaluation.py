import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as score

from models.ResNet50 import initialize_model as ResNet50
from models.Pretrained import initialize_model as Pretrained 
from models.DenseNet import initialize_model as DenseNet
from dataLoader import get_dataloaders
from train_evaluate import evaluate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# evaluation
print("Evaluation on validation and test set")
print("=" * 20)

num_classes = 5
# num_classes = 7   # for smear dataset
batch_size = 16
criterion = nn.CrossEntropyLoss()

model, input_size = ResNet50(num_classes)
model.load_state_dict(torch.load('weights/baseline1'))
# model.load_state_dict(torch.load('weights/baseline2'))
# model.load_state_dict(torch.load('weights/baseline3'))
# model.load_state_dict(torch.load('weights/baseline4'))
# model.load_state_dict(torch.load('weights/model1'))
model = model.to(device)
generate_validation_labels = True
dataloaders, class_name = get_dataloaders(input_size, batch_size, True)

val_loss, val_top1, val_top5, val_labels = evaluate(model, dataloaders['val'], criterion, is_labelled = True, generate_labels = generate_validation_labels, k = 5)

epoch_loss, top1_acc, top5_acc, test_labels = evaluate(model, dataloaders['test'], criterion, is_labelled = True, generate_labels = True, k = 5)
print("Top 1 accuracy on test set is", top1_acc)

# Get the confusion matrix from test
confusion_matrix = {x: [0,0,0,0,0] for x in class_name}
# confusion_matrix = {x: [0,0,0,0,0,0,0] for x in class_name}   for smear dataset

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