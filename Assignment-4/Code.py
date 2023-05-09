import torch
import os
import sys
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import time


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].

# Apply necessary image transfromations here 

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
set_all_seeds(0)

# root_path='/home/ravi_k/scratch/WORKING_DIR/Code/A4_Vision/group_1'
root_path=sys.argv[1]
train_dir = os.path.join(root_path,"train") # put path of training dataset
val_dir =   os.path.join(root_path,"valid") # put path of test dataset
test_dir =  os.path.join(root_path,"test") # put path of test dataset

train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,shuffle=True, num_workers=2)

val_set = torchvision.datasets.ImageFolder(root=val_dir,transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4,shuffle=True, num_workers=2)

test_set = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,shuffle=False, num_workers=2)


num_classes=len(find_classes(train_dir)[0])
num_epochs = 20         # desired number of training epochs
learning_rate = 0.001 
vgg_pretrained=None
# for param in model.parameters():
#     param.required_grad = False

try:
    vgg_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
except Exception as e:
    print("Unable to fetch pre-trained weights from internet")
    print("Using downloaded pre-trained weights")
    try:
        vgg_pretrained = torch.load("./vgg_16.pth")
    except Exception as e:
        print("Cannot find any downloaded weights")

# vgg_pretrained = torch.load("/home/ravi_k/scratch/WORKING_DIR/Code/A4_Vision/vgg_16.pth")

# Freeze training for all layers
for param in vgg_pretrained.features.parameters():
    param.require_grad = True #False

num_features = vgg_pretrained.classifier[6].in_features
features = list(vgg_pretrained.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, num_classes)]) # Add our layer with 25  outputs
vgg_pretrained.classifier = nn.Sequential(*features) # Replace the model classifier
# print(vgg_pretrained)


################### DO NOT EDIT THE BELOW CODE!!! #######################

os.makedirs('./models', exist_ok=True)



net=vgg_pretrained

# transfer the model to GPU
if torch.cuda.is_available():
    net = net.cuda()

########################################################################
# Train the network
# ^^^^^^^^^^^^^^^^^^^^

def train(epoch, train_loader, optimizer, criterion, model):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        # get the inputs
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('epoch %d training loss: %.3f' %(epoch + 1, running_loss / (len(train_loader))))

########################################################################
# Let us look at how the network performs on the test dataset.

def test(test_loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy=(100 * correct / total)
    print('Accuracy of the network on the test images: %f %%' % val_accuracy)
    return val_accuracy

#########################################################################
# get details of classes and class to index mapping in a directory



def classwise_test(test_loader, model):
########################################################################
# class-wise accuracy

    classes, _ = find_classes(train_dir)
    n_class = len(classes) # number of classes

    class_correct = list(0. for i in range(n_class))
    class_total = list(0. for i in range(n_class))
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c=(predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(n_class):
        print('Accuracy of %10s : %2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

print('Start Training')


def epoch_time(start_time,end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


criterion = nn.CrossEntropyLoss()
num_params = np.sum([p.nelement() for p in net.parameters()])
print("Number of parameters in the model are:"+str(num_params))

# Weight_decays=[1e-5,1e-4,1e-3,1e-2,1e-1,0]
# for wd in Weight_decays:
wd=1e-3
print("Weight decay Param is: "+str(wd))
# optimizer = optim.AdamW(net.parameters(),lr=learning_rate,weight_decay=wd)
optimizer=optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=wd)
best_val=0

for epoch in range(num_epochs):  # loop over the dataset multiple times
    start_time = time.time()
    print('epoch ', epoch + 1)
    train(epoch, train_loader, optimizer, criterion, net)
    print('Performing testing on validation data-set')
    val_acc=test(val_loader, net)
    print("Classwise Testing on validation data-set")
    classwise_test(val_loader, net)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    if(val_acc>best_val):
        best_val=val_acc
        torch.save(net,'./models/'+str(wd)+'_best_model.pth')

    # save model checkpoint 

print("Training is completed")
net=torch.load('./models/'+str(wd)+'_best_model.pth')
if torch.cuda.is_available():
    net = net.cuda()

print('Loaded best-model')
print('Performing final testing on validation data-set')
test(val_loader, net)
print("Classwise Testing on validation data-set")
classwise_test(val_loader, net)
print('Performing testing on test data-set')
test(test_loader, net)
print("Classwise Testing")
classwise_test(test_loader, net)
print('Finished Training')
