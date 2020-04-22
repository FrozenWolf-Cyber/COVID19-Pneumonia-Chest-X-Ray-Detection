import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import torchvision
from PIL import ImageFile
import matplotlib.pyplot as plt
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

PATH_TRAIN = 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'
EPOCHS = 20
BATCH_SIZE = 32
TOTAL_SIZE = len(os.listdir(PATH_TRAIN+"/NORMAL")) + len(os.listdir(PATH_TRAIN+"/INFECTED"))
STEPS_PER_EPOCH = TOTAL_SIZE // BATCH_SIZE
IMAGE_H , IMAGE_W = 224 , 224

transform = torchvision.transforms.Compose(
    [  # Applying Augmentation
        torchvision.transforms.Resize((IMAGE_H, IMAGE_W)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)  # Normalizing data

# Intitalizing the train data loader

train_dataset = torchvision.datasets.ImageFolder(
    root=PATH_TRAIN, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True
)

# Intitalizing the test data loader

test_dataset = torchvision.datasets.ImageFolder(
    root=PATH_TEST, transform=test_transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True
)

model_ft = torchvision.models.vgg16_bn(pretrained=True)  # Initializing vgg16

num_features = model_ft.classifier[6].in_features
features = list(model_ft.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
model_ft.classifier = nn.Sequential(*features)  # Replace the model classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft.to(device)  # Sending model to device
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model_ft.parameters(), lr=0.0007
)  # lr should be kept low so that the pre-trained weights don't change easily


loss_history = []

for i in range(EPOCHS):
    start = time.time()
    print("-----------------------EPOCH "+ str(i)+ " -----------------------------------" )
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == STEPS_PER_EPOCH:
            break
        optimizer.zero_grad()  # Resetting gradients
        # Sending input , target to device
        data = data.to(device)
        target = target.to(device)
        output = model_ft(data)
        loss = criterion(output, target.reshape((BATCH_SIZE,)).long())
        loss_history.append(loss.detach())
        loss.backward()
        optimizer.step()  # Optimizing the model

        # Checking train accuracy

        correct = 0
        incorrect = 0
        for p in range(BATCH_SIZE):
            a = []
            for j in output[p]:
                a.append(float(j.detach()))

            pred = a.index(max(a))

            if pred == int(target[p]):
                correct = correct + 1

            else:

                incorrect = incorrect + 1

        print("EPOCH "+ str(i)+ " MINIBATCH: "+ str(batch_idx) + "/" + str(STEPS_PER_EPOCH)+ " LOSS: "+ str(loss_history[-1])+ " ACC: " + str(correct / (incorrect + correct)),end="\r")

    end = time.time()
    print("EPOCH "+ str(i)+ " LOSS "+ str(sum(loss_history[-STEPS_PER_EPOCH:]) / STEPS_PER_EPOCH) + " ETA: "+ str(start - end)+ " \n MAX LOSS: " + str(max(loss_history[-STEPS_PER_EPOCH:]))+ " MIN LOSS: " + str(min(loss_history[-STEPS_PER_EPOCH:])))

plt.plot(loss_history)

def get_test():
    test_loss = []
    correct = 0
    incorrect = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx == STEPS_PER_TEST_EPOCH:
            break

        # Model is used to predict the test data so we are switching off the gradient

        with torch.no_grad():

            data = data.to(device)
            target = target.long().to(device)
            output = model_ft(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)

            for i in range(TEST_BATCH_SIZE):
                a = []
                for j in output[i]:
                    a.append(float(j.detach()))

                pred = a.index(max(a))

                if pred == int(target[i]):
                    correct = correct + 1

                else:
                    incorrect = incorrect + 1

        test_loss.append(float(loss.detach()))
    print("CORRECT:- " + str(correct), "INCORRECT:- " + str(incorrect))
    return (
        "TEST_ACC : "
        + str(correct / (correct + incorrect))
        + " TEST_LOSS:- "
        + str(float(sum(test_loss) / len(test_loss)))
    )
    
get_test() # Checking accuracy of model with test data
