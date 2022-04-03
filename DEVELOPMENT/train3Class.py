import argparse
from dataset3Class import BSD
from ShuffleNet2 import ShuffleNet2
import torch as t
from torch.utils import data
import torch.nn as nn
import copy
import time
import numpy as np

def train_model(model, dataloaders, loss_fn, optimizer, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    best_loss = 100
    bes_count = 0
    count = 0
    val_acc_history = []
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            if phase == "train":
                model.train()
            else:
                model.eval()
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                with t.autograd.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels) 
                preds = outputs.argmax(dim=1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += t.sum(preds.view(-1) == labels.view(-1)).item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print("Phase {} loss: {}, acc: {}".format(phase, epoch_loss, epoch_acc))
            if phase == "val" and epoch_acc >= best_acc : #and epoch_loss <= best_loss
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                bes_count = bes_count + 1
                t.save(model.state_dict(), "BSD_Model_3C_Last" + str(count) + ".pkl")
                print("Best model Accuracy:", best_acc)
            if phase == "val":
                count = count + 1
                print("epoch #",count)
                val_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == '__main__':
    epochs = 50
    batchsize = 32
    num_classes = 3
    input_size = 256
    net_type = 2
    name = "4Class"
    trainfolder = "./train3Class"
    pkl_path = "BSD_Model_3C_"+ name +".pkl"
    dataloader = {}
    train_dataset = BSD(trainfolder, train=True)
    train_loader = data.DataLoader(train_dataset, batch_size = batchsize, shuffle=True)
    dataloader["train"] = train_loader
    val_dataset = BSD(trainfolder, train=False, test=False)                               
    val_loader = data.DataLoader(val_dataset, batch_size = batchsize, shuffle=True)
    dataloader["val"] = val_loader
    if t.cuda.is_available():
        device = t.device("cuda")
    else:
        device = t.device("cpu")
    print(device)
    model = ShuffleNet2(num_classes, input_size, net_type)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model, val_logs = train_model(model, dataloader, loss_fn, optimizer, epochs)
    t.save(model.state_dict(), pkl_path)
    print("Model saved to", pkl_path)
