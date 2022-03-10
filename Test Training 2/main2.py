import argparse
from dataset import DogCat
import ShuffleNet2
import torch as t
import torch.nn as nn
from torch.utils import data
import time
	args = vars(ap.parse_args())
	path = args["datapath"]
	train_sign = args["train"]
	epochs = args["epochs"]

	batchsize = args["batchsize"]
	dataloader = {}
	if train_sign:
		train_dataset = DogCat("./dogvscat/train", train=True)
		
		train_loader = data.DataLoader(train_dataset,
	                               batch_size = batchsize,
	                               shuffle=True)
		dataloader["train"] = train_loader
	
	val_dataset = DogCat("./dogvscat/train", train=False, test=False)                               
	val_loader = data.DataLoader(val_dataset,
                             batch_size = batchsize,
                             shuffle=True)
	dataloader["val"] = val_loader

	use_gpu = args["use_gpu"]
	if use_gpu:
		if t.cuda.is_available():
			device = t.device("cuda")
		else:
			print("You don't have gpu")

	else:
		device = t.device("cpu")
	# device = t.device("cuda" if t.cuda.is_available() else "cpu")
	model_path = args["pretrained"]

	num_classes = args["classes"]
	input_size = args["inputsize"]
	net_type = args["nettype"]
	model_type = args["model"]
	
model = ShuffleNet2(num_classes, input_size, net_type)
model = model.to(device)



loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)



def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
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
                    outputs = model(inputs) # bsize * 2 , because it is a binary classification
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
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)    
    return model, val_acc_history



model, val_logs = train_model(model, dataloader, loss_fn, optimizer)




torch.save(model.state_dict(), "./save/" + str(int(time.time()))+'.pkl')

