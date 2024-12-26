import argparse
import pysam
import random
import numpy as np
import pandas as pd

import time
import os


import models
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


params = argparse.ArgumentParser(description='train model')
params.add_argument("--data",help="peaks file",required=True)
params.add_argument("--model",help="model name",default="Basset")
params.add_argument("--name",help="train name",required=True)
params.add_argument("--activate",help="activate loss",required=True)
params.add_argument("--seqlen",help="sequence length",required=True)
params.add_argument("--task",help="task number",required=True)
params.add_argument("--seed",help="random seed",default=1401,type=int)
params.add_argument("--device",help="device cuda",default="cuda:0")
params.add_argument("--fasta",help="fasta seq file",required=True)
params.add_argument("--epoch",help="epochs",default=100)
params.add_argument("--lr",help="learn rata",default=0.0001)
params.add_argument("--batch",help="batch size",default=256)
params.add_argument("--outpath",help="model name")
args = params.parse_args()



def set_random_seed(random_seed = 40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.seed = random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed) 
    torch.cuda.manual_seed_all(random_seed)


def pearson_loss(x,y):
        mx = torch.mean(x, dim=0, keepdim=True)
        my = torch.mean(y, dim=0, keepdim=True)
        xm, ym = x - mx, y - my
    
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        loss = torch.mean(1-cos(xm,ym))
        return loss

def pearson_r(x,y):
        mx = torch.mean(x, dim=0, keepdim=True)
        my = torch.mean(y, dim=0, keepdim=True)
        xm, ym = x - mx, y - my
    
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)


        return torch.mean(cos(xm,ym))

fasta_file = args.fasta

fasta = pysam.FastaFile(fasta_file)

class BinaryDataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]


task_name = args.name
batch_size = int(args.batch)
epochs = int(args.epoch)
data_ = np.load(args.data,allow_pickle=True)
device = args.device if torch.cuda.is_available() else "cpu"
task_num = args.task

set_random_seed(args.seed)

linear_units = utils.linear_units_dict[args.model]["%sbp"%args.seqlen]

# Create a model
if args.activate == "relu":
    model = eval(f"models.{args.model}({task_num},{linear_units},'relu')")
elif args.activate == "exp":
    model = eval(f"models.{args.model}({task_num},{linear_units},'exp')")
model.to(device)


#定义训练参数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=float(args.lr),eps=0.00001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=3,gamma=0.8)

model_name = "%s_%s_%s_%s_%s_repression"%(task_name,args.model,args.seed,args.seqlen,args.activate)
early_stopping = utils.EarlyStopping(save_path=args.outpath,model_name=model_name,patience=5)




# Create a DataLoader for the training set

train_dataset = BinaryDataset(data_["train_data"],data_["train_label"])
train_loader = DataLoader(train_dataset, batch_size=int(args.batch),shuffle=True,num_workers=8) 


# Create a DataLoader for the validation set
val_dataset = BinaryDataset(data_["test_data"],data_["test_label"])
val_loader = DataLoader(val_dataset, batch_size=int(args.batch), shuffle=False,num_workers=8)



# Set the number of epochs and early stopping
# model_name = '%s_%s_%s_model_%s.pt'%(args.model,"mse",args.seqlen,args.seed)
num_epochs = args.epoch
# early_stopping = utils.EarlyStopping(save_path = args.outpath,model_name=model_name,patience=3,verbose=True, delta=0)


# Create a list to store the training loss
train_loss_list = []

# Create a list to store the validation loss
val_loss_list = []

# create a list to store the train time
time_train_list = []

# Train the model

##calculate the time
start = time.time()
for epoch in range(num_epochs):
    train_loss = 0
    for index,(sequences, coverage) in enumerate(train_loader):
        sequences = sequences.float().to(device).permute(0,2,1)
        coverage = coverage.float().to(device).unsqueeze(1)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = F.softplus(model(sequences))
        # print(outputs,coverage)
        # print(outputs,coverage)
        # Compute the loss
        # loss = criterion(torch.log2(outputs+1), torch.log2(coverage+1))
        loss = criterion(torch.log2(outputs+1), torch.log2(coverage+1))

        train_loss += loss.item()
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Print the loss every 100 batches
        if index % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{index+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Pearson: {pearson_r(torch.log2(outputs+1), torch.log2(coverage+1))}")
    # Compute the validation loss
    val_loss = 0
    val_pred = []
    val_true = []

    model.eval()
    with torch.no_grad():
        for sequences, coverage in val_loader:
            sequences = sequences.float().to(device).permute(0,2,1)
            coverage = coverage.float().to(device).unsqueeze(1)
            outputs = F.softplus(model(sequences))
            val_pred.extend(outputs.cpu().detach().numpy())
            val_true.extend(coverage.cpu().detach().numpy())
            loss = criterion(torch.log2(outputs+1), torch.log2(coverage+1))
            val_loss += loss.item()

    # Print the training and validation loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f},Pearson: {pearson_r(torch.log2(outputs+1), torch.log2(coverage+1))}")
    # Append the training and validation loss to the respective lists
    train_loss_list.append(train_loss/len(train_loader))
    val_loss_list.append(val_loss/len(val_loader))
    # Check if the validation loss has improved
    early_stopping(val_loss/len(val_loader), model)
    if early_stopping.early_stop:
        print("Early stopping")
        #save val_pred and val_true npz
        # np.savez(os.path.join(args.outpath,"%s_val.npz"%model_name),val_pred=np.vstack(val_pred),val_true=np.vstack(val_true))
        break
    np.savez(os.path.join(args.outpath,"%s_val.npz"%model_name),val_pred=np.vstack(val_pred),val_true=np.vstack(val_true))
    # Adjust the learning rate
    scheduler.step()


end = time.time()
print(f"Training time: {end-start}s")
