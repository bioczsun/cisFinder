import argparse
import pysam
import random
import numpy as np
import pandas as pd
import collections
import warnings
warnings.filterwarnings("ignore")

from torchmetrics import AUROC


import models
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
print("Starting the script...")
params = argparse.ArgumentParser(description='train model')
params.add_argument("--data",help="peaks file",required=True)
params.add_argument("--name",help="train name",required=True)
params.add_argument("--model",help="model name",default="Basset")
params.add_argument("--activate",help="activate loss",required=True)
params.add_argument("--seqlen",help="sequence length",required=True)
params.add_argument("--task",help="task number",required=True)
params.add_argument("--seed",help="random seed",default=40,type=int)
params.add_argument("--device",help="device cuda",default="cuda:0")
params.add_argument("--epoch",help="epochs",default=500,type=int)
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



class BinaryDataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]

set_random_seed(args.seed)
task_name = args.name
batch_size = int(args.batch)
epochs = int(args.epoch)
data_ = np.load(args.data,allow_pickle=True)
device = args.device if torch.cuda.is_available() else "cpu"
task_num = int(args.task)

# Create a model
if args.activate == "relu":
    if args.model == "scBasset":
        model = models.scBasset(task_num)
    else:
        linear_units = utils.linear_units_dict[args.model]["%sbp"%args.seqlen]
        model = eval(f"models.{args.model}({task_num},{linear_units},'relu')")

if args.activate == "gelu":
    if args.model == "scBasset":
        model = models.scBasset(task_num)
    else:
        linear_units = utils.linear_units_dict[args.model]["%sbp"%args.seqlen]
        model = eval(f"models.{args.model}({task_num},{linear_units},'gelu')")

elif args.activate == "exp":
    linear_units = utils.linear_units_dict[args.model]["%sbp"%args.seqlen]
    model = eval(f"models.{args.model}({task_num},{linear_units},'exp')")
model.to(device)

# 初始化权重
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1  # 提高 momentum 值，加速均值和方差的更新

model.apply(initialize_weights)

#定义训练参数
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=float(args.lr),eps=0.00001,betas=(0.95,0.9995))

model_save_path = "%s_%s_%s_%s_%s"%(task_name,args.model,args.seed,args.seqlen,args.activate)
early_stopping = utils.EarlyStopping(save_path=args.outpath,model_name=model_save_path,patience=5)

auroc = AUROC(task="multilabel", average="macro", num_labels=task_num)  # 在循环外部初始化
auroc_test = AUROC(task="multilabel", average="macro", num_labels=task_num)  # 在循环外部初始化


# #计算标签的权重
# counter = collections.Counter(data_["train_label"])
# counter = dict(counter)
# weights = torch.tensor([int(counter[k]) for k in counter],dtype=torch.float) / len(data_["train_label"])
# samples_weights = weights[data_["train_label"]]
# sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights,len(samples_weights),replacement=True)


# Create a DataLoader for the training set

train_dataset = BinaryDataset(data_["train_data"],data_["train_label"])
train_loader = DataLoader(train_dataset, batch_size=int(args.batch),num_workers=5,shuffle=True) 


# Create a DataLoader for the validation set
val_dataset = BinaryDataset(data_["test_data"],data_["test_label"])
val_loader = DataLoader(val_dataset, batch_size=int(args.batch), shuffle=False,num_workers=5)


#define train
def train(model,train_dataloader,test_dataloader,optimizer,criterion,epochs,early_stopping,save_path,model_name,device):
    '''
    train model
    '''
    train_loss_ls = []
    test_loss_ls = []
    for epoch in range(epochs):
        model.train()
        total_step = len(train_dataloader)
        running_loss = 0.0
        train_auc = 0.0

        # train_label_ls = []
        # train_pred_ls = []
        # test_label_ls = []
        # test_pred_ls = []

        for index,(data,label) in enumerate(train_dataloader):

            ##get single batch data
            train_data =data.float().to(device).transpose(1,2)
            if task_num > 1:
                train_label = label.to(device)
            else:
                train_label = label.to(device).view(-1,1)


            optimizer.zero_grad()

            outputs = model(train_data)
            outputs = F.sigmoid(outputs)
            loss = criterion(outputs,train_label.float())

            # train_label_ls.append(train_label.int().detach())
            # train_pred_ls.append(outputs.detach())

            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            auroc.update(outputs, train_label.int())
            

                # train_auc += auroc

            if (index + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch + 1, epochs, index + 1, total_step, loss.item()))
        epoch_loss = running_loss / (len(train_dataloader))
        epoch_auc = train_auc / (len(train_dataloader))
        train_loss_ls.append(epoch_loss)

        epoch_auc = auroc.compute()
        auroc.reset()

        print("------------eval------------------------")
        test_total_step = len(test_dataloader)
        test_runing_loss = 0.0
        eval_auroc = 0.0
        with torch.no_grad():
            model.eval()
            for _,(data,label) in enumerate(test_dataloader):

                ##get single batch data
                test_data = data.float().to(device).transpose(1,2)
                if task_num > 1:
                    test_label = label.to(device)
                else:
                    test_label = label.to(device).view(-1,1)

                #forward
                outputs = model(test_data)
                outputs = F.sigmoid(outputs)
                auroc_test.update(outputs, test_label.int())
                # test_label_ls.append(test_label.int().detach())
                # test_pred_ls.append(outputs.detach())


                test_loss = criterion(outputs,test_label.float())
                test_runing_loss += test_loss.item()


        epoch_test_loss = test_runing_loss / test_total_step
        test_loss_ls.append(epoch_test_loss)

        eval_auroc = auroc_test.compute()
        auroc_test.reset() # 重置度量用于下一个 epoch


        print('当前Epoch [{}/{}], Train Loss: {:.4f},Train AUC: {:.4f}, Eval Loss: {:.4f},Eval AUC: {:.4f}'.format(epoch + 1, epochs, epoch_loss,epoch_auc,epoch_test_loss,eval_auroc))


        #early stopping and step scheduler
        early_stopping(-epoch_auc,model)

        # scheduler.step()

        #达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break #跳出迭代，结束训练


    return train_loss_ls,test_loss_ls





train_loss_ls,test_loss_ls = train(model,train_loader,val_loader,optimizer,criterion,epochs,early_stopping,args.outpath,model_save_path,device)
np.savez("%s/%s_loss_eval_metrics.npz"%(args.outpath,model_save_path),train_loss=train_loss_ls,test_loss = test_loss_ls) 



