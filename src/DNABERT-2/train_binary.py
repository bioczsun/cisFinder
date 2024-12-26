import argparse
import pysam
import random
import numpy as np
import pandas as pd
import collections


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
params.add_argument("--data",help="train and test data",required=True)
params.add_argument("--name",help="train name",required=True)
params.add_argument("--model",help="model name",default="Basset")
params.add_argument("--hidden",help="model name",default=768,type=int)
params.add_argument("--seqlen",help="model input sequence",default=400,type=int)
params.add_argument("--task",help="task number",required=True)
params.add_argument("--seed",help="random seed",default=40,type=int)
params.add_argument("--device",help="device cuda",default="cuda:0")
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

# fasta_file = args.fasta

# fasta = pysam.FastaFile(fasta_file)

class BinaryDataset(Dataset):
    def __init__(self, data,label):
        self.data = data.astype(float)
        self.label = label.astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        # print(f"Index: {idx}, Data shape: {data.shape}, Label shape: {label.shape}")
        return torch.tensor(data), torch.LongTensor([label])

set_random_seed(args.seed)
task_name = args.name
batch_size = int(args.batch)
epochs = int(args.epoch)
data_ = np.load(args.data,allow_pickle=True)
device = args.device if torch.cuda.is_available() else "cpu"
hidden_size = args.hidden
task_num = args.task

# Create a model
model = model = eval(f"models.{args.model}({hidden_size},{task_num})")
model.to(device)

#定义训练参数
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=float(args.lr),eps=0.00001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=3,gamma=0.8)

model_save_path = "%s_%s_%s_%s"%(task_name,args.model,args.seed,args.seqlen)
early_stopping = utils.EarlyStopping(save_path=args.outpath,model_name=model_save_path,patience=5)





# Create a DataLoader for the training set

train_dataset = BinaryDataset(data_["train_data"],data_["train_label"])
train_loader = DataLoader(train_dataset, batch_size=int(args.batch),num_workers=5,shuffle=True) 


# Create a DataLoader for the validation set
val_dataset = BinaryDataset(data_["test_data"],data_["test_label"])
val_loader = DataLoader(val_dataset, batch_size=int(args.batch), shuffle=False,num_workers=5)


#define train
def train(model,train_dataloader,test_dataloader,optimizer,criterion,epochs,early_stopping,scheduler,save_path,model_name,device):
    '''
    train model
    '''
    train_loss_ls = []
    test_loss_ls = []
    train_auc_ls = []
    test_auc_ls = []
    for epoch in range(epochs):
        model.train()
        total_step = len(train_dataloader)
        running_loss = 0.0
        train_auc = 0.0
        for index,(data,label) in enumerate(train_dataloader):

            ##get single batch data
            train_data =data.float().to(device)
            train_label = label.to(device).view(-1,1)


            optimizer.zero_grad()

            outputs = model(train_data)
            outputs = F.sigmoid(outputs)
            loss = criterion(outputs,train_label.float())

            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            y_scores = outputs.cpu().detach().numpy()
            fpr, tpr, _ = roc_curve(train_label.cpu(), y_scores)

            auroc = auc(fpr, tpr)
            train_auc += auroc

            if (index + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, auc: {:.4f}'
                        .format(epoch + 1, epochs, index + 1, total_step, loss.item(), auroc))
        epoch_loss = running_loss / (len(train_dataloader))
        epoch_auc = train_auc / (len(train_dataloader))
        train_loss_ls.append(epoch_loss)
        train_auc_ls.append(epoch_auc)



        print("------------eval------------------------")
        test_total_step = len(test_dataloader)
        test_runing_loss = 0.0
        test_true = []
        test_score = []
        test_pred = []
        with torch.no_grad():
            model.eval()
            for _,(data,label) in enumerate(test_dataloader):

                ##get single batch data
                test_data = data.float().to(device)
                test_label = label.to(device).view(-1,1)

                #forward
                outputs = model(test_data)
                outputs = F.sigmoid(outputs)


                test_loss = criterion(outputs,test_label.float())
                test_runing_loss += test_loss.item()

                #calculate auc and acc,recall
                y_scores = outputs.cpu().detach().numpy()
                binary_label = np.where(outputs.cpu().detach().numpy()>0.5,1,0)
                test_true.extend(test_label.cpu().numpy())
                test_pred.extend(binary_label)
                test_score.extend(y_scores)

        fpr, tpr, _ = roc_curve(test_true,test_score)
        eval_auroc = auc(fpr, tpr)

        epoch_test_loss = test_runing_loss / test_total_step

        print('当前Epoch [{}/{}], Train Loss: {:.4f},Train AUC: {:.4f}, Eval Loss: {:.4f},Eval AUC: {:.4f}'.format(epoch + 1, epochs, epoch_loss,epoch_auc,epoch_test_loss,eval_auroc))
        print(classification_report(test_true,test_pred))

        test_loss_ls.append(epoch_test_loss)
        test_auc_ls.append(eval_auroc)

        #early stopping and step scheduler
        early_stopping(-eval_auroc,model)

        scheduler.step()

        #达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            eval_metrics, fpr_tpr_data, precision_recall_data = eval(model,test_dataloader,save_path,model_name,criterion,device)
            break #跳出迭代，结束训练

    return train_loss_ls,test_loss_ls,train_auc_ls,test_auc_ls,eval_metrics, fpr_tpr_data, precision_recall_data



def eval(model,test_dataloader,save_path,model_name,criterion,device):
    eval_model = model
    eval_model.load_state_dict(torch.load("%s/%s_best_network.pth"%(save_path,model_name)))
    eval_model.to(device)
    test_runing_loss = 0.0
    test_true = []
    test_score = []
    test_pred = []
    with torch.no_grad():
        eval_model.eval()
        for _,(data,label) in enumerate(test_dataloader):
            test_data =data.float().to(device)
            test_label = label.to(device).view(-1,1)
            outputs = model(test_data)
            outputs = F.sigmoid(outputs)
            test_loss = criterion(outputs,test_label.float())
            test_runing_loss += test_loss.item()
            y_scores = outputs.cpu().detach().numpy()

            binary_label = np.where(outputs.cpu().detach().numpy()>0.5,1,0)
            test_true.extend(test_label.cpu().numpy())
            test_pred.extend(binary_label)
            test_score.extend(y_scores)

        fpr, tpr, _ = roc_curve(test_true,test_score)
        eval_auroc = auc(fpr, tpr)

        precision, recall, thresholds = precision_recall_curve(test_true,test_score)
        auprc = auc(recall, precision)

        fpr_tpr_data = {"fpr": fpr, "tpr": tpr}
        precision_recall_data = {"precision": precision, "recall": recall}

        eval_metrics = {
            "classification_report": classification_report(test_true, test_pred),
            "eval_auroc": eval_auroc,
            "eval_auprc": auprc
        }

        return eval_metrics, fpr_tpr_data, precision_recall_data

        # df = pd.DataFrame({"fpr":fpr,"tpr":tpr})
        # df.to_csv("%s/eval_result_%s.csv"%(save_path,model_name))
        # df1 = pd.DataFrame({"precision":precision,"recall":recall})
        # df1.to_csv("%s/eval_pr_result_%s.csv"%(save_path,model_name))
        # f = open("%s/classification_report_%s.txt"%(save_path,model_name),"w")
        # f.write(classification_report(test_true,test_pred))
        # f.write("\neval_auroc:%s"%eval_auroc)
        # f.write("\neval_auprc:%s"%auprc)
        # f.close()

#start train

train_loss_ls,test_loss_ls,train_auc_ls,test_auc_ls,eval_metrics, fpr_tpr_data, precision_recall_data = train(model,train_loader,val_loader,optimizer,criterion,epochs,early_stopping,scheduler,args.outpath,model_save_path,device)
np.savez("%s/%s_loss_eval_metrics.npz"%(args.outpath,model_save_path),train_loss=train_loss_ls,test_loss = test_loss_ls,train_auc = train_auc_ls,test_auc = test_auc_ls,
         eval_metrics=eval_metrics,fpr_tpr_data=fpr_tpr_data,precision_recall_data=precision_recall_data) 
# np.save("%s/train_loss_%s.npy"%(args.outpath,model_save_path),train_loss_ls)
# np.save("%s/test_loss_%s.npy"%(args.outpath,model_save_path),test_loss_ls)
# np.save("%s/train_auc_%s.npy"%(args.outpath,model_save_path),train_auc_ls)
# np.save("%s/test_auc_%s.npy"%(args.outpath,model_save_path),test_auc_ls)  



