import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class DNABert(nn.Module):
    def __init__(self, hidden_size=768, num_classes=1):
        super(DNABert, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
            )

    def forward(self, x):
        x = self.fc(x)
        return x
    

class HyenaDNA(nn.Module):
    def __init__(self, hidden_size=768, num_classes=1):
        super(HyenaDNA, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
            )

    def forward(self, x):
        x = self.fc(x)
        return x
    
# a = DNABert()
# b = torch.randn(1,768)
# out = a(b)
# print(out.shape)