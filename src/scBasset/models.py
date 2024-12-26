import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class ExpActivation(nn.Module):
    """
    Exponential activation function from Koo & Ploenzke, 2021 (PMID: 34322657)
    """
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)

class Unsqueeze(torch.nn.Module):
    """
    Unsqueeze for sequential models
    """
    def forward(self, x):
        return x.unsqueeze(-1)

class DeepSEA(nn.Module):
    def __init__(self,classes,linear_units,activate,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        conv_kernel_size = 8
        pool_kernel_size = 2
        if activate == 'relu':
            activation = nn.ReLU()
        if activate == 'gelu':
            activation = nn.GELU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=conv_kernel_size, stride=1, padding=0,bias=False),
            nn.BatchNorm1d(320),
            activation,
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=conv_kernel_size, stride=1, padding=0,bias=False),
            nn.BatchNorm1d(480),
            activation,
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Conv1d(in_channels=480, out_channels=960, kernel_size=conv_kernel_size, stride=1, padding=0,bias=False),
            nn.BatchNorm1d(960),
            activation,
            nn.Dropout(0.2)
            )
        self.fc = nn.Sequential(
            nn.Linear(linear_units, 2048,bias=False),
            activation,
            nn.Dropout(p=0.2),
            nn.Linear(2048, 32,bias=False),
            activation,
            nn.Linear(32, classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class Basset(nn.Module):
    def __init__(self, classes,linear_units,activate,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if activate == 'relu':
            activation = nn.ReLU()
        if activate == 'gelu':
            activation = nn.GELU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19, stride=1, padding=9,bias=False),
            nn.BatchNorm1d(300),
            activation,
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=300, out_channels=200, kernel_size=11, stride=1, padding=5,bias=False),
            nn.BatchNorm1d(200),
            activation,
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7, stride=1, padding=3,bias=False),
            nn.BatchNorm1d(200),
            activation,
            nn.MaxPool1d(kernel_size=4, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=linear_units, out_features=2048,bias=False),
            activation,
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=32,bias=False),
            activation,
            nn.Linear(in_features=32, out_features=classes)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class DanQ(nn.Module):
    def __init__(self,classes,linear_units,activate,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if activate == 'relu':
            activation = nn.ReLU()
        if activate == 'gelu':
            activation = nn.GELU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=320,kernel_size=19,padding=9,bias=False),
            activation,
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(input_size=320, hidden_size=320, num_layers=2, batch_first=True, bidirectional=True,bias=False)

        self.fc = nn.Sequential(
            nn.Linear(linear_units, 2048,bias=False),
            activation,
            nn.Linear(2048, 32,bias=False),
            activation,
            nn.Linear(32, classes)
        )
    
    def forward(self, x):
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        out,(hn,cn) = self.lstm(x)
        out = out.transpose(1,2)
        out = out.contiguous().view(x.size()[0],-1)
        out = self.fc(out)
        return out
    

class ExplaiNN(nn.Module):
    def __init__(self,classes, input_length,activate,num_cnns=300,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if activate == 'relu':
            activation = nn.ReLU()
        if activate == 'gelu':
            activation = nn.GELU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.Conv1d = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19,padding=19//2,bias=False),
                nn.BatchNorm1d(num_cnns),
                activation,
                nn.MaxPool1d(10),
                )
        self.Linear = nn.Sequential(
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(input_length / 10)*num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns,bias=False),
                nn.BatchNorm1d(100 * num_cnns),
                activation,
                nn.Dropout(0.3),
                nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns,bias=False),
                nn.BatchNorm1d(1 * num_cnns),
                activation,
                nn.Flatten(),
        )

        self.classifier = nn.Linear(num_cnns,classes)

    def forward(self,x):
        out = self.Conv1d(x)
        out = self.Linear(out)
        out1 = self.classifier(out)
        return out1
    
class PositionalEncoding(nn.Module):
    # Taken from: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
       
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
       
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
    
class SATORI(nn.Module):
    def __init__(self,classes, linear_units,activate) -> None:
        super().__init__()
        self.numOutputChannels = 8*64
        self.numMultiHeads = 8
        self.SingleHeadSize = 64
        self.usePE = False

        if activate == 'relu':
            activation = nn.ReLU()
        if activate == 'gelu':
            activation = nn.GELU()
        elif activate == 'exp':
            activation = ExpActivation()

        if self.usePE:
            self.pe = PositionalEncoding(d_model = self.numOutputChannels,dropout=0.1)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=self.numOutputChannels, kernel_size=19, padding=9,bias=False),
            nn.BatchNorm1d(self.numOutputChannels),
            activation,
            nn.MaxPool1d(kernel_size=10),
            nn.Dropout(0.2)
        )
        self.RNN = nn.LSTM(input_size=self.numOutputChannels, hidden_size=self.numOutputChannels // 2, num_layers=2, batch_first=True, bidirectional=True,dropout=0.4,bias=False)
        self.MultiheadAttention = nn.MultiheadAttention(embed_dim=self.numOutputChannels, num_heads=self.numMultiHeads,bias=False)
        self.MultiheadLinear = nn.Sequential(nn.Linear(self.numOutputChannels,self.numOutputChannels,bias=False),
                                             activation,)
        self.fc = nn.Sequential(nn.Linear(linear_units,2048,bias=False),
                                activation,
                                nn.Linear(2048,32,bias=False),
                                activation,
                                nn.Linear(32,classes)
                                )

    def forward(self,x):
        x = self.layer1(x)
        x = x.permute(0,2,1)
        x = self.RNN(x)
        x = x[0].permute(1,0,2)
        x = self.MultiheadAttention(x,x,x)
        x = x[0].permute(1,0,2)
        x = self.MultiheadLinear(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    

class CNN_Transformer(nn.Module):
    def __init__(self,classes,linear_units,activate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if activate == 'relu':
            activation = nn.ReLU()
        if activate == 'gelu':
            activation = nn.GELU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19, padding=9,bias=False),
            nn.BatchNorm1d(300),
            activation,
            nn.MaxPool1d(kernel_size=10),
            nn.Dropout(0.2)
        )
        self.transformer = nn.TransformerEncoderLayer(d_model=300,nhead=6)

        self.fc = nn.Sequential(
            nn.Linear(linear_units, 2048,bias=False),
            activation,
            nn.Linear(2048, 32,bias=False),
            activation,
            nn.Linear(32, classes)
        )

    def forward(self, x):
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        x = self.transformer(x)

        out = x.contiguous().view(x.size()[0],-1)
        out = self.fc(out)
        return out
    

class CNN_Attention(nn.Module):
    def __init__(self,classes,linear_units,activate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if activate == 'relu':
            activation = nn.ReLU()
        if activate == 'gelu':
            activation = nn.GELU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=300,kernel_size=19,padding=9,bias=False),
            nn.BatchNorm1d(300),
            activation,
            nn.MaxPool1d(10),
        )
        self.multiatten = nn.MultiheadAttention(300,4,batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(linear_units,2048,bias=False),
            activation,
            nn.Linear(2048,32,bias=False),
            activation,
            nn.Linear(32,classes),
        )
    def forward(self,x):
        x = self.conv(x)
        x = x.transpose(1,2)
        x,weights = self.multiatten(x,x,x)
        x = x.transpose(1,2).reshape(x.size(0),-1)
        x = self.linear(x)
        return x
    

class CNN(nn.Module):
    def __init__(self, classes, linear_units,activate) -> None:
        super(CNN, self).__init__()

        if activate == 'relu':
            activation = nn.ReLU()
        if activate == 'gelu':
            activation = nn.GELU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19, padding=9,bias=False),
            nn.BatchNorm1d(300),
            activation,
            nn.MaxPool1d(6, 6),
        )
        self.dense = nn.Sequential(
            nn.Linear(linear_units, 2048,bias=False),
            activation,
            nn.Linear(2048, 32,bias=False),
            activation,
            nn.Linear(32, classes),
        )


    def forward(self, x):
        out = self.conv1d(x)
        out1 = out.contiguous().view(x.size()[0], -1)
        out = self.dense(out1)
        return out

class scBasset(nn.Module):
    def __init__(self,num_class, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=288, kernel_size=17, padding=8,bias=False),
            nn.BatchNorm1d(288),
            nn.GELU(),
            nn.MaxPool1d(3),#288*448

            nn.Conv1d(in_channels=288, out_channels=288, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(288),
            nn.GELU(),
            nn.MaxPool1d(2),#288*224

            nn.Conv1d(in_channels=288, out_channels=323, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(323),
            nn.GELU(),
            nn.MaxPool1d(2),#323*112

            nn.Conv1d(in_channels=323, out_channels=363, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(363),
            nn.GELU(),
            nn.MaxPool1d(2),#363*56

            nn.Conv1d(in_channels=363, out_channels=407, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(407),
            nn.GELU(),
            nn.MaxPool1d(2),#407*28

            nn.Conv1d(in_channels=407, out_channels=456, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(456),
            nn.GELU(),
            nn.MaxPool1d(2),#456*14

            nn.Conv1d(in_channels=456, out_channels=512, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(2),#512*7

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1,bias=False),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1792, 32,bias=False),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.GELU()
        )

        self.linear = nn.Sequential(
            nn.Linear(32, num_class)
        )

    def forward(self, x):
        x = self.conv1d(x)
        x = self.dense(x)
        x = self.linear(x)
        return x
# a = torch.randn(2,4,1344)
# model = scBasset(1722)
# print(model(a).shape)
        
# fft = FFTConv1d(in_channels=128,out_channels=128,kernel_size=601,padding=300)
# a = torch.randn(32,128,600)
# print(fft(a).shape)
# model = Basset(1,14600,"relu")
# print(model)
# class RelativePositionEncoding(nn.Module):
#     def __init__(self, max_length, d_model):
#         super(RelativePositionEncoding, self).__init__()
#         self.max_length = max_length
#         self.d_model = d_model

#         self.relative_position_embeddings = nn.Embedding(2 * max_length - 1, d_model)

#     def forward(self, length):
#         positions = torch.arange(length, dtype=torch.long, device=self.relative_position_embeddings.weight.device)
#         relative_positions_matrix = positions[:, None] - positions[None, :]
#         relative_positions_matrix += self.max_length - 1
#         return self.relative_position_embeddings(relative_positions_matrix)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads, max_length):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.d_model = d_model

#         self.qkv_linear = nn.Linear(d_model, 3 * d_model)
#         self.out_linear = nn.Linear(d_model, d_model)
#         self.relative_position_encoding = RelativePositionEncoding(max_length, d_model // num_heads)

#     def forward(self, x):
#         batch_size, seq_length, _ = x.size()
#         qkv = self.qkv_linear(x).view(batch_size, seq_length, 3, self.num_heads, self.d_model // self.num_heads)
#         q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

#         q = q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
#         k = k.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
#         v = v.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

#         relative_positions = self.relative_position_encoding(seq_length)  # (seq_length, seq_length, head_dim)

#         scores = torch.einsum('bhqd, bhkd -> bhqk', q, k)
#         relative_scores = torch.einsum('bhqd, qkd -> bhqk', q, relative_positions)

#         scores += relative_scores
#         attn = F.softmax(scores, dim=-1)

#         context = torch.einsum('bhqk, bhvd -> bhqd', attn, v)
#         context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)
#         return self.out_linear(context)

# class TransformerEncoderLayerWithRelativePosition(nn.Module):
#     def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, max_length=512):
#         super(TransformerEncoderLayerWithRelativePosition, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads, max_length)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = F.relu

#     def forward(self, src):
#         src2 = self.self_attn(src)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src
    
# class Basset_ExplaiNN(nn.Module):
#     def __init__(self, input_length, num_cnns, classes, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv1d = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
#             nn.BatchNorm1d(num_cnns),
#             nn.ReLU(),
#             nn.MaxPool1d(6),
#         )
#         self.pe = nn.Embedding(int(input_length/6), num_cnns)
#         self.transformer = TransformerEncoderLayerWithRelativePosition(d_model=num_cnns, num_heads=6, dim_feedforward=512)
#         self.classifier = nn.Sequential(
#                 nn.Linear(num_cnns * int(input_length/6), 256),
#                 nn.ReLU(),
#                 nn.Linear(256, classes)
#                 )
#         self.explainn = nn.Sequential(
#                 nn.MaxPool1d(6),
#                 nn.Flatten(),
#                 Unsqueeze(),
#                 nn.Conv1d(in_channels=int(input_length / 6 / 6)*num_cnns,
#                           out_channels=10 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(10 * num_cnns),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Conv1d(in_channels=10 * num_cnns,
#                           out_channels=1 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(1 * num_cnns),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(num_cnns, classes)
#         )
#     def forward(self, x):
#         conv1 = self.conv1d(x)
#         transformer = self.transformer(conv1.permute(0,2,1))
#         transformer = self.transformer(conv1.permute(0,2,1)) + conv1.permute(0,2,1)
#         transformer = transformer.contiguous().view(x.size()[0], -1)
#         transformer_out = self.classifier(transformer)
#         explainn_out = self.explainn(conv1)
#         return transformer_out, explainn_out

# class AbsolutePositionEncoding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super(AbsolutePositionEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)
        
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.d_model = d_model

#         self.qkv_linear = nn.Linear(d_model, 3 * d_model)
#         self.out_linear = nn.Linear(d_model, d_model)
#         self.attention_dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         batch_size, seq_length, _ = x.size()
#         qkv = self.qkv_linear(x).view(batch_size, seq_length, 3, self.num_heads, self.d_model // self.num_heads)
#         q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

#         q = q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
#         k = k.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
#         v = v.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

#         scores = torch.einsum('bhqd, bhkd -> bhqk', q, k) / math.sqrt(self.d_model // self.num_heads)
#         attn = F.softmax(scores, dim=-1)
#         attn = self.attention_dropout(attn)

#         context = torch.einsum('bhqk, bhvd -> bhqd', attn, v)
#         context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)
#         return self.out_linear(context)

# class TransformerEncoderLayerWithAbsolutePosition(nn.Module):
#     def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, max_length=512):
#         super(TransformerEncoderLayerWithAbsolutePosition, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.position_encoding = AbsolutePositionEncoding(d_model, max_length)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = F.relu

#     def forward(self, src):
#         src = self.position_encoding(src)
#         src2 = self.self_attn(src)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

# class Basset_ExplaiNN(nn.Module):
#     def __init__(self, input_length, num_cnns, classes, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv1d = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
#             nn.BatchNorm1d(num_cnns),
#             nn.ReLU(),
#             nn.MaxPool1d(6),
#         )
#         self.pe = nn.Embedding(int(input_length/6), num_cnns)
#         self.transformer = TransformerEncoderLayerWithAbsolutePosition(d_model=num_cnns, num_heads=6, dim_feedforward=512)
#         self.classifier = nn.Sequential(
#                 nn.Linear(num_cnns * int(input_length/6), 256),
#                 nn.ReLU(),
#                 nn.Linear(256, classes)
#                 )
#         self.explainn = nn.Sequential(
#                 nn.MaxPool1d(6),
#                 nn.Flatten(),
#                 Unsqueeze(),
#                 nn.Conv1d(in_channels=int(input_length / 6 / 6)*num_cnns,
#                           out_channels=10 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(10 * num_cnns),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Conv1d(in_channels=10 * num_cnns,
#                           out_channels=1 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(1 * num_cnns),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(num_cnns, classes)
#         )
#     def forward(self, x):
#         conv1 = self.conv1d(x)
#         transformer = self.transformer(conv1.permute(0,2,1))
#         transformer = self.transformer(conv1.permute(0,2,1)) + conv1.permute(0,2,1)
#         transformer = transformer.contiguous().view(x.size()[0], -1)
#         transformer_out = self.classifier(transformer)
#         explainn_out = self.explainn(conv1)
#         return transformer_out, explainn_out
# a = torch.randn(32, 4, 600)
# model = Basset_ExplaiNN(600, 300, 1)
# print(model(a)[0].shape)
    


# a = torch.randn(32,4,600)
# model = SATORI(1,131520,"relu")

# print(model(a).shape)

# class GRIM_Basset_ExplaiNN(nn.Module):
#     def __init__(self,input_length,num_cnns,linear_units,classes, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.conv1d = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=num_cnns, kernel_size=19, padding=9),
#             nn.BatchNorm1d(num_cnns),
#             nn.ReLU(),
#             nn.MaxPool1d(6)
#         )

#         self.basset_conv = nn.Sequential(
#             nn.Conv1d(in_channels=num_cnns, out_channels=200, kernel_size=11, padding=5),
#             nn.BatchNorm1d(200),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             # nn.Dropout(0.3),
#             nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7, padding=3),
#             nn.BatchNorm1d(200),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#             nn.Linear(in_features=linear_units, out_features=256),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#         )

#         self.basset_linear = nn.Sequential(
#             nn.Linear(in_features=256, out_features=128),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(in_features=128, out_features=classes)
#         )

#         self.explainn = nn.Sequential(
#                 nn.Flatten(),
#                 Unsqueeze(),
#                 nn.Conv1d(in_channels=int(input_length / 6)*num_cnns,
#                           out_channels=10 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(10 * num_cnns),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.Conv1d(in_channels=10 * num_cnns,
#                           out_channels=1 * num_cnns, kernel_size=1,
#                           groups=num_cnns),
#                 nn.BatchNorm1d(1 * num_cnns),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(num_cnns,classes)
#         )

#         self.Discriminator = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#         )

#     def forward(self,x,device):

#         generator = torch.Generator(device=device)
#         generator.manual_seed(0)  # 你可以根据需要设置随机种子  
#         shuffled_indices = torch.argsort(torch.rand(x.shape, generator=generator,device=device), dim=1)
#         shuffled_x = torch.gather(x, dim=1, index=shuffled_indices)

#         local_embedding = self.conv1d(x)
#         global_embedding = self.basset_conv(local_embedding)

#         local_embedding_shuffle = self.conv1d(shuffled_x)
#         global_embedding_shuffle = self.basset_conv(local_embedding_shuffle)

#         basset_out = self.basset_linear(global_embedding)
#         explainn_out = self.explainn(local_embedding)

#         local_encoding_expanded = local_embedding.unsqueeze(-1)
#         global_encoding_expanded = global_embedding.unsqueeze(1).unsqueeze(1)
#         outer_product_tensor = local_encoding_expanded * global_encoding_expanded
#         global_max_pooled = torch.max(outer_product_tensor.view(x.shape[0], 150 * 200, 256), dim=1)[0]
#         positive_scores = self.Discriminator(global_max_pooled)

#         global_embedding_shuffle = global_embedding_shuffle.unsqueeze(1).unsqueeze(1)
#         outer_product_tensor_shuffle = local_encoding_expanded * global_embedding_shuffle
#         global_max_pooled_shuffle = torch.max(outer_product_tensor_shuffle.view(x.shape[0], 150 * 200, 256), dim=1)[0]
#         negative_scores = self.Discriminator(global_max_pooled_shuffle)

#         #计算loss
#         positive_loss = F.softplus(-positive_scores).mean()
#         negative_loss = F.softplus(negative_scores).mean()

#         loss = positive_loss + negative_loss



#         return loss,basset_out,explainn_out
    
# class Expert(nn.Module):
#     def __init__(self,input_dim, output_dim, kernel_size, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels=4,out_channels=32,kernel_size=19,padding=9),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(3),
#             nn.Conv1d(in_channels=32,out_channels=32,kernel_size=11,padding=5),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(in_channels=32,out_channels=32,kernel_size=7,padding=3),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#             nn.Linear(32*50,128),
#             nn.ReLU(),
#             nn.Linear(128,64)
#         )
#     def forward(self,x):
#         return self.conv(x)

# class MoeModel(nn.Module):
#     def __init__(self, input_dim, output_dim, num_experts,top_k, kernel_size=3):
#         super(MoeModel, self).__init__()
#         self.top_k = top_k
#         self.experts = nn.ModuleList([Expert(input_dim, output_dim, kernel_size) for _ in range(num_experts)])
#         self.gating_network = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=128, kernel_size=kernel_size, padding=kernel_size//2),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.MaxPool1d(10),
#             nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding=3),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#             nn.Linear(128*30, num_experts),
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64*self.top_k, 128),
#             nn.ReLU(),
#             nn.Linear(128,1))

#     def forward(self, x):
#         # Calculate gating weights
#         gating_weights = self.gating_network(x)
#         top_k_values, top_k_indices = torch.topk(F.softmax(gating_weights, dim=1), self.top_k, dim=1)

#         # 初始化输出和权重
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         expanded_outputs = expert_outputs.unsqueeze(1).expand(-1, self.top_k, -1, -1)
        
#         # 调整索引张量的维度并执行 gather 操作
#         expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[2])
#         selected_outputs = expanded_outputs.gather(2, expanded_indices)

#         # 加权求和
#         final_output = (selected_outputs * top_k_values.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

#         out = self.fc(final_output)
#         return out
    

# class Expert(nn.Module):
#     def __init__(self,input_dim, output_dim, kernel_size, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels=4,out_channels=1,kernel_size=19,padding=9),
#             nn.BatchNorm1d(1),
#             nn.ReLU(),
#             nn.MaxPool1d(10),
#             nn.Flatten(),
#             Unsqueeze(),
#             nn.Conv1d(in_channels=60,
#                         out_channels=10, kernel_size=1,
#                         groups=1),
#             nn.BatchNorm1d(10 * 1),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Conv1d(in_channels=10 * 1,
#                         out_channels=32 * 1, kernel_size=1,
#                         groups=1),
#             nn.BatchNorm1d(32 * 1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#     def forward(self,x):
#         return self.conv(x)

# class MoeModel(nn.Module):
#     def __init__(self, input_dim, output_dim, num_experts,top_k, kernel_size=3):
#         super(MoeModel, self).__init__()
#         self.top_k = top_k
#         self.experts = nn.ModuleList([Expert(input_dim, output_dim, kernel_size) for _ in range(num_experts)])
#         self.gating_network = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=16, kernel_size=kernel_size, padding=kernel_size//2),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(10),
#             nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, padding=3),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten(),
#             nn.Linear(16*30, num_experts),
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32*self.top_k, 128),
#             nn.ReLU(),
#             nn.Linear(128,1))

#     def forward(self, x):
#         # Calculate gating weights
#         gating_weights = self.gating_network(x)
#         top_k_values, top_k_indices = torch.topk(F.softmax(gating_weights, dim=1), self.top_k, dim=1)

#         # 初始化输出和权重
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         expanded_outputs = expert_outputs.unsqueeze(1).expand(-1, self.top_k, -1, -1)
        
#         # 调整索引张量的维度并执行 gather 操作
#         expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[2])
#         selected_outputs = expanded_outputs.gather(2, expanded_indices)

#         # 加权求和
#         final_output = (selected_outputs * top_k_values.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

#         out = self.fc(final_output)
#         return out
    
# a = torch.randn(32,4,600)
# model = MoeModel(600,1,300,30)
# print(model(a).shape)
    
# class RotaryPositionEmbedding:
#     def __init__(self, dim):
#         self.dim = dim
#         # 逆频率，用于计算旋转位置编码
#         self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

#     def get_position_embedding(self, seq_len, device):
#         """
#         生成旋转位置编码
#         :param seq_len: 序列长度
#         :param device: 设备
#         :return: sin 和 cos 编码后的位置嵌入
#         """
#         pos = torch.arange(seq_len, dtype=torch.float32, device=device)
#         sinusoid_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
#         emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
#         return emb

#     def apply_rotary_pos_emb(self, k, pos_emb):
#         """
#         应用旋转位置编码到 Key
#         :param k: 键向量 (B, num_heads, L, head_dim) - batch size, number of heads, sequence length, dimension per head
#         :param pos_emb: 位置编码，形状应为 (L, head_dim)
#         :return: 应用了旋转位置编码后的键向量
#         """
#         B, num_heads, L, head_dim = k.shape

#         # 确保 pos_emb 的形状正确
#         if pos_emb.shape[0] != L or pos_emb.shape[1] != head_dim:
#             raise ValueError(f"Position embedding shape {pos_emb.shape} does not match required shape ({L}, {head_dim})")

#         # 生成 cos_pos 和 sin_pos
#         cos_pos = pos_emb[:, None, :].repeat(1, 2, 1)  # 形状调整为 (L, 2, head_dim)
#         sin_pos = pos_emb[:, None, :].repeat(1, 2, 1)

#         # 应用旋转编码
#         k_rot = (k * cos_pos) + (torch.cat((-k[..., 1::2], k[..., ::2]), dim=-1) * sin_pos)
        
#         return k_rot


# class CustomMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, rotary_embedding=None, dropout=0.1):
#         super(CustomMultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.dropout = nn.Dropout(dropout)
        
#         self.rotary_embedding = rotary_embedding  # 旋转位置编码

#     def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
#         B, L, D = query.size()
#         # 投影为多头
#         Q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L, head_dim)
#         K = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

#         # 如果有旋转位置编码
#         if self.rotary_embedding is not None:
#             pos_emb = self.rotary_embedding.get_position_embedding(seq_len=L, device=query.device)
#             K = self.rotary_embedding.apply_rotary_pos_emb(K, pos_emb)
        
#         # 计算注意力得分
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         if attn_mask is not None:
#             scores += attn_mask
#         attn_weights = torch.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
        
#         # 加权求和 Value
#         context = torch.matmul(attn_weights, V)
#         context = context.transpose(1, 2).contiguous().view(B, L, D)
        
#         # 输出层
#         output = self.out_proj(context)
#         return output

    


