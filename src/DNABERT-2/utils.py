import torch
import torch.nn as nn
import numpy as np
import os



onehot_nuc = {'A': [1, 0, 0, 0],
              'C': [0, 1, 0, 0],
              'G': [0, 0, 1, 0],
              'T': [0, 0, 0, 1],
              'N': [0, 0, 0, 0]}

def onehot_seq(seq):
    """
    Convert a nucleotide sequence to its one-hot encoding.
    Parameters:
    seq (str): The input DNA sequence.
    Returns:
    numpy.ndarray: The one-hot encoded representation of the input sequence as a 2D numpy array.
    Example:
    Input: 'ACGTN'
    Output: array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])

    Note: 'A', 'C', 'G', 'T' represent the nucleotide bases adenine, cytosine, guanine, and thymine, respectively, while 'N' is used to represent an unknown or ambiguous base.
    """
    one_hot_ls = []
    for nuc in str(seq).upper():
        if nuc in onehot_nuc:
            one_hot_ls.append(onehot_nuc[nuc])
        else:
            one_hot_ls.append(onehot_nuc["N"])
    return np.array(one_hot_ls)

#transform a sequence to K-mer vector (default: K=6)
def seq_to_kspec(seq, K=6):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}
    kspec_vec = np.zeros((4**K,1))
    for i in range(len(seq)-K+1):
        sub_seq = seq[i:(i+K)]
        index = 0
        for j in range(K):
            index += encoding_matrix[sub_seq[j]]*(4**(K-j-1))
        kspec_vec[index] += 1
    return kspec_vec

import numpy as np
import random
import torch



def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
    """ dna_1hot

    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
      n_sample:  sample ACGT for N

    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim:seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype='float16')
    else:
        seq_code = np.zeros((seq_len, 4), dtype='bool')
        
    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == 'A':
                seq_code[i, 0] = 1
            elif nt == 'C':
                seq_code[i, 1] = 1
            elif nt == 'G':
                seq_code[i, 2] = 1
            elif nt == 'T':
                seq_code[i, 3] = 1
            else:
                # Set N or unknown bases to zero
                seq_code[i, :] = 0

    return seq_code

def dna_1hot_index(seq, n_sample=False):
  """ dna_1hot_index

    Args:
      seq:       nucleotide sequence.

    Returns:
      seq_code:  index int array representation.
    """
  seq_len = len(seq)
  seq = seq.upper()

  # map nt's to a len(seq) of 0,1,2,3
  seq_code = np.zeros(seq_len, dtype='uint8')
    
  for i in range(seq_len):
    nt = seq[i]
    if nt == 'A':
      seq_code[i] = 0
    elif nt == 'C':
      seq_code[i] = 1
    elif nt == 'G':
      seq_code[i] = 2
    elif nt == 'T':
      seq_code[i] = 3
    else:
      if n_sample:
        seq_code[i] = random.randint(0,3)
      else:
        seq_code[i] = 4

  return seq_code

class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path,model_name,patience=3,verbose=False, delta=0):
        """
        Args:
            save_path : model save dir
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        super(EarlyStopping, self).__init__()
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_name = model_name

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, '%s_best_network.pth'%self.model_name)
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

linear_units_dict = {
    "DeepSEA": {
        "200bp": 35520,
        "256bp": 48960,
        "400bp": 83520,
        "600bp": 131520,
        "800bp": 179520,
        "1000bp": 227520
    },
    "Basset": {
        "200bp": 4600,
        "256bp": 6000,
        "400bp": 9600,
        "600bp": 14600,
        "800bp": 19600,
        "1000bp": 24600
    },
    "DanQ": {
        "200bp": 9600,
        "256bp": 12160,        
        "400bp": 19200,
        "600bp": 29440,
        "800bp": 39040,
        "1000bp": 48640
    },
    "ExplaiNN": {
        "200bp": 200,
        "256bp": 256,
        "400bp": 400,
        "600bp": 600,
        "800bp": 800,
        "1000bp": 1000
    },
    "SATORI": {
        "200bp": 10240,
        "256bp": 12800,
        "400bp": 20480,
        "600bp": 30720,
        "800bp": 40960,
        "1000bp": 51200
    },
    "CNN_Transformer": {
        "200bp": 6000,
        "256bp": 7500,
        "400bp": 12000,
        "600bp": 18000,
        "800bp": 24000,
        "1000bp": 30000
    },
    "CNN_Attention": {
        "200bp": 6000,
        "256bp": 7500,
        "400bp": 12000,
        "600bp": 18000,
        "800bp": 24000,
        "1000bp": 30000
    },
    "CNN": {
        "200bp": 9900,
        "256bp": 12600,
        "400bp": 19800,
        "600bp": 30000,
        "800bp": 39900,
        "1000bp": 49800
    }
}
