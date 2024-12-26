import argparse
import os
import utils
import random
import pysam
import scipy
import numpy as np
import pandas as pd

def set_random_seed(random_seed = 40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

set_random_seed(40)

params = argparse.ArgumentParser(description='Generate training and testing sets')
params.add_argument("--peaks",help="peaks file",required=True)
params.add_argument("--nopeaks",help="nopeaks file",required=True)
params.add_argument("--len",help="sequence length",required=True)
params.add_argument("--fasta",help="fasta seq file",required=True)
params.add_argument("--valid_chrom",help="valid_chrom",nargs='+',required=True)
params.add_argument("--all_inactivate",help="Whether to use all inactive regions as negative samples",required=True)
params.add_argument("--name",help="cell name",required=True)
params.add_argument("--outpath",help="output path",required=True)
args = params.parse_args()


peaks_file = args.peaks
nopeaks_file = args.nopeaks
fasta = args.fasta
valid_chrom = args.valid_chrom
name = args.name
all_inacitvate = args.all_inactivate

# print(valid_chrom)

# print("chr8" == valid_chrom[0])

def is_standard_chrom(chrom):
    if chrom.startswith("chr") and len(chrom) < 6:
        if chrom[3].isdigit() or chrom[3] == "X":
            return True
        else:
            return False
    else:
        return False


def get_train_test_data(peaks_file,nopeaks_file,fasta,outpath,seq_len=256,slide=1024):
    '''
    peals_file: peaks file
    nopeaks_file: nopeaks file
    fasta: fasta file
    seq_length: sequence length
    '''

    fasta = pysam.FastaFile(fasta)

    chromosomes = {k:v for k,v in zip(fasta.references,fasta.lengths)}#提取染色体长度


    #提取peaks
    df = pd.read_table(peaks_file, header=None)
    df.columns = ['chr', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']
    # df.sort_values(by="signalValue",ascending=False,inplace=True)
    bed_peaks = open(peaks_file).readlines()

    train_data_pos = []
    test_data_pos = []

    train_label_pos = []
    test_label_pos = []

    for line in bed_peaks:
        line = line.split()
        chr = line[0]
        start = int(line[1])
        end = int(line[2])
        peaks_mid = int(line[9])
        if chr not in valid_chrom:
            if is_standard_chrom(chr) and 0 < (start+peaks_mid-int(seq_len/2)) < chromosomes[chr] and 0 < (start+peaks_mid+int(seq_len/2))<chromosomes[chr]:
                # sequence = fasta.fetch(chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2))
                # seq_onehot = utils.onehot_seq(sequence)
                # train_data_pos.append(seq_onehot)
                train_data_pos.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
                train_label_pos.append(1)
        else:
            if is_standard_chrom(chr) and 0 < (start+peaks_mid-int(seq_len/2)) < chromosomes[chr] and 0 < (start+peaks_mid+int(seq_len/2))<chromosomes[chr]:
                # sequence = fasta.fetch(chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2))
                # seq_onehot = utils.onehot_seq(sequence)
                # test_data_pos.append(seq_onehot)
                test_data_pos.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
                test_label_pos.append(1)

    nopeaks_regions = []
    nopeaks_file = open(nopeaks_file).readlines()
    for i in nopeaks_file:
        i = i.split()
        if int(i[2]) - int(i[1]) ==slide:
            if is_standard_chrom(i[0]) and  0<(int(i[1])+int(slide/2)-int(seq_len/2))<chromosomes[i[0]] and 0<(int(i[2])-int(slide/2)+int(seq_len/2))<chromosomes[i[0]]:
                # sequence = fasta.fetch(i[0],int(i[1])+int(slide/2)-int(seq_len/2),int(i[2])-int(slide/2)+int(seq_len/2))
                # seq_onehot = utils.onehot_seq(sequence)
                # nopeaks_regions.append(seq_onehot)
                nopeaks_regions.append([i[0],int(i[1])+int(slide/2)-int(seq_len/2),int(i[2])-int(slide/2)+int(seq_len/2)])  

    #划分数据集


    train_data_neg = []
    test_data_neg = []

    train_label_neg = []
    test_label_neg = []

            
    for i in nopeaks_regions:
        if i[0] in valid_chrom:
            test_data_neg.append(i)
            test_label_neg.append(0)
        else:
            train_data_neg.append(i)
            train_label_neg.append(0)
    if all_inacitvate == "True":
        train_data = train_data_pos +train_data_neg
        train_label = train_label_pos + train_label_neg

    else:
        train_data = train_data_pos + random.choices(train_data_neg,k=len(train_data_pos))
        train_label = train_label_pos + random.choices(train_label_neg,k=len(train_data_pos))


    test_data = test_data_pos + random.choices(test_data_neg,k=len(test_data_pos))
    test_label = test_label_pos + random.choices(test_label_neg,k=len(test_label_pos))
    print(len(test_data_neg),len(test_data_pos))
    train_data_onehot = []
    for i in train_data:
        sequence = fasta.fetch(i[0],i[1],i[2])
        seq_onehot = utils.onehot_seq(sequence)
        train_data_onehot.append(seq_onehot)
    test_data_onehot = []
    for i in test_data:
        sequence = fasta.fetch(i[0],i[1],i[2])
        seq_onehot = utils.onehot_seq(sequence)
        test_data_onehot.append(seq_onehot)
    train_data = np.array(train_data_onehot)
    test_data = np.array(test_data_onehot)
    data_outpath = "%s/train_test_%s_%s.npz"%(outpath,name,seq_len)
    print("save data to %s"%data_outpath)

    np.savez(data_outpath,train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label)#,train_cov = train_cov_ls,test_cov=test_cov_ls)

get_train_test_data(peaks_file,nopeaks_file,fasta,seq_len=int(args.len),slide=4096,outpath=args.outpath)
