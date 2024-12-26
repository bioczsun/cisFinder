import argparse
import utils
import random
import pysam
import pyBigWig
import numpy as np

def set_random_seed(random_seed = 40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

set_random_seed(40)

params = argparse.ArgumentParser(description='Generate training and testing sets')
params.add_argument("--peaks",help="peaks file",required=True)
params.add_argument("--nopeaks",help="nopeaks file",required=True)
params.add_argument("--bigWig",help="nopeaks file",required=True)
params.add_argument("--len",help="sequence length",required=True)
params.add_argument("--fasta",help="fasta seq file",required=True)
params.add_argument("--valid_chrom",help="valid_chrom",nargs='+',required=True)
params.add_argument("--all_inactivate",help="valid_chrom",required=True)
params.add_argument("--name",help="cell name",required=True)
params.add_argument("--outpath",help="output path",required=True)
args = params.parse_args()


peaks_file = args.peaks
nopeaks_file = args.nopeaks
pyBigWig_file = args.bigWig
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
    bigWig = pyBigWig.open(pyBigWig_file)

    chromosomes = {k:v for k,v in zip(fasta.references,fasta.lengths)}#提取染色体长度


    #提取peaks
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
                train_data_pos.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
                # bigWig_data = np.nan_to_num(bigWig.values(chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)),nan=0.0).mean()
                # train_label_pos.append(bigWig_data)
        else:
            if is_standard_chrom(chr) and 0 < (start+peaks_mid-int(seq_len/2)) < chromosomes[chr] and 0 < (start+peaks_mid+int(seq_len/2))<chromosomes[chr]:
                test_data_pos.append([chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)])
                # bigWig_data = np.nan_to_num(bigWig.values(chr,start+peaks_mid-int(seq_len/2),start+peaks_mid+int(seq_len/2)),nan=0.0).mean()
                # test_label_pos.append(bigWig_data)

    nopeaks_regions = []
    nopeaks_file = open(nopeaks_file).readlines()
    for i in nopeaks_file:
        i = i.split()
        if int(i[2]) - int(i[1]) ==slide:
            if is_standard_chrom(i[0]) and  0<(int(i[1])+int(slide/2)-int(seq_len/2))<chromosomes[i[0]] and 0<(int(i[2])-int(slide/2)+int(seq_len/2))<chromosomes[i[0]]:
                nopeaks_regions.append([i[0],int(i[1])+int(slide/2)-int(seq_len/2),int(i[2])-int(slide/2)+int(seq_len/2)])  

    #划分数据集


    train_data_neg = []
    test_data_neg = []
       
    for i in nopeaks_regions:
        if i[0] in valid_chrom:
            test_data_neg.append(i)
        else:
            train_data_neg.append(i)

    train_data = train_data_pos + random.choices(train_data_neg,k=int(len(train_data_pos) * 0.2))


    test_data = test_data_pos

    train_data_onehot = []
    train_foldchange = []
    for i in train_data:
        sequence = fasta.fetch(i[0],i[1],i[2])
        seq_onehot = utils.onehot_seq(sequence)
        train_data_onehot.append(seq_onehot)
        bigWig_data = np.nan_to_num(bigWig.values(i[0],i[1],i[2]),nan=0.0).mean()
        train_foldchange.append(bigWig_data)

    test_data_onehot = []
    test_foldchange = []
    for i in test_data:
        sequence = fasta.fetch(i[0],i[1],i[2])
        seq_onehot = utils.onehot_seq(sequence)
        test_data_onehot.append(seq_onehot)
        bigWig_data = np.nan_to_num(bigWig.values(i[0],i[1],i[2]),nan=0.0).mean()
        test_foldchange.append(bigWig_data)

    train_data = np.array(train_data_onehot)
    test_data = np.array(test_data_onehot)
    train_label = np.array(train_foldchange)
    test_label = np.array(test_foldchange)

    data_outpath = "%s/train_test_foldchange_%s_%s.npz"%(outpath,name,seq_len)
    print("save data to %s"%data_outpath)
    print(len(train_data),len(test_data))
    np.savez(data_outpath,train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label)#,train_cov = train_cov_ls,test_cov=test_cov_ls)

get_train_test_data(peaks_file,nopeaks_file,fasta,seq_len=int(args.len),slide=4096,outpath=args.outpath)
