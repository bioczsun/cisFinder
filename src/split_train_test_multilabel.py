import argparse
import utils
import random
import pysam
from tqdm import tqdm
import numpy as np

def set_random_seed(random_seed = 40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

set_random_seed(40)

params = argparse.ArgumentParser(description='Generate training and testing sets')
params.add_argument("--peaks",help="peaks file",required=True)
params.add_argument("--fasta",help="fasta seq file",required=True)
params.add_argument("--name",help="Basset",required=True)
params.add_argument("--valid_chrom",help="valid_chrom",nargs='+',required=True)
params.add_argument("--outpath",help="output path",required=True)
args = params.parse_args()


peaks_file = args.peaks
fasta = args.fasta
valid_chrom = args.valid_chrom
name = args.name


def is_standard_chrom(chrom):
    if chrom.startswith("chr") and len(chrom) < 6:
        if chrom[3].isdigit() or chrom[3] == "X":
            return True
        else:
            return False
    else:
        return False


def get_train_test_data(peaks_file,fasta,outpath,seq_len=600):
    '''
    peals_file: peaks file
    fasta: fasta file
    seq_length: sequence length
    '''

    fasta = pysam.FastaFile(fasta)

    chromosomes = {k:v for k,v in zip(fasta.references,fasta.lengths)}#提取染色体长度


    #提取peaks
    bed_peaks = open(peaks_file).readlines()[1:]

    train_data = []
    test_data = []

    train_label = []
    test_label = []
    par = tqdm(bed_peaks)
    for line in par:
        line = line.strip().split()
        chr = line[0].split(":")[0]
        start = int(line[0].split(":")[1].split("-")[0])
        target = np.array(line)[1:].astype(int)
        end = start + 600
        if chr not in valid_chrom:
            if is_standard_chrom(chr) and 0 < start < chromosomes[chr] and 0 < (start+seq_len)<chromosomes[chr]:
                sequence = fasta.fetch(chr,start,end)
                seq_onehot = utils.onehot_seq(sequence)
                train_data.append(seq_onehot)
                train_label.append(target)
        else:
            if is_standard_chrom(chr) and 0 < start < chromosomes[chr] and 0 < (start+seq_len)<chromosomes[chr]:
                sequence = fasta.fetch(chr,start,end)
                seq_onehot = utils.onehot_seq(sequence)
                test_data.append(seq_onehot)
                test_label.append(target)

    print(len(train_data),len(test_data))
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    data_outpath = "%s/train_test_%s_%s.npz"%(outpath,name,seq_len)
    print("save data to %s"%data_outpath)

    np.savez(data_outpath,train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label)#,train_cov = train_cov_ls,test_cov=test_cov_ls)

get_train_test_data(peaks_file,fasta,seq_len=600,outpath=args.outpath)
