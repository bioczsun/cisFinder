import argparse
import os
import utils
import random
import pysam
import pyBigWig
import numpy as np
import pandas as pd
from collections import namedtuple
from multiprocessing import Pool

from tqdm import tqdm

def set_random_seed(random_seed=40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

set_random_seed(40)

params = argparse.ArgumentParser(description='Generate training and testing sets')
params.add_argument("--peaks", help="peaks file", required=True)
params.add_argument("--target", help="nopeaks file", required=True)
# params.add_argument("--fasta", help="fasta seq file", required=True)
params.add_argument("--outpath", help="output path", required=True)
args = params.parse_args()

peaks_file = open(args.peaks).readlines()
target_file = open(args.target).readlines()[1:]
# fasta = args.fasta

def is_standard_chrom(chrom):
    if chrom.startswith("chr") and len(chrom) < 6:
        if chrom[3].isdigit() or chrom[3] == "X":
            return True
        else:
            return False
    else:
        return False

Contig = namedtuple('Contig', ['chrom', 'start', 'end', 'name'])

def read(bigwig, chrm, start, end):
    cov_open = bigwig
    cov = cov_open.values(chrm, start, end, numpy=True).astype('float16')
    return cov

def read_bw_frombigwig(bigwig, contigs, file_name):
    cov_open = pyBigWig.open(bigwig, 'r')
    cov_ls = []
    for contig in contigs:
        chrom, start, end, name = contig
        signal = read(cov_open, chrom, int(start), int(end))
        signal_mean = signal.mean()
        signal_mean = np.nan_to_num(signal_mean, nan=0.0)
        cov_ls.append(signal_mean)
    np.save(file_name, np.array(cov_ls))

def process_target_file_line(line):
    line = line.strip().split()
    bigwig_acc = line[0]
    bigwig_file = os.path.join("/home/suncz/work/part1/code/data", bigwig_acc + ".bigWig")
    bed_acc = line[1]
    read_bw_frombigwig(bigwig_file, contigs, os.path.join(args.outpath, bed_acc + ".npy"))

# Prepare contigs
contigs = []
for line in peaks_file:
    chrom, start, end, name = line.strip().split()
    if is_standard_chrom(chrom):
        contig = Contig(chrom, start, end, name)
        contigs.append(contig)

# Use multiprocessing with 40 processes to process target file lines in parallel
if __name__ == "__main__":
    with Pool(40) as pool:
        for _ in tqdm(pool.imap_unordered(process_target_file_line, target_file), total=len(target_file)):
            pass
