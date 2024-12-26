# cisFinder
cisFinder (Sequence Function Model Comprehensive Evaluator) is a framework for the evaluation of genomic deep learning models.  In addition to assessing model prediction performance, this framework also evaluates:
- The impact of input sequence length.
- The selection of positive and negative samples.
- Predictions across different sequencing depths.
- The interpretability of learned representations.
- The prediction of variant effects in high-significance regions. 

## Getting started
Installation (Please install anaconda first)
```shell
git clone https://github.com/bioczsun/SeqFMER.git
conda create -n SeqFMER python=3.11
conda activate SeqFMER
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install meme captum jupyter pandas pysam bedtools bedops pybigwig scikit-learn matplotlib -c conda-forge -c bioconda
```
## Usage

### Data preprocessing
```shell
cd SeqFMER/data/ref

# Download hg38.fa
wget "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.fa.gz" 

gzip -d hg38.fa.gz && cd ..

# Generates a bed file with 4096bp windows
bedtools makewindows -g ref/hg38.chrom.size -w 4096 > ref/hg38_4096.bed

# Sorts the bed file
sort -k1,1 -k2,2n ref/hg38_4096.bed > ref/hg38_4096_sorted.bed
```


### Generation of negative samples
```shell
bedops -n 1 ref/hg38_4096_sorted.bed GM12878/ENCFF470YYO.bed > GM12878/nopeaks.bed;

cd .. && mkdir -p train_results/GM12878
```

### Segmentation of dataset
```shell
python src/split_train_test.py \
    --peaks data/GM12878/ENCFF470YYO.bed \
    --nopeaks data/GM12878/nopeaks.bed \
    --fasta data/ref/hg38.fa \
    --len 600 \
    --outpath data/GM12878
```


### Evaluation of the model performances at a sequence length of 600bp (binary classification)
Scope of --model: DanQ ExplaiNN Basset DeepSEA SATORI CNN_Transformer CNN_Attention CNN
```shell
#DanQ
for activate in relu exp;do
    python src/train_binary.py \
        --data data/GM12878/train_test_allneg_600.npz \
        --model DanQ \
        --linear_units 29440 \
        --activate $activate \
        --device cuda:0 \
        --fasta data/ref/hg38.fa \
        --outpath train_results/GM12878 \
        --seqlen 600 --seed 40 \
        --batch 256 --lr 0.0001;
done

#ExplaiNN
for activate in relu exp;do
    python src/train_binary.py \
        --data data/GM12878/train_test_allneg_600.npz \
        --model ExplaiNN \
        --linear_units 600 \
        --activate $activate \
        --device cuda:0 \
        --fasta data/ref/hg38.fa \
        --outpath train_results/GM12878 \
        --seqlen 600 --seed 40 \
        --batch 256 --lr 0.0001;
done

#Basset
for activate in relu exp;do
    python src/train_binary.py \
        --data data/GM12878/train_test_allneg_600.npz \
        --model Basset \
        --linear_units 14600 \
        --activate $activate \
        --device cuda:0 \
        --fasta data/ref/hg38.fa \
        --outpath train_results/GM12878 \
        --seqlen 600 --seed 40 \
        --batch 256 --lr 0.0001;
done
```

### Evaluation of integration models
```shell
python src/train_binary_hybrid.py \
    --data data/GM12878/train_test_allneg_600.npz \
    --model DanQ_ExplaiNN \
    --linear_units 14600 \
    --device cuda:0 \
    --fasta data/ref/hg38.fa \
    --outpath train_results/GM12878 \
    --activate exp \
    --seqlen 600 --seed 40 \
    --batch 128 --lr 0.0001;
```



### Model interpretability
```shell
python src/explain.py \
    --data data/GM12878/train_test_allneg_600.npz \
    --model Basset \
    --activate relu \
    --device cuda:0 \
    --fasta data/ref/hg38.fa \
    --model_dir train_results/GM12878 \
    --outpath evaluate_explainer/meme \
    --phase GM12878 \
    --seqlen 600 --seed 40 \
    --batch 4096 --lr 0.0001;

mkdir -p evaluate_explainer/meme/600/GM12878/relu/Basset

tomtom -no-ssc evaluate_explainer/meme/600/motif_GM12878_Basset_relu.meme \
    -verbosity 2 \
    -min-overlap 5 \
    -dist pearson \
    -evalue \
    -thresh 10.0 \
    -oc evaluate_explainer/meme/600/GM12878/relu/Basset \
    -eps data/ref/JASPAR2022_CORE_vertebrates_non-redundant_v2.meme
```
