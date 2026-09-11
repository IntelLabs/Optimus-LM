#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: bash $0 <dataset_dir>"
    exit 1
fi

dataset_dir=$1

# Directory to store the dataset files
data_dir="$dataset_dir/data"
mkdir -p $data_dir

sub_datasets="algebraic-stack dclm open-web-math pes2o starcoder wiki"

PARALLEL_DOWNLOADS=8

for sub_dataset in $sub_datasets; do
    echo "Downloading $sub_dataset"
    # Directory to store sub_dataset files
    sub_dataset_dir=$data_dir/$sub_dataset
    mkdir -p $sub_dataset_dir
    cat $dataset_dir/urls_${sub_dataset}.txt | xargs -n 1 -P $PARALLEL_DOWNLOADS wget -c -P $sub_dataset_dir
done

echo "OLMoE-mix-0924 dataset download completed."
