import math
import time
import os
import numpy as np
import sys

import argparse

# HF model to tokenizer mapping (Many to one mapping)
model_tokenizer_map = {
    "allenai/OLMoE-1B-7B-0924": "olmo",
    "allenai/OLMo-1B-hf": "olmo",
    "allenai/OLMo-7B-hf": "olmo",
    "meta-llama/Llama-3.1-8B" : "llama",
    "deepseek-ai/DeepSeek-V3" : "deepseek"
}

# Argument parser
parser = argparse.ArgumentParser(description="Generate metadata for training")
parser.add_argument("--datasets_dir", type=str, default="datasets/", help="Directory to datasets directory.")
parser.add_argument("--model_choice", type=str, default="allenai/OLMoE-1B-7B-0924", help="Model choice for tokenizer.")
parser.add_argument("--dataset_choice", type=str, default="allenai/OLMoE-mix-0924", help="Dataset choice for tokenizer.")
parser.add_argument("--num_files", type=int, required=True, help="Number of files to process")
parser.add_argument("--context_size", type=int, default=2048, help="Context size for training")
parser.add_argument("--tokens_per_shard", type=int, default=(50*256*12*1*4096), help="Number of tokens per shard")
args = parser.parse_args()

assert args.model_choice in model_tokenizer_map, f"Model {args.model_choice} not supported. Supported models are {list(model_tokenizer_map.keys())}"

# Paths
dataset_dir_name = args.dataset_choice.replace("/", "_")
urls_file_path = os.path.join(args.datasets_dir, dataset_dir_name , "urls.txt")
dataset_path = os.path.join(args.datasets_dir, dataset_dir_name, "data")
preprocessed_dataset_path = os.path.join(args.datasets_dir, dataset_dir_name, "preprocessed", model_tokenizer_map[args.model_choice])
tokenized_dataset_path = os.path.join(preprocessed_dataset_path, "tokenized_data")
sharded_dataset_path = os.path.join(preprocessed_dataset_path, f"sharded_data_f{args.num_files}_c{args.context_size}")

# Distributed setup
rank = int(os.getenv("PALS_RANKID", "0"))
world_size = int(os.getenv("PMI_SIZE", "1"))

if rank == 0:
    os.makedirs(sharded_dataset_path, exist_ok=True)

num_tokens_per_shard = int((args.tokens_per_shard // args.context_size ) * args.context_size)
num_instances_per_shard = num_tokens_per_shard // args.context_size

# Get file names
tokenized_file_paths = []
with open(urls_file_path, "r") as fh:
    for url in fh:
        tokens = url.strip().split("/")
        category, file_name = tokens[-2], tokens[-1]
        tokenized_file_path = os.path.join(tokenized_dataset_path, category + "_" + file_name.split(".")[0]+".npy")
        tokenized_file_paths.append(tokenized_file_path)

assert args.num_files <= len(tokenized_file_paths), f"Only {len(tokenized_file_paths)} files are present"

shuffling_order_path = os.path.join(preprocessed_dataset_path, f"shuffling_order_f{args.num_files}_c{args.context_size}_1.npy")
if rank == 0:
    print(f"Loading shuffling_order from {shuffling_order_path}", flush=True)
shuffling_order = np.load(shuffling_order_path)

num_instances_path = os.path.join(preprocessed_dataset_path, f"num_instances_f{args.num_files}_c{args.context_size}.npy")
if rank == 0:
    print(f"Loading num_instances_list from {num_instances_path}", flush=True)
num_instances_list = np.load(num_instances_path)

num_instances = sum(num_instances_list)
num_shards = math.ceil(num_instances / num_instances_per_shard)

cum_num_instances_list = [0] + list(np.cumsum(num_instances_list))
if rank == 0:
    print(f"Number of data files          : {args.num_files}", flush=True)
    print(f"Context size                  : {args.context_size}", flush=True)
    print(f"Number of training tokens (US): {(num_instances * args.context_size)*1e-9:.1f} B", flush=True)
    print(f"Number of tokens per shard    : {num_tokens_per_shard*1e-9:.1f} B", flush=True)
    print(f"Number of shards              : {num_shards}", flush=True)
    print(f"Number of instances per shard : {num_instances_per_shard}", flush=True)
    print(f"Steps on 3K GPU run (BS=1)    : {num_instances_per_shard // 3072}", flush=True)

if rank == 0:
    print("Generating permuted file_ids", flush=True)
file_ids = np.zeros(num_instances, dtype=np.int64)
for file_id in range(args.num_files):
    file_ids[cum_num_instances_list[file_id]:cum_num_instances_list[file_id+1]] = file_id
perm_file_ids = file_ids[shuffling_order]

prev_time = time.time()

# File handles
data_list = [None] * args.num_files

for shard_id in range(num_shards):
    if shard_id % world_size != rank:
        continue
    
    shard_file_path = os.path.join(sharded_dataset_path, f"shard_{shard_id}.npy")
    status_file_path = shard_file_path.replace(".npy", ".completed")

    if os.path.exists(status_file_path):
        print(f"Rank {rank} : Shard {shard_id} already generated. Skipping.", flush=True)
        continue

    # Create and write a shard
    start_inst_id = shard_id * num_instances_per_shard
    end_inst_id = min(num_instances, (shard_id+1) * num_instances_per_shard)
    shard_num_instances = end_inst_id - start_inst_id

    print(f"Rank {rank} : Generating shard {shard_id}", flush=True)
    
    # Creating shard shard_id
    shard_tokens = np.zeros(shard_num_instances*args.context_size, dtype=np.int32)
    for local_inst_id in range(shard_num_instances):
        inst_id = start_inst_id + local_inst_id

        # Find shfl_inst_id in the files
        shfl_inst_id = shuffling_order[inst_id]
        file_index = perm_file_ids[inst_id]

        if data_list[file_index] is None:
            data_list[file_index] = np.load(tokenized_file_paths[file_index], mmap_mode='r')

        file_inst_id = shfl_inst_id - cum_num_instances_list[file_index]
        shard_tokens[local_inst_id*args.context_size:(local_inst_id+1)*args.context_size] = \
            data_list[file_index][file_inst_id*args.context_size:(file_inst_id+1)*args.context_size]

        if local_inst_id % 1000 == 0:
            time_in_min = ((time.time() - prev_time))/(60)
            print(f"Rank {rank} : Processed {local_inst_id} / {shard_num_instances} for shard {shard_id}, ETC : {((shard_num_instances / 1000) * time_in_min)/60:.2f} hr", flush=True)
            prev_time = time.time()

    # Save the shard
    print(f"Rank {rank} : Saving shard file {shard_id}", flush=True)
    np.save(shard_file_path, shard_tokens)
    print(f"Rank {rank} : Saved shard file {shard_id} ", flush=True)

    # Save completed marker file
    with open(status_file_path, "w") as completed_file:
        completed_file.write("1")