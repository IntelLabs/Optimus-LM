import os
import argparse
import numpy as np

np.random.seed(42)

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
parser.add_argument("--num_orders", type=int, default=1, help="That many shuffling orders will be generated")
args = parser.parse_args()

assert args.model_choice in model_tokenizer_map, f"Model {args.model_choice} not supported. Supported models are {list(model_tokenizer_map.keys())}"

# Paths
dataset_dir_name = args.dataset_choice.replace("/", "_")
urls_file_path = os.path.join(args.datasets_dir, dataset_dir_name , "urls.txt")
preprocessed_dataset_path = os.path.join(args.datasets_dir, dataset_dir_name, "preprocessed", model_tokenizer_map[args.model_choice])
tokenized_dataset_path = os.path.join(preprocessed_dataset_path, "tokenized_data")

# Get file paths
tokenized_file_paths = []
with open(urls_file_path, "r") as fh:
    for url in fh:
        tokens = url.strip().split("/")
        category, file_name = tokens[-2], tokens[-1]
        tokenized_file_path = os.path.join(tokenized_dataset_path, category + "_" + file_name.split(".")[0]+".npy")
        tokenized_file_paths.append(tokenized_file_path)

assert args.num_files <= len(tokenized_file_paths), f"Only {len(tokenized_file_paths)} files are present"

#In each file, leave tokens(file) % (context_size)
print("Get number of instances")
num_instances_list = []
for file_id in range(args.num_files):
    tokenized_file_path = tokenized_file_paths[file_id]
    print(f"Processing {tokenized_file_path}")
    file_data_mmap = np.load(tokenized_file_path, mmap_mode='r')
    file_num_instances = (file_data_mmap.shape[0]) // (args.context_size)
    num_instances_list.append(file_num_instances)

num_instances = sum(num_instances_list)
print(f"Number of training instances : {num_instances}", flush=True)
print(f"Number of training tokens    : {(num_instances * args.context_size)*1e-9:.1f} B tokens")

for i in range(args.num_orders):    
    print(f"Generating shuffling order {i + 1}")
    indices = np.arange(num_instances)
    shuffling_order = np.random.permutation(indices)

    shuffling_order_path = os.path.join(preprocessed_dataset_path, f"shuffling_order_f{args.num_files}_c{args.context_size}_{i+1}.npy")
    print(f"Saving shuffling order at {shuffling_order_path}")
    np.save(shuffling_order_path, shuffling_order)
    print("Saved shuffling order")

num_instances_path = os.path.join(preprocessed_dataset_path, f"num_instances_f{args.num_files}_c{args.context_size}.npy")
print(f"Saving instance counts at {num_instances_path}")
np.save(num_instances_path, np.array(num_instances_list))
print("Saved instance counts order")