"""
# Input : A file with all download urls for the dataset and tokenizer choice.
# Output : A folder with tokenized files in .npy format.

url format : <...>/category/file_name.<extension>
tokenized file format : <category>_<file_name>.npy
"""

import os
import time
import gzip
import json
import argparse

import zstandard as zstd
import io

from transformers import AutoTokenizer

import numpy as np

# Helper function
def replace_extension(file_name):
    base_name = file_name.split(".")[0]
    return f"{base_name}.npy"

def get_number_of_documents(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    num_documents = 0

    if ext == '.zst':
        with open(file_path, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    num_documents += 1
    elif ext == '.gz':
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                num_documents += 1
    elif ext == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(file_path)
        num_documents = table.num_rows
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return num_documents

def read_compressed_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.zst':
        with open(file_path, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    data = json.loads(line.rstrip('\n'))
                    document = data["text"]
                    yield document
                    # yield line.rstrip('\n')
    elif ext == '.gz':
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.rstrip('\n'))
                document = data["text"]
                yield document
                # yield line.rstrip('\n')
    elif ext == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(file_path)
        texts = table.column('text')
        for text in texts:
            yield text.as_py()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


parser = argparse.ArgumentParser(description="Tokenize dataset files.")
parser.add_argument("--datasets_dir", type=str, default="datasets/", help="Directory to datasets directory.")
parser.add_argument("--model_choice", type=str, default="allenai/OLMoE-1B-7B-0924", help="Model choice for tokenizer.")
parser.add_argument("--dataset_choice", type=str, default="allenai/OLMoE-mix-0924", help="Dataset choice for tokenizer.")
parser.add_argument("--num_files", type=int, required=True, help="Number of files to tokenize.")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

args = parser.parse_args()

model_tokenizer_map = {
    "allenai/OLMoE-1B-7B-0924": "olmo",
    "allenai/OLMo-1B-hf": "olmo",
    "allenai/OLMo-7B-hf": "olmo",
    "meta-llama/Llama-3.1-8B" : "llama",
    "deepseek-ai/DeepSeek-V3" : "deepseek"
}

assert args.model_choice in model_tokenizer_map, f"Model {args.model_choice} not supported. Supported models are {list(model_tokenizer_map.keys())}"

# Paths
dataset_dir_name = args.dataset_choice.replace("/", "_")
urls_file_path = os.path.join(args.datasets_dir, dataset_dir_name , "urls.txt")
dataset_path = os.path.join(args.datasets_dir, dataset_dir_name, "data")
preprocessed_dataset_path = os.path.join(args.datasets_dir, dataset_dir_name, "preprocessed", model_tokenizer_map[args.model_choice])
tokenized_dataset_path = os.path.join(preprocessed_dataset_path, "tokenized_data")

# Distributed setup
rank = int(os.getenv("PALS_RANKID", "0"))
world_size = int(os.getenv("PMI_SIZE", "1"))

if rank == 0:
    os.makedirs(tokenized_dataset_path, exist_ok=True)

if rank == 0:
    print(f"Dataset choice  : {args.dataset_choice}", flush=True)
    print(f"Model choice    : {args.model_choice}", flush=True)
    print(f"Number of files : {args.num_files}", flush=True)
    print(f"Verbose mode    : {args.verbose}", flush=True)

# Get file names
file_paths = []
tokenized_file_paths = []
with open(urls_file_path, "r") as fh:
    for url in fh:
        tokens = url.strip().split("/")
        category, file_name = tokens[-2], tokens[-1]
        file_path = os.path.join(dataset_path, category, file_name)
        tokenized_file_path = os.path.join(tokenized_dataset_path, category + "_" + replace_extension(file_name))
        file_paths.append(file_path)
        tokenized_file_paths.append(tokenized_file_path)

assert args.num_files <= len(file_paths), f"Only {len(file_paths)} files are present"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_choice)

# Tokenize files
for file_id in range(args.num_files):
    if (file_id % world_size) != rank:
        continue

    file_path = file_paths[file_id]
    tokenized_file_path = tokenized_file_paths[file_id]
    status_file_path = tokenized_file_path.replace(".npy", ".completed")

    if os.path.exists(status_file_path):
        print(f"Rank {rank} : File {file_path} already tokenized. Skipping.", flush=True)
        continue
    
    train_tokens = []
    num_documents = get_number_of_documents(file_path)
    print(f"Rank {rank} : Processing file {file_path}", flush=True)
    print(f"Rank {rank} : Tokenized file will be saved at {tokenized_file_path}", flush=True)
    print(f"Rank {rank} : Number of documents in file {file_path} is {num_documents}", flush=True)
    num_file_tokens = 0
    doc_id = 0
    start_time = time.time()
    for document in read_compressed_file(file_path):
        tokens = tokenizer(document, add_special_tokens=False)["input_ids"]
        tokens.append(tokenizer.eos_token_id)

        # Add tokens to train_tokens        
        train_tokens.extend(tokens)

        # Tracking
        num_file_tokens += len(tokens) - 1

        # Verbose
        doc_id += 1
        if args.verbose and (doc_id % 2000) == 0:
            end_time = time.time()
            eta = ((end_time - start_time) * (num_documents - doc_id) / 2000) / 60
            print(f"Rank {rank} : Documents processed : {doc_id} / {num_documents} , Tokens processed {num_file_tokens}, Time elapsed {(end_time - start_time):.2f} seconds, ETA : {eta:.2f} minutes", flush=True)
            start_time = time.time()
    print(f"Rank {rank} : Processed file {file_path}. Number of file tokens {num_file_tokens}", flush=True)

    # Save train_tokens as numpy array
    print(f"Rank {rank} : Saving file {tokenized_file_path}", flush=True)
    train_tokens_np = np.array(train_tokens, dtype=np.int32)
    np.save(tokenized_file_path, train_tokens_np)
    print(f"Rank {rank} : Saved file {tokenized_file_path}", flush=True)

    # Save completed marker file
    with open(status_file_path, "w") as completed_file:
        completed_file.write("1")
    