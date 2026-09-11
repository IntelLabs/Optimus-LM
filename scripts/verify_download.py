import os
import argparse
import requests

parser = argparse.ArgumentParser(description="Verify URLs in dataset files.")
parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset files.")
parser.add_argument("--file_start_id", type=int, default=0, help="Starting index for file verification.")
parser.add_argument("--file_end_id", type=int, default=-1, help="Ending index for file verification (exclusive).")
args = parser.parse_args()

# Get urls and file paths
urls_path = os.path.join(args.dataset_dir, "urls.txt")
urls = []
data_file_paths = []
with open(urls_path, 'r') as file:
    for line in file:
        url = line.strip()
        urls.append(url)
        tokens = url.split("/")
        category, file_name = tokens[-2], tokens[-1]
        file_path = os.path.join(args.dataset_dir, "data", category, file_name)
        data_file_paths.append(file_path)


num_data_files = len(urls)
if args.file_end_id == -1:
    args.file_end_id = num_data_files

assert args.file_end_id <= num_data_files, f"file_end_id {args.file_end_id} exceeds number of URLs {num_data_files}"

for file_id in range(args.file_start_id, args.file_end_id):
    url, file_path = urls[file_id], data_file_paths[file_id]
    # print(f"Processing url {url} at file path {file_path}")
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code == 200:
            remote_size = int(response.headers.get('Content-Length', 0))
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
            elif os.path.getsize(file_path) != remote_size:
                print(f"File size mismatch for {file_path}: local size = {os.path.getsize(file_path)}, remote size = {remote_size}")
        else:
            print(f"Failed to fetch size for {url}, status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error checking URL {url}: {e}")

# python3 verify_download.py --dataset_dir datasets/allenai_OLMoE-mix-0924/ --file_start_id 0 --file_end_id 16 | tee log.txt