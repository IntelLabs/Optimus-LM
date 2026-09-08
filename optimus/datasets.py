import os
import numpy as np

from torch.utils.data import Dataset

class PclDataPrallelDataset(Dataset):
    def __init__(self, dataset_dir, num_data_files, context_size, dp_size=1, dp_ind=0):
        self.context_size = context_size
        self.dp_size = dp_size
        self.dp_ind = dp_ind
        self.base_dp_idx = 0

        # Relevant paths
        urls_file_path = os.path.join(dataset_dir, "../../urls.txt")
        tokenized_dataset_path = os.path.join(dataset_dir, "tokenized_data")

        # Get file paths
        self.tokenized_file_paths = []
        with open(urls_file_path, "r") as fh:
            for url in fh:
                tokens = url.strip().split("/")
                category, file_name = tokens[-2], tokens[-1]
                tokenized_file_path = os.path.join(tokenized_dataset_path, category + "_" + file_name.split(".")[0]+".npy")
                self.tokenized_file_paths.append(tokenized_file_path)

        # Setting the number of data files
        num_data_files = len(self.tokenized_file_paths) if num_data_files == -1 else num_data_files

        self.data_list = [None] * num_data_files
        self.num_instances_list = np.load(os.path.join(dataset_dir, f"num_instances_f{num_data_files}_c{context_size}.npy"))
        self.shuffling_order = np.load(os.path.join(dataset_dir, f"shuffling_order_f{num_data_files}_c{context_size}_1.npy"), mmap_mode='r')

        self.num_instances = np.sum(self.num_instances_list)
        self.cum_instances_list = [0] + list(np.cumsum(self.num_instances_list))

    def set_base_dp_idx(self, base_dp_idx):
        self.base_dp_idx = base_dp_idx

    def __len__(self):
        return (self.num_instances // self.dp_size)

    def __getitem__(self, dp_idx):
        serial_idx = (self.base_dp_idx + dp_idx) * self.dp_size + self.dp_ind
        idx = self.shuffling_order[serial_idx] # Overriding idx 

        # Which file does the idx belong to?
        file_index = None
        for i in range(len(self.cum_instances_list)):
            if idx >= self.cum_instances_list[i] and idx < self.cum_instances_list[i+1]:
                file_index = i
                break

        # Get the data from that file
        local_idx = idx - self.cum_instances_list[file_index]
        start_index = local_idx * self.context_size
        end_index = start_index + self.context_size

        # Lazy file loading
        if self.data_list[file_index] is None:
            self.data_list[file_index] = np.load(self.tokenized_file_paths[file_index], mmap_mode='r')

        data = self.data_list[file_index][start_index:end_index]
        return np.copy(data)

class PclShardedDataParallelDataset(Dataset):
    def __init__(self, dataset_dir, num_data_files, context_size, dp_size=1, dp_ind=0):
        self.context_size = context_size
        self.dp_size = dp_size
        self.dp_ind = dp_ind
        self.base_dp_idx = 0

        # Setting the number of instances per shard
        num_tokens_per_shard = 50 * 256 * 12 * 1 * 4096 # 0.6B shards
        # num_tokens_per_shard = 128 * 128 * 12 * 1 * 4096 # 0.8B shards
        assert num_tokens_per_shard % context_size == 0, "num_tokens_per_shard should be divisible by context_size"
        self.num_instances_per_shard = num_tokens_per_shard // context_size

        # Setting the number of data files
        urls_file_path = os.path.join(dataset_dir, "../../urls.txt")
        num_data_files = len(open(urls_file_path, "r").readlines()) if num_data_files == -1 else num_data_files
        
        # Setting the number of shards
        shard_data_dir = os.path.join(dataset_dir, f"sharded_data_f{num_data_files}_c{context_size}")
        # num_shards = 3671
        # self.sharded_file_paths = [os.path.join(shard_data_dir, f"shard_{i}.npy") for i in range(num_shards)]
        shard_file_names = [f for f in os.listdir(shard_data_dir) if f.endswith('.npy')]
        shard_file_names = sorted(shard_file_names, key=lambda x: int(x.split("_")[1].split(".npy")[0]))
        assert min([int(f.split("_")[1].split(".npy")[0]) for f in shard_file_names]) == 0, "Shards should start from 0"
        assert max([int(f.split("_")[1].split(".npy")[0]) for f in shard_file_names]) == len(shard_file_names) - 1, "Shards should end at num_shards - 1"
        self.sharded_file_paths = [os.path.join(shard_data_dir, f) for f in shard_file_names]
        num_shards = len(self.sharded_file_paths)

        # Setting the number of instances
        num_instances_path = os.path.join(dataset_dir, f"num_instances_f{num_data_files}_c{context_size}.npy")
        self.num_instances_list = np.load(num_instances_path)
        self.num_instances = np.sum(self.num_instances_list)

        # File handles placeholder
        self.data_list = [None] * num_shards

    def __len__(self):
        return (self.num_instances // self.dp_size)

    def set_base_dp_idx(self, base_dp_idx):
        self.base_dp_idx = base_dp_idx

    def __getitem__(self, dp_idx):
        idx = (self.base_dp_idx + dp_idx) * self.dp_size + self.dp_ind
        
        file_index = idx // self.num_instances_per_shard
        local_idx = idx % self.num_instances_per_shard

        # Lazy file loading
        if self.data_list[file_index] is None:
            self.data_list[file_index] = np.load(self.sharded_file_paths[file_index], mmap_mode='r')

        # Get the data from that file
        start_index = local_idx * self.context_size
        end_index = start_index + self.context_size

        data = self.data_list[file_index][start_index:end_index]
        return np.copy(data)