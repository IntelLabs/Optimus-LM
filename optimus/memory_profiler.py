import torch

from .mapper import ParallelDpPpEpTpMapper

class MemoryLogInfo():
    def __init__(self, tag, init=False):
        self.tag = tag
        self.memory_allocated = torch.xpu.memory_allocated() if not init else 0
        self.memory_reserved = torch.xpu.memory_reserved() if not init else 0
        self.memory_peak = torch.xpu.max_memory_allocated() if not init else 0

class MemoryTracker():
    def __init__(self, pmap=None):
        self.pmap = pmap if pmap is not None else ParallelDpPpEpTpMapper()
        self.log_list = [MemoryLogInfo("Init", init=True)]

    def log(self, tag):
        cur_entry = MemoryLogInfo(tag)
        self.log_list.append(cur_entry)
        self.print_current_entry(self.log_list[-1], self.log_list[-2])
    
    def print_current_entry(self, cur_entry, prev_entry):
        diff_alloc = cur_entry.memory_allocated - prev_entry.memory_allocated
        diff_resv = cur_entry.memory_reserved - prev_entry.memory_reserved
        diff_peak = cur_entry.memory_peak - prev_entry.memory_peak
        # Filtering
        filter_condition = (self.pmap.dp_ind == 0) and (self.pmap.ep_ind == 0) and (self.pmap.pp_ind == 0)
        # filter_condition = (self.pmap.dp_ind == 0) and (self.pmap.ep_ind == 0) and (self.pmap.is_first_stage_rank or self.pmap.is_last_stage_rank)
        # filter_condition = (self.pmap.dp_ind == 0) and (self.pmap.ep_ind == 0)
        # filter_condition = True
        if filter_condition:
            print(f"dp{self.pmap.dp_ind:03d}-pp{self.pmap.pp_ind:02d}-ep{self.pmap.ep_ind:02d} {cur_entry.tag:25s}, {cur_entry.memory_allocated/1024**2:8.2f} MB, {cur_entry.memory_reserved/1024**2:8.2f} MB, {cur_entry.memory_peak/1024**2:8.2f} MB, {diff_alloc/1024**2:8.2f} MB, {diff_resv/1024**2:8.2f} MB, {diff_peak/1024**2:8.2f} MB", flush=True)

    def print_all_entries(self):
        print(f"{'Tag':25s}, {'Alloc':>7s}, {'Resv':>7s}, {'Peak':>7s}, {'DiffA':>7s}, {'DiffR':>7s}, {'DiffP':>7s}")
        prev = self.log_list[0]
        for log in self.log_list[1:]:
            diff_alloc = log.memory_allocated - prev.memory_allocated
            diff_resv = log.memory_reserved - prev.memory_reserved
            diff_peak = log.memory_peak - prev.memory_peak
            print(f"{log.tag:25s}, {log.memory_allocated/1024**2:8.2f} MB, {log.memory_reserved/1024**2:8.2f} MB, {log.memory_peak/1024**2:8.2f} MB, {diff_alloc/1024**2:8.2f} MB, {diff_resv/1024**2:8.2f} MB, {diff_peak/1024**2:8.2f} MB")
            prev = log