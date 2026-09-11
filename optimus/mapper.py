import os
import torch
import itertools

class ShapeND:
    def __init__(self, sizes, permute_order=None):
        self.dim = len(sizes)
        self.permute_order = permute_order if permute_order != None else list(range(self.dim))
        assert len(self.permute_order) == self.dim
        self.sizes = [sizes[p] for p in self.permute_order]

    def get_flat_index(self, *args):
        assert len(args) == self.dim
        index = args
        # Permute index before flattening
        index = [index[p] for p in self.permute_order] 
        flat_index = 0
        multiplier = 1
        for i in range(self.dim-1, -1, -1):
            flat_index += index[i] * multiplier
            multiplier *= self.sizes[i]
        return flat_index

    def get_index(self, flat_index):
        index = [0] * self.dim
        for i in range(self.dim-1, -1, -1):
            index[i] = flat_index % self.sizes[i]
            flat_index = flat_index // self.sizes[i]
        # Permute index after unflattening
        index = tuple([index[p] for p in self.permute_order])
        return index
        
    def __str__(self):
        orig_shapes =  [self.sizes[p] for p in self.permute_order]
            
        shape_str = "Shape : ["
        for i in range(self.dim):
            shape_str += f"{orig_shapes[i]}"
            if i != self.dim-1:
                shape_str += " "
        shape_str += "]"
        return shape_str

class ParallelDpTpMapper:
    def __init__(self, 
        data_parallelism = 1,
        tensor_parallelism = 1,
        rank=0, 
        create_groups=False):
        # Parallelims
        self.data_parallelism = data_parallelism
        self.tensor_parallelism = tensor_parallelism

        # Shorthands
        self.DP = data_parallelism
        self.TP = tensor_parallelism

        self.rank = rank
        self.permute_order = [0,1]
        self.shape = ShapeND([self.DP, self.TP], self.permute_order)

        self.dp_ind, self.tp_ind = self.shape.get_index(rank)
        self.mp_ind = self.tp_ind

        self.node = None
        self.rank_to_node_list = []
        if torch.distributed.is_initialized():
            job_id = os.getenv("PBS_JOBID", "0").split(".")[0]
            launch_info_dir = os.path.join("/lus/flare/projects/Intel-Aurora/dvooturi/launch_info/", job_id)
            hostfile_path = os.path.join(launch_info_dir, "hostfile")
            if os.path.exists(hostfile_path):
                # Reading hostfile to get node list
                nodes = open(hostfile_path, 'r').readlines()
                nodes = [line.strip() for line in nodes if line.strip()]
                # Setting node for current rank
                world_size = torch.distributed.get_world_size()
                local_world_size = world_size // len(nodes)
                self.node = nodes[self.rank // local_world_size]
                # Setting rank to node list to enable soft node failure detection
                for r in range(world_size):
                    self.rank_to_node_list.append(nodes[r // local_world_size])                

        # Required process groups
        if not create_groups:
            self.dp_group = -1
            self.tp_group = -1
        else:
            self.create_groups()

    def create_groups(self):
        self.dp_group = self.create_and_get_dp_group()
        self.tp_group = self.create_and_get_tp_group()

    def get_index(self):
        return f"(DP: {self.dp_ind}, TP : {self.tp_ind})"

    def get_rank_with_index(self):
        return f"Rank : {self.rank} ({self.get_index()})"

    def create_and_get_dp_group(self):
        group_ranks_list = []
        groups_list = []
        for t in range(self.TP):
            group_ranks = [self.shape.get_flat_index(d, t) for d in range(self.DP)]
            group = torch.distributed.new_group(group_ranks)
            group_ranks_list.append(group_ranks)
            groups_list.append(group)
        
        group_id = self.tp_ind
        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group
    
    def create_and_get_tp_group(self):
        group_ranks_list = []
        groups_list = []
        for d in range(self.DP):
            group_ranks = [self.shape.get_flat_index(d, t) for t in range(self.TP)]
            group = torch.distributed.new_group(group_ranks)
            group_ranks_list.append(group_ranks)
            groups_list.append(group)
        
        group_id = self.dp_ind 
        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group

    def __str__(self):
        string = f"******* Process map information for rank {self.rank} *******" + "\n"
        string += f"Node : {self.node}" + "\n"
        string += f"Rank : {self.rank}, (dp_ind : {self.dp_ind}, tp_ind : {self.tp_ind})" + "\n"

        def get_print_string(group, name):
            data = f"{name} group ranks : "
            data += f"{torch.distributed.get_process_group_ranks(group)}" if group != -1 else "[-1]"
            data += "\n"
            return data

        string += get_print_string(self.tp_group, "TP")
        string += get_print_string(self.dp_group, "DP")
        string += "*"*50
        
        return string

class ParallelDpEpMapper:
    def __init__(self, 
        data_parallelism = 1,
        expert_parallelism = 1,
        rank=0, 
        create_groups=False):
        # Parallelims
        self.data_parallelism = data_parallelism
        self.expert_parallelism = expert_parallelism

        # Shorthands
        self.DP = data_parallelism
        self.EP = expert_parallelism

        self.rank = rank
        self.permute_order = [0,1]
        self.shape = ShapeND([self.DP, self.EP], self.permute_order)

        self.dp_ind, self.ep_ind = self.shape.get_index(rank)
        self.mp_ind = self.ep_ind

        self.node = None
        self.rank_to_node_list = []
        if torch.distributed.is_initialized():
            job_id = os.getenv("PBS_JOBID", "0").split(".")[0]
            launch_info_dir = os.path.join("/lus/flare/projects/Intel-Aurora/dvooturi/launch_info/", job_id)
            hostfile_path = os.path.join(launch_info_dir, "hostfile")
            if os.path.exists(hostfile_path):
                # Reading hostfile to get node list
                nodes = open(hostfile_path, 'r').readlines()
                nodes = [line.strip() for line in nodes if line.strip()]
                # Setting node for current rank
                world_size = torch.distributed.get_world_size()
                local_world_size = world_size // len(nodes)
                self.node = nodes[self.rank // local_world_size]
                # Setting rank to node list to enable soft node failure detection
                for r in range(world_size):
                    self.rank_to_node_list.append(nodes[r // local_world_size])                

        # Required process groups
        if not create_groups:
            self.dp_group = -1
            self.ep_group = -1
        else:
            self.create_groups()

    def create_groups(self):
        self.dp_group = self.create_and_get_dp_group()
        self.ep_group = self.create_and_get_ep_group()

    def get_index(self):
        return f"(DP: {self.dp_ind}, EP : {self.ep_ind})"

    def get_rank_with_index(self):
        return f"Rank : {self.rank} ({self.get_index()})"

    def create_and_get_dp_group(self):
        group_ranks_list = []
        groups_list = []
        for e in range(self.EP):
            group_ranks = [self.shape.get_flat_index(d, e) for d in range(self.DP)]
            group = torch.distributed.new_group(group_ranks)
            group_ranks_list.append(group_ranks)
            groups_list.append(group)
        
        group_id = self.ep_ind
        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group
    
    def create_and_get_ep_group(self):
        group_ranks_list = []
        groups_list = []
        for d in range(self.DP):
            group_ranks = [self.shape.get_flat_index(d, e) for e in range(self.EP)]
            group = torch.distributed.new_group(group_ranks)
            group_ranks_list.append(group_ranks)
            groups_list.append(group)

        group_id = self.dp_ind
        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group


    def __str__(self):
        string = f"******* Process map information for rank {self.rank} *******" + "\n"
        string += f"Node : {self.node}" + "\n"
        string += f"Rank : {self.rank}, (dp_ind : {self.dp_ind}, ep_ind : {self.ep_ind})" + "\n"

        def get_print_string(group, name):
            data = f"{name} group ranks : "
            data += f"{torch.distributed.get_process_group_ranks(group)}" if group != -1 else "[-1]"
            data += "\n"
            return data

        string += get_print_string(self.ep_group, "EP")
        string += get_print_string(self.dp_group, "DP")
        string += "*"*50
        
        return string

class ParallelDpEpTpMapper:
    def __init__(self, 
        data_parallelism = 1,
        expert_parallelism = 1,
        tensor_parallelism = 1,
        rank=0, 
        create_groups=False):
        # Parallelims
        self.data_parallelism = data_parallelism
        self.expert_parallelism = expert_parallelism
        self.tensor_parallelism = tensor_parallelism

        # Shorthands
        self.DP = data_parallelism
        self.EP = expert_parallelism
        self.TP = tensor_parallelism

        self.rank = rank
        self.permute_order = [0,1,2]
        self.shape = ShapeND([self.DP, self.EP, self.TP], self.permute_order)

        self.dp_ind, self.ep_ind, self.tp_ind = self.shape.get_index(rank)
        self.dpep_ind = self.dp_ind * self.EP + self.ep_ind
        
        self.is_model_parallel = (self.EP > 1) or (self.TP > 1)
        self.mp_ind = self.ep_ind * self.TP + self.tp_ind

        self.node = None
        self.rank_to_node_list = []
        if torch.distributed.is_initialized():
            job_id = os.getenv("PBS_JOBID", "0").split(".")[0]
            launch_info_dir = os.path.join("/lus/flare/projects/Intel-Aurora/dvooturi/launch_info/", job_id)
            hostfile_path = os.path.join(launch_info_dir, "hostfile")
            if os.path.exists(hostfile_path):
                # Reading hostfile to get node list
                nodes = open(hostfile_path, 'r').readlines()
                nodes = [line.strip() for line in nodes if line.strip()]
                # Setting node for current rank
                world_size = torch.distributed.get_world_size()
                local_world_size = world_size // len(nodes)
                self.node = nodes[self.rank // local_world_size]
                # Setting rank to node list to enable soft node failure detection
                for r in range(world_size):
                    self.rank_to_node_list.append(nodes[r // local_world_size])                

        # Required process groups
        if not create_groups:
            self.dp_group = -1
            self.ep_group = -1
            self.tp_group = -1
            self.dpep_group = -1
        else:
            self.create_groups()

    def create_groups(self):
        self.dp_group = self.create_and_get_dp_group()
        self.ep_group = self.create_and_get_ep_group()
        self.tp_group = self.create_and_get_tp_group()

        self.dpep_group = self.create_and_get_dpep_group()

    def get_index(self):
        return f"(DP: {self.dp_ind}, EP : {self.ep_ind}, TP : {self.tp_ind})"

    def get_rank_with_index(self):
        return f"Rank : {self.rank} ({self.get_index()})"

    def create_and_get_dp_group(self):
        group_ranks_list = []
        groups_list = []
        for e in range(self.EP):
            for t in range(self.TP):
                group_ranks = [self.shape.get_flat_index(d, e, t) for d in range(self.DP)]
                group = torch.distributed.new_group(group_ranks)
                group_ranks_list.append(group_ranks)
                groups_list.append(group)
        
        group_id = self.ep_ind * self.TP + self.tp_ind
        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group
    
    def create_and_get_ep_group(self):
        group_ranks_list = []
        groups_list = []
        for d in range(self.DP):
            for t in range(self.TP):
                group_ranks = [self.shape.get_flat_index(d, e, t) for e in range(self.EP)]
                group = torch.distributed.new_group(group_ranks)
                group_ranks_list.append(group_ranks)
                groups_list.append(group)
        
        group_id = self.dp_ind * self.TP + self.tp_ind
        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group

    def create_and_get_tp_group(self):
        group_ranks_list = []
        groups_list = []
        for d in range(self.DP):
            for e in range(self.EP):
                group_ranks = [self.shape.get_flat_index(d, e, t) for t in range(self.TP)]
                group = torch.distributed.new_group(group_ranks)
                group_ranks_list.append(group_ranks)
                groups_list.append(group)
        
        group_id = self.dp_ind * self.EP + self.ep_ind
        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group
    
    def create_and_get_dpep_group(self):
        group_ranks_list = []
        groups_list = []
        for t in range(self.TP):
            group_ranks = [self.shape.get_flat_index(d, e, t) for d in range(self.DP) for e in range(self.EP)]
            group = torch.distributed.new_group(group_ranks)
            group_ranks_list.append(group_ranks)
            groups_list.append(group)
        
        group_id = self.tp_ind
        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group
    

    def __str__(self):
        string = f"******* Process map information for rank {self.rank} *******" + "\n"
        string += f"Node : {self.node}" + "\n"
        string += f"Rank : {self.rank}, (dp_ind : {self.dp_ind}, ep_ind : {self.ep_ind}, tp_ind : {self.tp_ind})" + "\n"

        def get_print_string(group, name):
            data = f"{name} group ranks : "
            data += f"{torch.distributed.get_process_group_ranks(group)}" if group != -1 else "[-1]"
            data += "\n"
            return data

        string += get_print_string(self.tp_group, "TP")
        string += get_print_string(self.ep_group, "EP")
        string += get_print_string(self.dp_group, "DP")
        string += get_print_string(self.dpep_group, "DPEP")
        string += "*"*50
        
        return string

class ParallelDpPpEpTpMapper:
    def __init__(self, 
        data_parallelism = 1,
        pipeline_parallelism = 1,
        expert_parallelism = 1,
        tensor_parallelism = 1,
        rank=0, 
        create_groups=False,
        use_pp_first = False):
        # Parallelims
        self.data_parallelism = data_parallelism
        self.pipeline_parallelism = pipeline_parallelism
        self.expert_parallelism = expert_parallelism
        self.tensor_parallelism = tensor_parallelism

        # Shorthands
        self.DP = data_parallelism
        self.PP = pipeline_parallelism
        self.EP = expert_parallelism
        self.TP = tensor_parallelism

        self.rank = rank
        self.permute_order = [0,1,2,3] if (not use_pp_first) else [1,0,2,3]
        self.shape = ShapeND([self.DP, self.PP, self.EP, self.TP], self.permute_order)

        self.dp_ind, self.pp_ind, self.ep_ind, self.tp_ind = self.shape.get_index(rank)
        self.dpep_ind = self.dp_ind * (self.EP) + self.ep_ind
        
        self.is_model_parallel = (self.PP > 1) or (self.EP > 1) or (self.TP > 1)
        self.mp_ind = self.pp_ind * (self.EP * self.TP) + self.ep_ind * (self.TP) + self.tp_ind
        self.mp_rank = self.rank - self.dp_ind * (self.PP * self.EP * self.TP)

        # Pipeline related helper arguments
        self.is_first_stage_rank = self.pp_ind == 0
        self.is_last_stage_rank = self.pp_ind == self.PP-1
        self.pp_ind_reverse = (self.PP - 1) - self.pp_ind

        self.prev_stage_rank = self.shape.get_flat_index(self.dp_ind, (self.pp_ind-1)%self.PP, self.ep_ind, self.tp_ind)
        self.next_stage_rank = self.shape.get_flat_index(self.dp_ind, (self.pp_ind+1)%self.PP, self.ep_ind, self.tp_ind)
        self.first_stage_ranks = [self.shape.get_flat_index(self.dp_ind, 0, e, t) for e in range(self.EP) for t in range(self.TP)]
        self.last_stage_ranks  = [self.shape.get_flat_index(self.dp_ind, self.PP-1, e, t) for e in range(self.EP) for t in range(self.TP)]
        self.all_stage_ranks = []
        for p in range(self.PP):
            cur_stage_ranks = [self.shape.get_flat_index(self.dp_ind, p, e, t) for e in range(self.EP) for t in range(self.TP)]
            self.all_stage_ranks.append(cur_stage_ranks)

        self.node = None
        self.rank_to_node_list = []
        if torch.distributed.is_initialized():
            job_id = os.getenv("PBS_JOBID", "0").split(".")[0]
            launch_info_dir = os.path.join("/lus/flare/projects/Intel-Aurora/dvooturi/launch_info/", job_id)
            hostfile_path = os.path.join(launch_info_dir, "hostfile")
            if os.path.exists(hostfile_path):
                # Reading hostfile to get node list
                nodes = open(hostfile_path, 'r').readlines()
                nodes = [line.strip() for line in nodes if line.strip()]
                # Setting node for current rank
                world_size = torch.distributed.get_world_size()
                local_world_size = world_size // len(nodes)
                self.node = nodes[self.rank // local_world_size]
                # Setting rank to node list to enable soft node failure detection
                for r in range(world_size):
                    self.rank_to_node_list.append(nodes[r // local_world_size])                

        # Required process groups
        if not create_groups:
            self.dp_group = -1
            self.tp_group = -1
            self.ep_group = -1
            self.dpep_group = -1
        else:
            self.create_groups()

    def create_groups(self):
        self.dp_group = self.create_and_get_dp_group()
        self.tp_group = self.create_and_get_tp_group()
        self.ep_group = self.create_and_get_ep_group()
        self.dpep_group = self.create_and_get_dpep_group()

    def get_index(self):
        return f"(DP: {self.dp_ind}, EP : {self.ep_ind}, TP : {self.tp_ind})"

    def get_rank_with_index(self):
        return f"Rank : {self.rank} ({self.get_index()})"

    def create_and_get_dp_group(self):
        group_ranks_list = []
        groups_list = []

        for p, e, t in itertools.product(range(self.PP), range(self.EP), range(self.TP)):
            group_ranks = [self.shape.get_flat_index(d, p, e, t) for d in range(self.DP)]
            group = torch.distributed.new_group(group_ranks)
            group_ranks_list.append(group_ranks)
            groups_list.append(group)
        
        group_id = self.pp_ind * (self.EP * self.TP) + \
                   self.ep_ind * (self.TP) + \
                   self.tp_ind

        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group
    
    def create_and_get_tp_group(self):
        group_ranks_list = []
        groups_list = []
        for d, p, e in itertools.product(range(self.DP), range(self.PP), range(self.EP)):
            group_ranks = [self.shape.get_flat_index(d, p, e, t) for t in range(self.TP)]
            group = torch.distributed.new_group(group_ranks)
            group_ranks_list.append(group_ranks)
            groups_list.append(group)
        
        group_id = self.dp_ind * (self.PP * self.EP) + \
                   self.pp_ind * (self.EP) + \
                   self.ep_ind
        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group

    def create_and_get_ep_group(self):
        group_ranks_list = []
        groups_list = []
        for d, p, t in itertools.product(range(self.DP), range(self.PP), range(self.TP)):
            group_ranks = [self.shape.get_flat_index(d, p, e, t) for e in range(self.EP)]
            group = torch.distributed.new_group(group_ranks)
            group_ranks_list.append(group_ranks)
            groups_list.append(group)
        
        group_id = self.dp_ind * (self.PP * self.TP) + \
                   self.pp_ind * (self.TP) + \
                   self.tp_ind

        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group
    
    def create_and_get_dpep_group(self):
        group_ranks_list = []
        groups_list = []
        for p,t in itertools.product(range(self.PP), range(self.TP)):
            group_ranks = [self.shape.get_flat_index(d, p, e, t) for d in range(self.DP) for e in range(self.EP)]
            group = torch.distributed.new_group(group_ranks)
            group_ranks_list.append(group_ranks)
            groups_list.append(group)
        
        group_id = self.pp_ind * (self.TP) + \
                   self.tp_ind

        group = groups_list[group_id] if self.rank in group_ranks_list[group_id] else -1
        return group

    def __str__(self):
        string = f"******* Process map information for rank {self.rank} *******" + "\n"
        string += f"Node : {self.node}" + "\n"
        string += f"Rank : {self.rank}, (dp_ind : {self.dp_ind}, pp_ind : {self.pp_ind}, ep_ind : {self.ep_ind}, tp_ind : {self.tp_ind})" + "\n"

        def get_print_string(group, name):
            data = f"{name} group ranks : "
            data += f"{torch.distributed.get_process_group_ranks(group)}" if group != -1 else "[-1]"
            data += "\n"
            return data

        string += get_print_string(self.tp_group, "TP")
        string += get_print_string(self.ep_group, "EP")
        string += get_print_string(self.dp_group, "DP")
        string += get_print_string(self.dpep_group, "DPEP")

        string += f"First/Last stage ranks : {self.first_stage_ranks, self.last_stage_ranks}" + "\n"
        string += f"All stage ranks : {self.all_stage_ranks}" + "\n"

        string += "*"*50
        
        return string
