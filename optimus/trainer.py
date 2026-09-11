import time

import torch

from .dutils import print_rank_0
from .utils import print_debug

from optimus.models.common.ral_functions import single_input_load_balancing_loss_func, layer_level_load_balancing_loss_func

from inspect import currentframe
def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

class ParallelTrainerOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

def get_coord_info_from_flat_index(index, D1, D2, D3, flip_d2=False):
    if index < 0:
        return None, None, None, None
    d1 = index // (D2 * D3)
    d2d3 = index % (D2 * D3)
    d2 = d2d3 // D3
    d3 = d2d3 % D3
    if flip_d2:
        d2 = (D2 - 1) - d2
    d1d3 = d1 * D3 + d3
    return d1, d2, d3, d1d3

def get_flat_index_from_coord_info(d1, d2, d3, D1, D2, D3):
    index = d1 * (D2 * D3) + d2 * D3 + d3
    return index

class Config():
    def __init__(self,
        num_hidden_layers=36,
        hidden_size=4096,
        pipeline_parallelism=1,
        virtual_pipeline_parallelism=1):  
    
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.pipeline_parallelism = pipeline_parallelism
        self.virtual_pipeline_parallelism = virtual_pipeline_parallelism

        assert self.num_hidden_layers % self.pipeline_parallelism == 0, "num_hidden_layers must be divisible by pipeline_parallelism"
        assert (self.num_hidden_layers // self.pipeline_parallelism) % self.virtual_pipeline_parallelism == 0, "(num_hidden_layers / pipeline_parallelism) must be divisible by virtual_pipeline_parallelism"

    def __repr__(self):
        config_dict = self.__dict__
        lines = ['Config : {']
        for k, v in config_dict.items():
            lines.append(f'    "{k}": {repr(v)},')
        lines.append('}')
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()

class Model(torch.nn.Module):
    def __init__(self, config, pmap=None, scale_output=False):
        super(Model, self).__init__()
        self.config = config
        self.pmap = pmap
        self.scale_output = scale_output

        self.LP = config.num_hidden_layers // config.pipeline_parallelism
        self.LV = self.LP // config.virtual_pipeline_parallelism        
        self.layers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(self.LP)])

    def forward(self, x, vp_ind=0):
        for i in range(vp_ind*self.LV, (vp_ind+1)*self.LV):
            x = self.layers[i](x)
            if self.scale_output:
                x = x / self.layers[i].in_features # TEMPORARY DEBUG
            x = torch.nn.functional.relu(x)
        return x
    
    def set_parameters_from_full_module(self, full_module):
        for l in range(len(self.layers)):
            v = l // self.LV
            lv = l % self.LV
            gl = (v * self.config.pipeline_parallelism + self.pmap.pp_ind) * self.LV + lv
            # print(f"Rank : {self.pmap.rank}, {l}->{gl}", flush=True)
            self.layers[l].load_state_dict(full_module.layers[gl].state_dict())

class Interleaved1f1bBufferManager():
    def __init__(self, pmap, VPP, PP, NMB, reuse_buffers_in_1f1b=False):
        assert NMB % PP == 0, "In Interleaved1f1b schedule, the number of micro batches should be a multip of pipeline parallelism"
        self.pmap = pmap
        self.VPP = VPP
        self.NMB = NMB
        self.reuse_buffers_in_1f1b = reuse_buffers_in_1f1b
        self.NMBG = PP
        self.NG = NMB // PP
        
        if self.reuse_buffers_in_1f1b:
            self.num_act_buffers = (((self.VPP * self.NMBG) - self.pmap.pp_ind) + self.pmap.pp_ind_reverse) + 1 # Adding one extra
            self.num_ograd_buffers = 1
        else:
            self.num_act_buffers = self.VPP * self.NMB
            self.num_ograd_buffers = self.VPP * self.NMB
        
        self.fwd_indices_in_flight = [None] * self.num_act_buffers
        self.fwd_indices_in_drain = []
    
    def get_backward_buffer_id(self, index):
        # Ignode None index
        if index is None:
            return None
        
        # Ignore out of bound indices
        if (index < 0) or (index >= (self.VPP * self.NMB)):
            return None
        
        if self.reuse_buffers_in_1f1b:
            return 0
        else:
            return index

    def get_forward_buffer_id(self, index):
        # Ignore None index
        if index is None:
            return None

        # Ignore out of bound indices
        if (index < 0) or (index >= (self.VPP * self.NMB)):
            return None
        
        # Find the buffer assigned to this forward index
        buf_id = None
        for i in range(self.num_act_buffers):
            if self.fwd_indices_in_flight[i] == index:
                buf_id = i
                break

        if buf_id == None:
            assert False, f"Forward index {index} is not in flight. (Forwards in flight {self.fwd_indices_in_flight})"
        return buf_id

    def get_forward_index_from_backward_index(self, bwd_index):
        # Ignore None index
        if bwd_index == None:
            return None

        # Check bounds
        if (bwd_index < 0) or (bwd_index >= (self.VPP * self.NMB)):
            return None
        
        bwd_mg, bwd_vp, bwd_lmb, bwd_mb = get_coord_info_from_flat_index(bwd_index, self.NG, self.VPP, self.NMBG, flip_d2=True)
        index = get_flat_index_from_coord_info(bwd_mg, bwd_vp, bwd_lmb, self.NG, self.VPP, self.NMBG)

        return index

    def add_forward_to_flight(self, index):
        # Ignore out of bound indices
        if (index < 0) or (index >= (self.VPP * self.NMB)):
            return

        # Find an available slot
        buf_id = None
        for i in range(self.num_act_buffers):
            if self.fwd_indices_in_flight[i] == None:
                buf_id = i
                break
        
        # If no slots are available, pop from drain list
        if (buf_id == None) and (len(self.fwd_indices_in_drain) > 0):
            fwd_index_drain = self.fwd_indices_in_drain.pop(0)
            buf_id = self.get_forward_buffer_id(fwd_index_drain)
        
        if (buf_id != None):
            self.fwd_indices_in_flight[buf_id] = index
        else:
            assert False, "Issue with the logic for number of required buffers."
    
    def drain_forward_from_flight(self, index):
        if self.reuse_buffers_in_1f1b:
            if index != None:
                self.fwd_indices_in_drain.append(index)    

class ParallelTrainer:
    def __init__(self,
        config, model, pmap, 
        batch_size, micro_batch_size, context_size, opt_grad_acc_steps,
        dtype, device, pp_scheme = "gpipe", reuse_buffers_in_1f1b = True,
        verbose=False, memory_tracker=None):

        self.dtype = dtype
        self.device = device

        self.model = model
        self.config = config
        self.pmap = pmap

        self.pp_scheme = pp_scheme
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.context_size = context_size
        self.opt_grad_acc_steps = opt_grad_acc_steps
        assert batch_size % micro_batch_size == 0, "Batch size must be divisible by micro batch size"
        self.num_micro_batches = batch_size // micro_batch_size

        # Shorthands
        self.NMB = self.num_micro_batches # Shorthand
        self.PP = self.config.pipeline_parallelism

        # Model specific
        self.vocab_size = config.vocab_size

        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_hidden_layers_per_rank = config.num_hidden_layers // config.pipeline_parallelism
        self.num_hidden_layers_per_vpp_rank = self.num_hidden_layers_per_rank // config.virtual_pipeline_parallelism

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.use_aux_loss = config.output_router_logits
        self.use_global_router_aux_loss = config.output_router_logits and (not config.use_local_router_aux_loss)
        self.use_local_router_aux_loss = config.output_router_logits and (config.use_local_router_aux_loss)

        # Buffer reuse
        self.reuse_buffers_in_1f1b = reuse_buffers_in_1f1b

        act_buffer_shape = (micro_batch_size, context_size, config.hidden_size)
        self.input_act_buffers = []
        self.output_act_buffers = []
        self.output_grad_buffers = []

        if (self.pp_scheme == "gpipe"):
            for v in range(self.config.virtual_pipeline_parallelism):
                self.input_act_buffers.append(
                    [torch.zeros(act_buffer_shape, dtype=dtype, device=device, requires_grad=True) for i in range(self.num_micro_batches)]
                )
                self.output_act_buffers.append(
                    [None for i in range(self.num_micro_batches)]
                )
                self.output_grad_buffers.append(
                    [torch.zeros(act_buffer_shape, dtype=dtype, device=device) for i in range(self.num_micro_batches)]
                )
        else:
            PP = self.config.pipeline_parallelism
            VPP = self.config.virtual_pipeline_parallelism
            NMB = self.num_micro_batches
            NMBG = PP
            assert NMB % PP == 0, "For 1f1b pp_scheme, num_micro_batches must be divisible by pipeline_parallelism"

            num_act_buffers = (((VPP * NMBG) - self.pmap.pp_ind) + self.pmap.pp_ind_reverse + 1) if self.reuse_buffers_in_1f1b else (VPP * NMB)
            num_ograd_buffers = 1 if self.reuse_buffers_in_1f1b else (VPP * NMB)

            if ((self.pmap.dp_ind == 0) and (self.pmap.ep_ind == 0)):
                print(f"Rank {self.pmap.rank}: Allocating {num_act_buffers} input/output activation buffers and {num_ograd_buffers} output gradient buffers for 1f1b pp_scheme (Reuse Buffers: {self.reuse_buffers_in_1f1b})", flush=True)

            # Right now allocating everything regardless of reuse.
            self.input_act_buffers = [torch.zeros(act_buffer_shape, dtype=dtype, device=device, requires_grad=True) for i in range(num_act_buffers)]
            self.output_act_buffers = [None for i in range(num_act_buffers)]
            self.output_grad_buffers = [torch.zeros(act_buffer_shape, dtype=dtype, device=device) for i in range(num_ograd_buffers)]

            ##################################### MoE related ###############################
            # Assumption : No buffer reuse across VPP stages
            self.output_router_logits_buffers = [None for i in range(num_act_buffers)]
            self.prev_router_logits_buffers = [[None] * NMB for v in range(VPP)]
            self.all_router_logits_grad_buffers = [[None] * NMB for v in range(VPP)]
            self.prev_router_logits_grad_buffers = [[None] * NMB for v in range(VPP)] # References to all_router_logits_grad_buffers except the last part

            for v in range(VPP):
                if not ((self.pmap.is_first_stage_rank) and (v == 0)):
                    requires_grad = (self.pmap.is_last_stage_rank and (v == (VPP - 1)))
                    prev_nl = (v * PP + self.pmap.pp_ind) * self.num_hidden_layers_per_vpp_rank
                    prev_router_logits_shape = (prev_nl * micro_batch_size * context_size, self.config.num_experts)
                    for mb in range(NMB):
                        self.prev_router_logits_buffers[v][mb] = torch.zeros(prev_router_logits_shape, dtype=dtype, device=device, requires_grad=requires_grad)
                
                nl = (v * PP + (self.pmap.pp_ind+1)) * self.num_hidden_layers_per_vpp_rank
                all_router_logits_shape = (nl * micro_batch_size * context_size, self.config.num_experts)
                for mb in range(NMB):
                    self.all_router_logits_grad_buffers[v][mb] = torch.zeros(all_router_logits_shape, dtype=dtype, device=device)
                    self.prev_router_logits_grad_buffers[v][mb] = self.all_router_logits_grad_buffers[v][mb][:-1*(self.num_hidden_layers_per_vpp_rank*micro_batch_size*context_size)]
            ################################################################################

        self.verbose = verbose
        self.profiler = None
        self.memory_tracker = memory_tracker

    def loss_function(self, logits, labels, vocab_size):
        # Choosing first (S-1) for logits and last (S-1) for labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Cross entropy loss
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        return loss    
    
    def calculate_router_aux_loss(self, id, mb, vp):
        aux_loss = None
        if self.use_local_router_aux_loss:
            aux_loss = layer_level_load_balancing_loss_func(
                self.output_router_logits_buffers[id],
                self.num_experts, 
                self.num_experts_per_tok
            )
            # loss += (self.router_aux_loss_coef * aux_loss)
        
        if self.use_global_router_aux_loss:
            prev_router_logits_list = [self.prev_router_logits_buffers[vp][mb]] if (self.prev_router_logits_buffers[vp][mb] != None) else []
            cur_router_logits_list = [layer_gate for layer_gate in self.output_router_logits_buffers[id]]
            all_router_logits_list = prev_router_logits_list + cur_router_logits_list
            all_router_logits = torch.cat(all_router_logits_list, dim=0)
            aux_loss = single_input_load_balancing_loss_func(all_router_logits, self.num_experts, self.num_experts_per_tok)
            # loss += (self.router_aux_loss_coef * aux_loss)

        return aux_loss

    def send_router_logits_wrapper(self, id, mb, vp):
        if self.use_global_router_aux_loss:
            prev_router_logits_list = [self.prev_router_logits_buffers[vp][mb]] if (self.prev_router_logits_buffers[vp][mb] != None) else []
            cur_router_logits_list = [layer_gate for layer_gate in self.output_router_logits_buffers[id]]
            all_router_logits_list = prev_router_logits_list + cur_router_logits_list
            all_router_logits = torch.cat(all_router_logits_list, dim=0)
            torch.distributed.send(all_router_logits, self.pmap.next_stage_rank)

    def recv_router_logits_wrapper(self, mb, vp):
        if self.use_global_router_aux_loss:
            torch.distributed.recv(self.prev_router_logits_buffers[vp][mb], self.pmap.prev_stage_rank)

    def send_router_logits_grad_wrapper(self, mb, vp):
        if self.use_global_router_aux_loss:
            torch.distributed.send(self.prev_router_logits_grad_buffers[vp][mb], self.pmap.prev_stage_rank)

    def recv_router_logits_grad_wrapper(self, mb, vp):
        if self.use_global_router_aux_loss:
            torch.distributed.recv(self.all_router_logits_grad_buffers[vp][mb], self.pmap.next_stage_rank)

    def get_ral_accounted_scalar_loss_for_backward(self, oa_id, og_id, mb, vp):
        assert self.use_local_router_aux_loss or self.use_global_router_aux_loss
        
        if self.use_local_router_aux_loss:
            loss = torch.sum(self.output_act_buffers[oa_id] * self.output_grad_buffers[og_id]) # Dummy loss to avoid two backwards
            aux_loss = layer_level_load_balancing_loss_func(
                self.output_router_logits_buffers[oa_id],
                self.num_experts, 
                self.num_experts_per_tok
            )
            loss += (self.router_aux_loss_coef * aux_loss)
            # loss.backward()
        
        if self.use_global_router_aux_loss:
            loss = torch.sum(self.output_act_buffers[oa_id] * self.output_grad_buffers[og_id]) # Dummy loss to avoid two backwards

            aux_loss = None
            for layer_id,layer_gate in enumerate(self.output_router_logits_buffers[oa_id]):
                base_ind  = (vp * self.PP + self.pmap.pp_ind) * (self.num_hidden_layers_per_vpp_rank * self.micro_batch_size * self.context_size)
                start_ind = base_ind + (layer_id) * (self.micro_batch_size * self.context_size)
                end_ind   = base_ind + (layer_id+1) * (self.micro_batch_size * self.context_size)
                layer_gate_grad_buffer = self.all_router_logits_grad_buffers[vp][mb][start_ind:end_ind]
                if aux_loss is None:
                    aux_loss = torch.sum(layer_gate * layer_gate_grad_buffer)
                else:
                    aux_loss += torch.sum(layer_gate * layer_gate_grad_buffer)
            
            loss += aux_loss
            # loss.backward()
        return loss

    def step(self, input=None, labels=None, only_forward=False, output_grad=None, output_logits=False, debug=False):
        # Shorthands to enhance readability
        NMB = self.num_micro_batches
        MBS = self.micro_batch_size
        VPP = self.config.virtual_pipeline_parallelism
        PP = self.config.pipeline_parallelism        

        # Separating input into a list of micro batches
        input_list = [None for mb in range(NMB)]
        if self.pmap.is_first_stage_rank:
            assert input != None, "Input should be provided for the first stage ranks"
            input_list = [input[mb*MBS:(mb+1)*MBS] for mb in range(NMB)]
        
        # Separating either labels or output_grad into a list of micro batches
        labels_list = [None for mb in range(NMB)]
        output_grad_list = [None for mb in range(NMB)]
        if self.pmap.is_last_stage_rank:
            assert (labels != None) ^ (output_grad != None), "Either labels or output_grad should be provided for the last stage ranks"
            if labels != None:
                assert output_grad == None, "Output grads should not be provided for the last stage ranks when labels are provided."
                labels_list = [labels[mb*MBS:(mb+1)*MBS] for mb in range(NMB)]
            if output_grad != None:
                assert labels == None, "Labels should not be provided for the last stage ranks when output_grad is provided."
                output_grad_list = [output_grad[mb*MBS:(mb+1)*MBS] for mb in range(NMB)]
        
        # Creating loss list for the last stage ranks
        loss_list = [None for mb in range(NMB)]

        # Logits list for the last stage ranks (If output_logits is set to True)
        logits_list = [None for mb in range(NMB)]

        if self.pp_scheme == "gpipe":
            ############################# WARMUP-F #########################################
            for i in range(NMB - self.pmap.pp_ind):
                mb, vp_ind = i % NMB, i // NMB

                if (not self.pmap.is_first_stage_rank):
                    # All non-first pipeline stages receives buffer from previous pipeline stage
                    torch.distributed.recv(self.input_act_buffers[vp_ind][mb], self.pmap.prev_stage_rank)
                    print_debug(f"WUF Rank {self.pmap.rank:2d} : RA{mb}V{vp_ind}, {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)

                # Forward pass
                if self.pmap.is_first_stage_rank:
                    self.output_act_buffers[vp_ind][mb] = self.model(input_list[mb], vp_ind=vp_ind)
                else:
                    self.output_act_buffers[vp_ind][mb] = self.model(self.input_act_buffers[vp_ind][mb], vp_ind=vp_ind)
                print_debug(f"WUF Rank {self.pmap.rank:2d} : F{mb}V{vp_ind}", debug)
                torch.xpu.synchronize()

                if self.pmap.is_last_stage_rank and (vp_ind == (VPP-1)):
                    if output_logits:
                        logits_list[mb] = self.output_act_buffers[vp_ind][mb].detach().clone()
                    if labels != None:
                        # Calculate loss if labels are provided
                        logits = self.output_act_buffers[vp_ind][mb]
                        loss = self.loss_function(logits, labels_list[mb], self.vocab_size)

                        self.output_act_buffers[vp_ind][mb] = loss
                        loss_list[mb] = self.output_act_buffers[vp_ind][mb].detach().clone()
                        self.output_act_buffers[vp_ind][mb].div_(NMB*self.opt_grad_acc_steps) # Scaling loss by GAS

                if (not self.pmap.is_last_stage_rank):
                    # All non-last pipeline stages sends buffer to the next pipeline stage
                    torch.distributed.send(self.output_act_buffers[vp_ind][mb], self.pmap.next_stage_rank)
                    print_debug(f"WUF Rank {self.pmap.rank:2d} : SA{mb}V{vp_ind}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)
            ################################################################################

            ####################### STEADY-F ###############################################
            i_start = NMB - self.pmap.pp_ind
            i_end = i_start + (NMB * (VPP - 1))
            for i in range(i_start, i_end):
                mb, vp_ind = i % NMB, i // NMB
                
                torch.distributed.recv(self.input_act_buffers[vp_ind][mb], self.pmap.prev_stage_rank)
                print_debug(f"SSF Rank {self.pmap.rank:2d} : RA{mb}V{vp_ind}, {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)
                
                if self.pmap.is_last_stage_rank:
                    prev_i = i - (NMB - PP + 1)
                    prev_mb, prev_vp_ind = prev_i % NMB, prev_i // NMB
                    torch.distributed.send(self.output_act_buffers[prev_vp_ind][prev_mb], self.pmap.next_stage_rank)
                    print_debug(f"SSF Rank {self.pmap.rank:2d} : SA{prev_mb}V{prev_vp_ind}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)

                # Forward pass
                self.output_act_buffers[vp_ind][mb] = self.model(self.input_act_buffers[vp_ind][mb], vp_ind=vp_ind)
                torch.xpu.synchronize()
                print_debug(f"SSF Rank {self.pmap.rank:2d} : F{mb}V{vp_ind}", debug)

                if self.pmap.is_last_stage_rank and (vp_ind == (VPP-1)):
                    if output_logits:
                        logits_list[mb] = self.output_act_buffers[vp_ind][mb].detach().clone()
                    if labels != None:
                        # Calculate loss if labels are provided
                        logits = self.output_act_buffers[vp_ind][mb]
                        loss = self.loss_function(logits, labels_list[mb], self.vocab_size)

                        self.output_act_buffers[vp_ind][mb] = loss
                        loss_list[mb] = self.output_act_buffers[vp_ind][mb].detach().clone()
                        self.output_act_buffers[vp_ind][mb].div_(NMB*self.opt_grad_acc_steps) # Scaling loss by GAS

                if not self.pmap.is_last_stage_rank:
                    torch.distributed.send(self.output_act_buffers[vp_ind][mb], self.pmap.next_stage_rank)
                    print_debug(f"SSF Rank {self.pmap.rank:2d} : SA{mb}V{vp_ind}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)
            ################################################################################

            ################## WINDDOWN-F ##################################################
            i_start = (NMB * VPP) - self.pmap.pp_ind
            i_end = (NMB * VPP)
            for i in range(i_start, i_end):
                mb, vp_ind = i % NMB, i // NMB
                
                if not self.pmap.is_first_stage_rank:
                    torch.distributed.recv(self.input_act_buffers[vp_ind][mb], self.pmap.prev_stage_rank)
                    print_debug(f"WDF Rank {self.pmap.rank:2d} : RA{mb}V{vp_ind}, {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)

                # Forward pass
                self.output_act_buffers[vp_ind][mb] = self.model(self.input_act_buffers[vp_ind][mb], vp_ind=vp_ind)
                torch.xpu.synchronize()
                print_debug(f"WDF Rank {self.pmap.rank:2d} : F{mb}V{vp_ind}", debug)

                if self.pmap.is_last_stage_rank and (vp_ind == (VPP-1)):
                    if output_logits:
                        logits_list[mb] = self.output_act_buffers[vp_ind][mb].detach().clone()
                    if labels != None:
                        # Calculate loss if labels are provided
                        logits = self.output_act_buffers[vp_ind][mb]
                        loss = self.loss_function(logits, labels_list[mb], self.vocab_size)

                        self.output_act_buffers[vp_ind][mb] = loss
                        loss_list[mb] = self.output_act_buffers[vp_ind][mb].detach().clone()
                        self.output_act_buffers[vp_ind][mb].div_(NMB*self.opt_grad_acc_steps) # Scaling loss by GAS

                if not self.pmap.is_last_stage_rank:
                    torch.distributed.send(self.output_act_buffers[vp_ind][mb], self.pmap.next_stage_rank)
                    print_debug(f"WDF Rank {self.pmap.rank:2d} : SA{mb}V{vp_ind}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)
            ################################################################################
            
            if not only_forward:
                ############################# WARMUP-B #########################################
                for i in range(NMB - self.pmap.pp_ind_reverse):
                    mb, vp_ind = i % NMB, ((VPP - 1) - (i // NMB))

                    if (not self.pmap.is_last_stage_rank):
                        # All non-last pipeline stages receives buffer from next pipeline stage
                        torch.distributed.recv(self.output_grad_buffers[vp_ind][mb], self.pmap.next_stage_rank)
                        print_debug(f"WUB Rank {self.pmap.rank:2d} : RG{mb}V{vp_ind}, {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)

                    # Backward pass
                    if self.pmap.is_last_stage_rank:
                        self.output_act_buffers[vp_ind][mb].backward(output_grad_list[mb])
                    else:
                        self.output_act_buffers[vp_ind][mb].backward(self.output_grad_buffers[vp_ind][mb])
                    torch.xpu.synchronize()
                    print_debug(f"WUB Rank {self.pmap.rank:2d} : B{mb}V{vp_ind}", debug)

                    if (not self.pmap.is_first_stage_rank):
                        # All non-first pipeline stages sends buffer to previous pipeline stage
                        torch.distributed.send(self.input_act_buffers[vp_ind][mb].grad, self.pmap.prev_stage_rank)
                        print_debug(f"WUB Rank {self.pmap.rank:2d} : SG{mb}V{vp_ind}, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
                ################################################################################

                ############################# STEADY-B #########################################
                i_start = NMB - self.pmap.pp_ind_reverse
                i_end = i_start + (NMB * (VPP - 1))
                for i in range(i_start, i_end):
                    mb, vp_ind = i % NMB, ((VPP-1) - (i // NMB))

                    torch.distributed.recv(self.output_grad_buffers[vp_ind][mb], self.pmap.next_stage_rank)
                    print_debug(f"SSB Rank {self.pmap.rank:2d} : RG{mb}V{vp_ind}, {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)

                    if self.pmap.is_first_stage_rank:
                        prev_i = i - (NMB - PP + 1)
                        prev_mb, prev_vp_ind = prev_i % NMB, ((VPP-1) - (prev_i // NMB))
                        torch.distributed.send(self.input_act_buffers[prev_vp_ind][prev_mb].grad, self.pmap.prev_stage_rank)
                        print_debug(f"SSB Rank {self.pmap.rank:2d} : SG{prev_mb}V{prev_vp_ind}, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)

                    # Backward pass
                    self.output_act_buffers[vp_ind][mb].backward(self.output_grad_buffers[vp_ind][mb])
                    torch.xpu.synchronize()
                    print_debug(f"SSB Rank {self.pmap.rank:2d} : B{mb}V{vp_ind}", debug)

                    if not self.pmap.is_first_stage_rank:
                        # All non-first pipeline stages sends buffer to previous pipeline stage
                        torch.distributed.send(self.input_act_buffers[vp_ind][mb].grad, self.pmap.prev_stage_rank)
                        print_debug(f"SSB Rank {self.pmap.rank:2d} : SG{mb}V{vp_ind}, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
                ################################################################################

                ############################# WINDDOWN-B #######################################
                i_start = (NMB * VPP) - self.pmap.pp_ind_reverse
                i_end = (NMB * VPP)
                for i in range(i_start, i_end):
                    mb, vp_ind = i % NMB, ((VPP - 1) - (i // NMB))
                    
                    if (not self.pmap.is_last_stage_rank):
                        # All non-last pipeline stages receives buffer from next pipeline stage
                        torch.distributed.recv(self.output_grad_buffers[vp_ind][mb], self.pmap.next_stage_rank)
                        print_debug(f"WDB Rank {self.pmap.rank:2d} : RG{mb}V{vp_ind}, {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)

                    # Backward pass
                    self.output_act_buffers[vp_ind][mb].backward(self.output_grad_buffers[vp_ind][mb])
                    torch.xpu.synchronize()
                    print_debug(f"WDB Rank {self.pmap.rank:2d} : B{mb}V{vp_ind}", debug)

                    if (not self.pmap.is_first_stage_rank):
                        # All non-first pipeline stages sends buffer to previous pipeline stage
                        torch.distributed.send(self.input_act_buffers[vp_ind][mb].grad, self.pmap.prev_stage_rank)
                        print_debug(f"WDB Rank {self.pmap.rank:2d} : SG{mb}V{vp_ind}, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
                ################################################################################
        elif self.pp_scheme == "1f1b":
            assert NMB % PP == 0, "For 1f1b pp_scheme, num_micro_batches must be divisible by pipeline_parallelism"
            NG = NMB // PP # Number of micro batch groups
            NMBG = PP # Number of micro batches in a group

            def check_bounds(index):
                return (index >= 0) and (index < (NMB * VPP))
                
            def get_output_grad_buffer_id(index):
                mg, vp, lmb, mb = get_coord_info_from_flat_index(index, NG, VPP, NMBG, flip_d2=True)
                bid = (0 if (self.reuse_buffers_in_1f1b) else (vp * NMB + mb)) if check_bounds(index) else None
                return bid

            buffer_manager = Interleaved1f1bBufferManager(self.pmap, VPP, PP, NMB, self.reuse_buffers_in_1f1b)

            ############################# WARMUP ###########################################
            num_warmup_steps = (NMBG * VPP)
            fwd_flat_index = 0 - self.pmap.pp_ind
            for t in range(num_warmup_steps):
                prev_fwd_flat_index = (fwd_flat_index - 1)
                fwd_mg, fwd_vp, fwd_lmb, fwd_mb = get_coord_info_from_flat_index(fwd_flat_index, NG, VPP, NMBG)
                prev_fwd_mg, prev_fwd_vp, prev_fwd_lmb, prev_fwd_mb = get_coord_info_from_flat_index(prev_fwd_flat_index, NG, VPP, NMBG)

                # Buffer mangement
                buffer_manager.add_forward_to_flight(fwd_flat_index)
                ia_buf_id = buffer_manager.get_forward_buffer_id(fwd_flat_index)
                oa_buf_id_circ = buffer_manager.get_forward_buffer_id(prev_fwd_flat_index)
                oa_buf_id = buffer_manager.get_forward_buffer_id(fwd_flat_index)

                # Forward comms (Receive Activation)
                if check_bounds(fwd_flat_index):
                    if (not self.pmap.is_first_stage_rank) or (self.pmap.is_first_stage_rank and (fwd_vp != 0)):
                        print_debug(f"Line : {get_linenumber()}, T{t} WU Rank {self.pmap.rank} : B4-RA{fwd_mb}V{fwd_vp} to IAB{ia_buf_id}, {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)
                        torch.distributed.recv(self.input_act_buffers[ia_buf_id], self.pmap.prev_stage_rank)
                        self.recv_router_logits_wrapper(fwd_mb, fwd_vp)
                        print_debug(f"Line : {get_linenumber()}, T{t} WU Rank {self.pmap.rank} : A4-RA{fwd_mb}V{fwd_vp} to IAB{ia_buf_id}, {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)

                # Forward comms (Send Activation Previous)
                if check_bounds(prev_fwd_flat_index):
                    if (self.pmap.is_last_stage_rank and (prev_fwd_vp != (VPP-1))):
                        print_debug(f"Line : {get_linenumber()}, T{t} WU Rank {self.pmap.rank} : B4-SA{prev_fwd_mb}V{prev_fwd_vp} 4m OAB{oa_buf_id_circ}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)
                        torch.distributed.send(self.output_act_buffers[oa_buf_id_circ], self.pmap.next_stage_rank)
                        self.send_router_logits_wrapper(oa_buf_id_circ, prev_fwd_mb, prev_fwd_vp)
                        print_debug(f"Line : {get_linenumber()}, T{t} WU Rank {self.pmap.rank} : A4-SA{prev_fwd_mb}V{prev_fwd_vp} 4m OAB{oa_buf_id_circ}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)

                # Forward pass
                if check_bounds(fwd_flat_index):
                    if self.pmap.is_first_stage_rank and (fwd_vp == 0):
                        output_dict = self.model(input_list[fwd_mb], vp_ind=fwd_vp)
                        print_debug(f"Line : {get_linenumber()}, T{t} WU Rank {self.pmap.rank} : CMP-F{fwd_mb}V{fwd_vp} 4m ISL{fwd_mb} to OAB{oa_buf_id}", debug)
                    else:
                        output_dict = self.model(self.input_act_buffers[ia_buf_id], vp_ind=fwd_vp)
                        print_debug(f"Line : {get_linenumber()}, T{t} WU Rank {self.pmap.rank} : CMP-F{fwd_mb}V{fwd_vp} 4m IAB{ia_buf_id} to OAB{oa_buf_id}", debug)
                    self.output_act_buffers[oa_buf_id] = output_dict["output"]
                    self.output_router_logits_buffers[oa_buf_id] = output_dict["router_logits"]
                    torch.xpu.synchronize()

                    if self.pmap.is_last_stage_rank and (fwd_vp == (VPP-1)):
                        if output_logits:
                            logits_list[fwd_mb] = self.output_act_buffers[oa_buf_id].detach().clone()

                        if labels != None:
                            # Calculate loss if labels are provided
                            logits = self.output_act_buffers[oa_buf_id]
                            loss = self.loss_function(logits, labels_list[fwd_mb], self.vocab_size)
                            # Calculate auxiliary loss when applicable
                            aux_loss = self.calculate_router_aux_loss(oa_buf_id, fwd_mb, fwd_vp)
                            if aux_loss is not None:
                                loss += (self.router_aux_loss_coef * aux_loss)

                            self.output_act_buffers[oa_buf_id] = loss
                            loss_list[fwd_mb] = self.output_act_buffers[oa_buf_id].detach().clone()
                            self.output_act_buffers[oa_buf_id].div_(NMB*self.opt_grad_acc_steps) # Scaling loss by GAS

                # Forward comms (Send Activation)
                if check_bounds(fwd_flat_index):
                    if (not self.pmap.is_last_stage_rank) and (t != (num_warmup_steps-1)):
                        print_debug(f"Line : {get_linenumber()}, T{t} WU Rank {self.pmap.rank} : B4-SA{fwd_mb}V{fwd_vp} 4m OAB{oa_buf_id}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)
                        torch.distributed.send(self.output_act_buffers[oa_buf_id], self.pmap.next_stage_rank)
                        self.send_router_logits_wrapper(oa_buf_id, fwd_mb, fwd_vp)
                        print_debug(f"Line : {get_linenumber()}, T{t} WU Rank {self.pmap.rank} : A4-SA{fwd_mb}V{fwd_vp} 4m OAB{oa_buf_id}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)

                # Updating to next forward block
                fwd_flat_index += 1
            ################################################################################
            
            ############################# STEADY ###########################################
            num_steady_steps = 2 * (((NG-1)*(NMBG*VPP)) + (NMBG - 1))
            fwd_flat_index = (NMBG * VPP) - self.pmap.pp_ind
            bwd_flat_index = 0 - self.pmap.pp_ind_reverse
            for st in range(num_steady_steps):
                t = (num_warmup_steps + st) # Global time step
                if (st % 2 == 0):
                    prev_fwd_flat_index = (fwd_flat_index - 1)
                    prev_bwd_flat_index = (bwd_flat_index - 1)
                    bwd_mg, bwd_vp, bwd_lmb, bwd_mb = get_coord_info_from_flat_index(bwd_flat_index, NG, VPP, NMBG, flip_d2=True)
                    prev_fwd_mg, prev_fwd_vp, prev_fwd_lmb, prev_fwd_mb = get_coord_info_from_flat_index(prev_fwd_flat_index, NG, VPP, NMBG)
                    prev_bwd_mg, prev_bwd_vp, prev_bwd_lmb, prev_bwd_mb = get_coord_info_from_flat_index(prev_bwd_flat_index, NG, VPP, NMBG, flip_d2=True)
                    
                    # Buffer management
                    fwd_eq_bwd_flat_index = buffer_manager.get_forward_index_from_backward_index(bwd_flat_index)
                    fwd_eq_prev_bwd_flat_index = buffer_manager.get_forward_index_from_backward_index(prev_bwd_flat_index)
                    og_buf_id = buffer_manager.get_backward_buffer_id(bwd_flat_index)
                    ia_buf_id = buffer_manager.get_forward_buffer_id(fwd_eq_prev_bwd_flat_index)
                    oa_buf_id = buffer_manager.get_forward_buffer_id(fwd_eq_bwd_flat_index)
                    oa_buf_id_forsend = buffer_manager.get_forward_buffer_id(prev_fwd_flat_index)
                    
                    # Backward comms (Receive Gradient)
                    if check_bounds(bwd_flat_index):
                        if (not self.pmap.is_last_stage_rank) or (self.pmap.is_last_stage_rank and (bwd_vp != (VPP - 1))):
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : B4-RG{bwd_mb}V{bwd_vp} to OGB{og_buf_id}, {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)
                            torch.distributed.recv(self.output_grad_buffers[og_buf_id], self.pmap.next_stage_rank)
                            self.recv_router_logits_grad_wrapper(bwd_mb, bwd_vp)
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : A4-RG{bwd_mb}V{bwd_vp} to OGB{og_buf_id}, {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)

                    # Backward comms (Send Gradient Previous)
                    if check_bounds(prev_bwd_flat_index):
                        if (self.pmap.is_first_stage_rank and (prev_bwd_vp != 0)): 
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : B4-SG{prev_bwd_mb}V{prev_bwd_vp} 4m IAB{ia_buf_id}.grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
                            torch.distributed.send(self.input_act_buffers[ia_buf_id].grad, self.pmap.prev_stage_rank)
                            self.send_router_logits_grad_wrapper(prev_bwd_mb, prev_bwd_vp)
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : A4-SG{prev_bwd_mb}V{prev_bwd_vp} 4m IAB{ia_buf_id}.grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)

                    # Backward compute
                    if check_bounds(bwd_flat_index):
                        if self.pmap.is_last_stage_rank and (bwd_vp == VPP-1):
                            self.output_act_buffers[oa_buf_id].backward(output_grad_list[bwd_mb])
                            if self.use_global_router_aux_loss:
                                self.prev_router_logits_grad_buffers[bwd_vp][bwd_mb].copy_(self.prev_router_logits_buffers[bwd_vp][bwd_mb].grad)
                                self.prev_router_logits_buffers[bwd_vp][bwd_mb].grad.zero_() # Clearing gradient
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : CMP-B{bwd_mb}V{bwd_vp} on OAB{oa_buf_id} with OSL{bwd_mb}", debug)
                        else:
                            if (not self.use_local_router_aux_loss) and (not self.use_global_router_aux_loss) :
                                self.output_act_buffers[oa_buf_id].backward(self.output_grad_buffers[og_buf_id])
                            else:
                                self.get_ral_accounted_scalar_loss_for_backward(oa_buf_id, og_buf_id, bwd_mb, bwd_vp).backward()
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : CMP-B{bwd_mb}V{bwd_vp} on OAB{oa_buf_id} with OGB{og_buf_id}", debug)
                        torch.xpu.synchronize()
                    
                    # Forward comms (Send Activation)
                    if check_bounds(prev_fwd_flat_index):
                        if (not self.pmap.is_last_stage_rank):
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : B4-SA{prev_fwd_mb}V{prev_fwd_vp} 4m OAB{oa_buf_id_forsend}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)
                            torch.distributed.send(self.output_act_buffers[oa_buf_id_forsend], self.pmap.next_stage_rank)
                            self.send_router_logits_wrapper(oa_buf_id_forsend, prev_fwd_mb, prev_fwd_vp)
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : A4-SA{prev_fwd_mb}V{prev_fwd_vp} 4m OAB{oa_buf_id_forsend}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)

                    # Updating to next backward block
                    bwd_flat_index += 1
                else:
                    prev_fwd_flat_index = (fwd_flat_index - 1)
                    prev_bwd_flat_index = (bwd_flat_index - 1)
                    fwd_mg, fwd_vp, fwd_lmb, fwd_mb = get_coord_info_from_flat_index(fwd_flat_index, NG, VPP, NMBG)
                    prev_fwd_mg, prev_fwd_vp, prev_fwd_lmb, prev_fwd_mb = get_coord_info_from_flat_index(prev_fwd_flat_index, NG, VPP, NMBG)
                    prev_bwd_mg, prev_bwd_vp, prev_bwd_lmb, prev_bwd_mb = get_coord_info_from_flat_index(prev_bwd_flat_index, NG, VPP, NMBG, flip_d2=True)
                    
                    # Buffer management
                    fwd_eq_prev_bwd_flat_index = buffer_manager.get_forward_index_from_backward_index(prev_bwd_flat_index)
                    oa_buf_id_circ = buffer_manager.get_forward_buffer_id(prev_fwd_flat_index)
                    ia_buf_id_forsend = buffer_manager.get_forward_buffer_id(fwd_eq_prev_bwd_flat_index)
                    buffer_manager.drain_forward_from_flight(fwd_eq_prev_bwd_flat_index)
                    buffer_manager.add_forward_to_flight(fwd_flat_index)
                    ia_buf_id = buffer_manager.get_forward_buffer_id(fwd_flat_index)
                    oa_buf_id = buffer_manager.get_forward_buffer_id(fwd_flat_index)

                    # Clear grad buffer
                    if ia_buf_id is not None:
                        print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : Zeroing IAB{ia_buf_id}.grad", debug)
                        # self.input_act_buffers[ia_buf_id].grad = None
                        if self.input_act_buffers[ia_buf_id].grad != None:
                            self.input_act_buffers[ia_buf_id].grad.zero_()

                    # Forward comms (Receive Activation)
                    if check_bounds(fwd_flat_index):
                        if (not self.pmap.is_first_stage_rank) or (self.pmap.is_first_stage_rank and (fwd_vp != 0)):
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : B4-RA{fwd_mb}V{fwd_vp} to IAB{ia_buf_id}, {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)
                            torch.distributed.recv(self.input_act_buffers[ia_buf_id], self.pmap.prev_stage_rank)
                            self.recv_router_logits_wrapper(fwd_mb, fwd_vp)
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : A4-RA{fwd_mb}V{fwd_vp} to IAB{ia_buf_id}, {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)

                    # Forward comms (Send Activation Previous)
                    if check_bounds(prev_fwd_flat_index):
                        if (self.pmap.is_last_stage_rank and (prev_fwd_vp != (VPP-1))):
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : B4-SA{prev_fwd_mb}V{prev_fwd_vp} 4m OAB{oa_buf_id_circ}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)
                            torch.distributed.send(self.output_act_buffers[oa_buf_id_circ], self.pmap.next_stage_rank)
                            self.send_router_logits_wrapper(oa_buf_id_circ, prev_fwd_mb, prev_fwd_vp)
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : A4-SA{prev_fwd_mb}V{prev_fwd_vp} 4m OAB{oa_buf_id_circ}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)

                    # Forward compute
                    if check_bounds(fwd_flat_index):
                        if self.pmap.is_first_stage_rank and (fwd_vp == 0):
                            output_dict = self.model(input_list[fwd_mb], vp_ind=fwd_vp)
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : CMP-F{fwd_mb}V{fwd_vp} 4m ISL{fwd_mb} to OAB{oa_buf_id}", debug)
                        else:
                            output_dict = self.model(self.input_act_buffers[ia_buf_id], vp_ind=fwd_vp)
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : CMP-F{fwd_mb}V{fwd_vp} 4m IAB{ia_buf_id} to OAB{oa_buf_id}", debug)
                        self.output_act_buffers[oa_buf_id] = output_dict["output"]
                        self.output_router_logits_buffers[oa_buf_id] = output_dict["router_logits"]
                        torch.xpu.synchronize()

                        if self.pmap.is_last_stage_rank and (fwd_vp == (VPP-1)):
                            if output_logits:
                                logits_list[fwd_mb] = self.output_act_buffers[oa_buf_id].detach().clone()
                            if labels != None:
                                # Calculate loss if labels are provided
                                logits = self.output_act_buffers[oa_buf_id]
                                loss = self.loss_function(logits, labels_list[fwd_mb], self.vocab_size)
                                # Calculate auxiliary loss when applicable
                                aux_loss = self.calculate_router_aux_loss(oa_buf_id, fwd_mb, fwd_vp)
                                if aux_loss is not None:
                                    loss += (self.router_aux_loss_coef * aux_loss)

                                self.output_act_buffers[oa_buf_id] = loss
                                loss_list[fwd_mb] = self.output_act_buffers[oa_buf_id].detach().clone()
                                self.output_act_buffers[oa_buf_id].div_(NMB*self.opt_grad_acc_steps) # Scaling loss by GAS
                        
                    # Backwrad comms (Send Gradient)
                    if check_bounds(prev_bwd_flat_index):
                        if (not self.pmap.is_first_stage_rank):
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : B4-SG{prev_bwd_mb}V{prev_bwd_vp} 4m IAB{ia_buf_id_forsend}.grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
                            torch.distributed.send(self.input_act_buffers[ia_buf_id_forsend].grad, self.pmap.prev_stage_rank)
                            self.send_router_logits_grad_wrapper(prev_bwd_mb, prev_bwd_vp)
                            print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : A4-SG{prev_bwd_mb}V{prev_bwd_vp} 4m IAB{ia_buf_id_forsend}.grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)

                    # Updating to next forward block
                    fwd_flat_index += 1
            ################################################################################
            
            ############################# WINDDOWN #########################################
            num_winddown_steps = (NMBG * VPP)
            bwd_flat_index = (NG - 1) * (NMBG * VPP) + self.pmap.pp_ind
            for wt in range(num_winddown_steps):
                t = (num_warmup_steps + num_steady_steps + wt) # Global time step

                prev_bwd_flat_index = (bwd_flat_index - 1)
                bwd_mg, bwd_vp, bwd_lmb, bwd_mb = get_coord_info_from_flat_index(bwd_flat_index, NG, VPP, NMBG, flip_d2=True)
                prev_bwd_mg, prev_bwd_vp, prev_bwd_lmb, prev_bwd_mb = get_coord_info_from_flat_index(prev_bwd_flat_index, NG, VPP, NMBG, flip_d2=True)

                # Buffer management
                fwd_eq_bwd_flat_index = buffer_manager.get_forward_index_from_backward_index(bwd_flat_index)
                fwd_eq_prev_bwd_flat_index = buffer_manager.get_forward_index_from_backward_index(prev_bwd_flat_index)
                # Get buffer ids
                og_buf_id = buffer_manager.get_backward_buffer_id(bwd_flat_index)
                ia_buf_id_circ = buffer_manager.get_forward_buffer_id(fwd_eq_prev_bwd_flat_index)
                oa_buf_id = buffer_manager.get_forward_buffer_id(fwd_eq_bwd_flat_index)
                ia_buf_id = buffer_manager.get_forward_buffer_id(fwd_eq_bwd_flat_index)

                # Backward Comms (Receive Gradients)
                if check_bounds(bwd_flat_index):
                    if (not self.pmap.is_last_stage_rank) or (self.pmap.is_last_stage_rank and (bwd_vp != (VPP - 1))):
                        print_debug(f"Line : {get_linenumber()}, T{t} WD Rank {self.pmap.rank} : B4-RG{bwd_mb}V{bwd_vp} to OGB{og_buf_id}, {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)
                        torch.distributed.recv(self.output_grad_buffers[og_buf_id], self.pmap.next_stage_rank)
                        self.recv_router_logits_grad_wrapper(bwd_mb, bwd_vp)
                        print_debug(f"Line : {get_linenumber()}, T{t} WD Rank {self.pmap.rank} : A4-RG{bwd_mb}V{bwd_vp} to OGB{og_buf_id}, {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)
                
                # Backward Comms (Send Gradients Previous)
                if check_bounds(prev_bwd_flat_index) and (wt != 0):
                    if (self.pmap.is_first_stage_rank and (prev_bwd_vp != 0)):
                        print_debug(f"Line : {get_linenumber()}, T{t} WD Rank {self.pmap.rank} : B4-SG{prev_bwd_mb}V{prev_bwd_vp} 4m IAB{ia_buf_id_circ}.grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
                        torch.distributed.send(self.input_act_buffers[ia_buf_id_circ].grad, self.pmap.prev_stage_rank)
                        self.send_router_logits_grad_wrapper(prev_bwd_mb, prev_bwd_vp)
                        print_debug(f"Line : {get_linenumber()}, T{t} WD Rank {self.pmap.rank} : A4-SG{prev_bwd_mb}V{prev_bwd_vp} 4m IAB{ia_buf_id_circ}.grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)

                # Backward compute
                if check_bounds(bwd_flat_index):
                    if self.pmap.is_last_stage_rank and (bwd_vp == VPP-1):
                        self.output_act_buffers[oa_buf_id].backward(output_grad_list[bwd_mb])
                        if self.use_global_router_aux_loss:
                                self.prev_router_logits_grad_buffers[bwd_vp][bwd_mb].copy_(self.prev_router_logits_buffers[bwd_vp][bwd_mb].grad)
                                self.prev_router_logits_buffers[bwd_vp][bwd_mb].grad.zero_() # Clearing gradient
                        print_debug(f"Line : {get_linenumber()}, T{t} WD Rank {self.pmap.rank} : CMP-B{bwd_mb}V{bwd_vp} on OAB{oa_buf_id} with OSL{bwd_mb}", debug)
                    else:
                        if (not self.use_local_router_aux_loss) and (not self.use_global_router_aux_loss) :
                            self.output_act_buffers[oa_buf_id].backward(self.output_grad_buffers[og_buf_id])
                        else:
                            self.get_ral_accounted_scalar_loss_for_backward(oa_buf_id, og_buf_id, bwd_mb, bwd_vp).backward()
                        print_debug(f"Line : {get_linenumber()}, T{t} WD Rank {self.pmap.rank} : CMP-B{bwd_mb}V{bwd_vp} on OAB{oa_buf_id} with OGB{og_buf_id}", debug)
                    torch.xpu.synchronize()
                    
                # Backward Comms (Send Gradient)
                if check_bounds(bwd_flat_index):
                    if (not self.pmap.is_first_stage_rank):
                        print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : B4-SG{bwd_mb}V{bwd_vp} 4m IAB{ia_buf_id}.grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
                        torch.distributed.send(self.input_act_buffers[ia_buf_id].grad, self.pmap.prev_stage_rank)
                        self.send_router_logits_grad_wrapper(bwd_mb, bwd_vp)
                        print_debug(f"Line : {get_linenumber()}, T{t} SS Rank {self.pmap.rank} : A4-SG{bwd_mb}V{bwd_vp} 4m IAB{ia_buf_id}.grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)

                bwd_flat_index += 1
            ################################################################################
        else:
            raise NotImplementedError("Only gpipe pp_scheme is implemented in this version.")
        
        # Global Synchronization
        torch.xpu.synchronize()
        print_debug(f"Line : {get_linenumber()}, FL Rank {self.pmap.rank} : B4-SYNC (Timestamp : {time.ctime(time.time())})", debug)
        torch.distributed.barrier()
        print_debug(f"Line : {get_linenumber()}, FL Rank {self.pmap.rank} : A4-SYNC (Timestamp : {time.ctime(time.time())})", debug)

        # Gathering data for sending
        loss, logits = None, None
        if self.pmap.is_last_stage_rank:
            if output_logits:
                logits = torch.cat(logits_list)
            if labels != None:
                loss = torch.mean(torch.stack(loss_list))

        # Clear activation gradients
        if (not only_forward):
            print_rank_0("Setting activation gradients to None", skip=not self.verbose)
            if self.pp_scheme == "gpipe":
                for vp_ind in range(VPP):
                    for mb in range(NMB):
                        self.input_act_buffers[vp_ind][mb].grad = None
            else:
                for _ in self.input_act_buffers:
                    _.grad = None
        
        print_debug(f"Line : {get_linenumber()}, Rank {self.pmap.rank} : RETURN", debug)

        # First entry is always the final output of the forward pass.
        return ParallelTrainerOutput(loss=loss, logits=logits)
        
if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(0)

    # Initialize distributed training
    import os
    rank, world_size, local_rank, local_world_size = 0,1,0,1
    if int(os.getenv("PMI_SIZE", "1")) > 1:
        from optimus.dutils import setup_xpu_distributed
        rank, world_size, local_rank, local_world_size = setup_xpu_distributed("xccl")

    torch.xpu.synchronize()
    torch.distributed.barrier()

    from optimus.mapper import ParallelDpPpEpTpMapper
    pmap = ParallelDpPpEpTpMapper(
            pipeline_parallelism=world_size,
            rank=rank, 
            create_groups=True if world_size > 1 else False)

    # Run time configs
    dtype = torch.bfloat16
    # dtype = torch.float32
    device = "xpu"

    # Model and input config
    num_hidden_layers = 16
    hidden_size = 256
    batch_size = 8
    micro_batch_size = 1
    context_size = 4096
    pipeline_parallelism = world_size
    pp_scheme = "1f1b"
    virtual_pipeline_parallelism = 2
    reuse_buffers_in_1f1b = True
    
    output_logits = True
    only_forward = False
    debug = False
    fill_structured_input = False

    # Configs
    config_ref = Config(num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
    config = Config(num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, pipeline_parallelism=pipeline_parallelism, virtual_pipeline_parallelism=virtual_pipeline_parallelism)

    if rank == 0:
        print_str = ""
        print_str += f"Number of layers        : {num_hidden_layers}\n"
        print_str += f"Hidden size             : {hidden_size}\n"
        print_str += f"Batch size              : {batch_size}\n"
        print_str += f"Micro batch size        : {micro_batch_size}\n"
        print_str += f"Context size            : {context_size}\n"
        print_str += f"Pipeline parallelism    : {pipeline_parallelism}\n"
        print_str += f"Virtual pipeline stages : {virtual_pipeline_parallelism}\n"
        print_str += f"Reuse buffers in 1f1b ? : {reuse_buffers_in_1f1b}\n"
        print(print_str, flush=True)

    # Input
    input = torch.randn((batch_size, context_size, hidden_size), dtype=dtype, device=device, requires_grad=True)
    output_grad = torch.randn((batch_size, context_size, hidden_size), dtype=dtype, device=device)
    input_ref = torch.zeros((batch_size, context_size, hidden_size), dtype=dtype, device=device, requires_grad=True)
    with torch.no_grad():
        if fill_structured_input:
            for b in range(batch_size):
                input[b].mul_(0).add_(b+1)
        input_ref.data.copy_(input.data)

    # Models
    model_ref = Model(config_ref, scale_output=fill_structured_input)
    model = Model(config, pmap=pmap, scale_output=fill_structured_input)
    if fill_structured_input:
        with torch.no_grad():
            for l,layer in enumerate(model_ref.layers):
                # layer.weight.data.mul_(0).add_(l+1)
                layer.weight.data.mul_(0).add_(1)
    model.set_parameters_from_full_module(model_ref)

    # Setting dtype and device
    model_ref = model_ref.to(dtype).to(device)
    model = model.to(dtype).to(device)

    # Wrap model into Trainer to orchestrate pipelinel
    trainer = ParallelTrainer(
                config, model, pmap,
                batch_size, micro_batch_size, context_size,
                dtype, device, 
                pp_scheme=pp_scheme, reuse_buffers_in_1f1b=reuse_buffers_in_1f1b)

    output = trainer.step(input, output_grad, only_forward=only_forward, output_logits=output_logits, debug=debug) 

    # Reference
    from optimus.utils import print_error_info
    output_ref = model_ref(input_ref)
    if not only_forward:
        output_ref.backward(output_grad)
    
    if pmap.is_last_stage_rank:
        print_error_info(output, output_ref, "Output")
        
    if not only_forward:
        if pmap.is_first_stage_rank:
            print_error_info(input.grad, input_ref.grad, "Input gradient")
    
# bash launch_dist.sh 4 1 python virtual_pp_test.py
# bash launch_dist.sh 1 4 python virtual_pp_test.py