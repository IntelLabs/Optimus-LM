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

class ParallelTrainer:
    def __init__(self, 
        config, model, pmap, 
        batch_size, micro_batch_size, context_size, opt_grad_acc_steps,
        dtype, device, pp_scheme = "gpipe", reuse_buffers_in_1f1b = True,
        verbose=False, memory_tracker=None):

        self.dtype = dtype
        self.device = device

        self.model = model
        self.pmap = pmap

        self.pp_scheme = pp_scheme
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.context_size = context_size
        self.opt_grad_acc_steps = opt_grad_acc_steps
        assert batch_size % micro_batch_size == 0
        self.num_micro_batches = batch_size // micro_batch_size
        self.NMB = self.num_micro_batches # Shorthand

        # Loss function
        # self.loss_func = torch.nn.CrossEntropyLoss()
        self.vocab_size = config.vocab_size
        self.num_experts = getattr(config, 'num_experts', 1)
        self.num_experts_per_tok = getattr(config, 'num_experts_per_tok', 1)
        self.num_hidden_layers_per_rank = config.num_hidden_layers // pmap.pipeline_parallelism

        self.router_aux_loss_coef = getattr(config, 'router_aux_loss_coef', 0.0)
        self.use_aux_loss = getattr(config, 'output_router_logits', False)
        self.use_global_router_aux_loss = getattr(config, 'output_router_logits', False) and (not getattr(config, 'use_local_router_aux_loss', False))
        self.use_local_router_aux_loss = getattr(config, 'output_router_logits', False) and getattr(config, 'use_local_router_aux_loss', False)

        # Buffers
        self.reuse_buffers_in_1f1b = reuse_buffers_in_1f1b

        # Setting number of activation buffers
        self.num_act_buffers = self.num_micro_batches 
        if pp_scheme == "1f1b" and self.reuse_buffers_in_1f1b:
            self.num_act_buffers = pmap.PP + 1

        # Buffers
        act_buffer_shape = (micro_batch_size, context_size, config.hidden_size)
        self.input_act_buffers = [torch.zeros(act_buffer_shape, dtype=dtype, device=device, requires_grad=True) for i in range(self.num_act_buffers)]
        self.output_act_buffers = [None for i in range(self.num_act_buffers)]
        self.output_grad_buffers = [torch.zeros(act_buffer_shape, dtype=dtype, device=device)  for i in range(self.num_act_buffers)]

        self.output_router_logits_buffers = [None for i in range(self.num_act_buffers)]
        self.prev_router_logits_buffers = [None for i in range(self.num_act_buffers)]
        self.all_router_logits_grad_buffers = [None for i in range(self.num_act_buffers)]
        self.prev_router_logits_grad_buffers = [None for i in range(self.num_act_buffers)] # References to all_router_logits_grad_buffers except the last part

        if self.use_global_router_aux_loss:
            for i in range(self.num_act_buffers):
                if pmap.pp_ind > 0:
                    requires_grad = (pmap.pp_ind == (pmap.pipeline_parallelism-1))
                    prev_router_logits_shape = (pmap.pp_ind * (self.num_hidden_layers_per_rank * micro_batch_size * context_size), config.num_experts)
                    self.prev_router_logits_buffers[i] = torch.zeros(prev_router_logits_shape, dtype=dtype, device=device, requires_grad=requires_grad)

                all_router_logits_shape = ((pmap.pp_ind+1) * (self.num_hidden_layers_per_rank * micro_batch_size * context_size), config.num_experts)
                self.all_router_logits_grad_buffers[i] = torch.zeros(all_router_logits_shape, dtype=dtype, device=device)
                self.prev_router_logits_grad_buffers[i] = self.all_router_logits_grad_buffers[i][:-1*(self.num_hidden_layers_per_rank*micro_batch_size*context_size)]

        self.verbose = False
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

    def log_memory_info(self, tag):
        if self.memory_tracker is not None:
            self.memory_tracker.log(tag)

    def calculate_router_aux_loss(self, mb):
        aux_loss = None
        if self.use_local_router_aux_loss:
            aux_loss = layer_level_load_balancing_loss_func(
                self.output_router_logits_buffers[mb],
                self.num_experts, 
                self.num_experts_per_tok
            )
            # loss += (self.router_aux_loss_coef * aux_loss)
        
        if self.use_global_router_aux_loss:
            prev_router_logits_list = [self.prev_router_logits_buffers[mb]] if (self.prev_router_logits_buffers[mb] != None) else []
            cur_router_logits_list = [layer_gate for layer_gate in self.output_router_logits_buffers[mb]]
            all_router_logits_list = prev_router_logits_list + cur_router_logits_list
            all_router_logits = torch.cat(all_router_logits_list, dim=0)
            aux_loss = single_input_load_balancing_loss_func(all_router_logits, self.num_experts, self.num_experts_per_tok)
            # loss += (self.router_aux_loss_coef * aux_loss)

        return aux_loss

    def send_act_wrapper(self, id, tag=None):
        assert tag is not None
        torch.distributed.send(self.output_act_buffers[id], self.pmap.next_stage_rank, tag=tag)

    def recv_act_wrapper(self, id, tag=None):
        assert tag is not None
        torch.distributed.recv(self.input_act_buffers[id], self.pmap.prev_stage_rank, tag=tag)
    
    def send_grad_wrapper(self, id, tag=None):
        assert tag is not None
        torch.distributed.send(self.input_act_buffers[id].grad, self.pmap.prev_stage_rank, tag=tag)

    def recv_grad_wrapper(self, id, tag=None):
        assert tag is not None
        torch.distributed.recv(self.output_grad_buffers[id], self.pmap.next_stage_rank, tag=tag)

    def send_router_logits_wrapper(self, id, tag=None):
        assert tag is not None
        if self.use_global_router_aux_loss:
            prev_router_logits_list = [self.prev_router_logits_buffers[id]] if (self.prev_router_logits_buffers[id] != None) else []
            cur_router_logits_list = [layer_gate for layer_gate in self.output_router_logits_buffers[id]]
            all_router_logits_list = prev_router_logits_list + cur_router_logits_list
            all_router_logits = torch.cat(all_router_logits_list, dim=0)
            torch.distributed.send(all_router_logits, self.pmap.next_stage_rank, tag=tag)

    def recv_router_logits_wrapper(self, id, tag=None):
        assert tag is not None
        if self.use_global_router_aux_loss:
            torch.distributed.recv(self.prev_router_logits_buffers[id], self.pmap.prev_stage_rank, tag=tag)

    def send_router_logits_grad_wrapper(self, id, tag=None):
        assert tag is not None
        if self.use_global_router_aux_loss:
            torch.distributed.send(self.prev_router_logits_grad_buffers[id], self.pmap.prev_stage_rank, tag=tag)

    def recv_router_logits_grad_wrapper(self, id, tag=None):
        assert tag is not None
        if self.use_global_router_aux_loss:
            torch.distributed.recv(self.all_router_logits_grad_buffers[id], self.pmap.next_stage_rank, tag=tag)

    def get_ral_accounted_scalar_loss_for_backward(self, id):
        assert self.use_local_router_aux_loss or self.use_global_router_aux_loss
        
        if self.use_local_router_aux_loss:
            loss = torch.sum(self.output_act_buffers[id] * self.output_grad_buffers[id]) # Dummy loss to avoid two backwards
            aux_loss = layer_level_load_balancing_loss_func(
                self.output_router_logits_buffers[id],
                self.num_experts, 
                self.num_experts_per_tok
            )
            loss += (self.router_aux_loss_coef * aux_loss)
            # loss.backward()
        
        if self.use_global_router_aux_loss:
            loss = torch.sum(self.output_act_buffers[id] * self.output_grad_buffers[id]) # Dummy loss to avoid two backwards

            aux_loss = None
            for layer_id,layer_gate in enumerate(self.output_router_logits_buffers[id]):
                base_ind  = self.pmap.pp_ind * (self.num_hidden_layers_per_rank * self.micro_batch_size * self.context_size)
                start_ind = base_ind + (layer_id) * (self.micro_batch_size * self.context_size)
                end_ind   = base_ind + (layer_id+1) * (self.micro_batch_size * self.context_size)
                layer_gate_grad_buffer = self.all_router_logits_grad_buffers[id][start_ind:end_ind]
                if aux_loss is None:
                    aux_loss = torch.sum(layer_gate * layer_gate_grad_buffer)
                else:
                    aux_loss += torch.sum(layer_gate * layer_gate_grad_buffer)
            
            loss += aux_loss
            # loss.backward()
        return loss

    def step(self, input=None, labels=None, only_forward=False, output_grad=None, output_logits=False, debug=False):
        # Shorthands
        NMB = self.num_micro_batches
        MBS = self.micro_batch_size

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

        ######################################################################################
        print_rank_0(f"Doing forward {'' if only_forward else 'and backward'} computation", skip=not self.verbose)
        if self.pp_scheme == "gpipe":
            # Forward passes.
            for mb in range(NMB):
                # Receiving activation from previous stage
                if not self.pmap.is_first_stage_rank:
                    self.recv_act_wrapper(mb, tag=mb)
                    self.recv_router_logits_wrapper(mb, tag=(2*NMB+mb))
                    print_debug(f"Rank {self.pmap.rank:2d} : RA{mb} {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)

                # Compute
                #output_mb = self.model(self.input_act_buffers[mb])
                self.log_memory_info(f"Before F{mb}")
                if self.pmap.is_first_stage_rank:
                    output_dict = self.model(input_list[mb])
                else:
                    output_dict = self.model(self.input_act_buffers[mb])
                self.output_act_buffers[mb] = output_dict["output"]
                self.output_router_logits_buffers[mb] = output_dict["router_logits"]
                self.log_memory_info(f"After F{mb}")

                if self.pmap.is_last_stage_rank:
                    self.log_memory_info(f"Before FL{mb}")
                    if output_logits:
                        logits_list[mb] = self.output_act_buffers[mb].detach().clone()
                    if labels != None:
                        # Calculate loss if labels are provided
                        logits = self.output_act_buffers[mb]
                        loss = self.loss_function(logits, labels_list[mb], self.vocab_size)
                        # Calculate auxiliary loss when applicable
                        aux_loss = self.calculate_router_aux_loss(mb)
                        if aux_loss is not None:
                            loss += (self.router_aux_loss_coef * aux_loss)

                        self.output_act_buffers[mb] = loss
                        loss_list[mb] = self.output_act_buffers[mb].detach().clone()
                        self.output_act_buffers[mb].div_(NMB*self.opt_grad_acc_steps) # Scaling loss by GAS
                    
                    self.log_memory_info(f"After FL{mb}")
                # Ensuring forward kernels are completed
                torch.xpu.synchronize()
                print_debug(f"Rank {self.pmap.rank:2d} : F{mb}", debug)

                # Sending activation to the next stage
                if not self.pmap.is_last_stage_rank:
                    self.send_act_wrapper(mb, tag=mb)
                    self.send_router_logits_wrapper(mb, tag=mb+(2*NMB))
                    print_debug(f"Rank {self.pmap.rank:2d} : SA{mb} {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)
            
            # Backward passes
            if not only_forward:
                for mb in range(NMB):
                    # Receiving gradient from next stage
                    if not self.pmap.is_last_stage_rank:
                        self.recv_grad_wrapper(mb, tag=(NMB+mb))
                        self.recv_router_logits_grad_wrapper(mb, tag=(3*NMB+mb))
                        print_debug(f"Rank {self.pmap.rank:2d} : RG{mb} {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)
                    
                    # Compute
                    self.log_memory_info(f"Before B{mb}")
                    if self.pmap.is_last_stage_rank:
                        self.output_act_buffers[mb].backward(output_grad_list[mb])
                        if self.use_global_router_aux_loss:
                            self.prev_router_logits_grad_buffers[mb].copy_(self.prev_router_logits_buffers[mb].grad)
                            self.prev_router_logits_buffers[mb].grad.zero_() # Clearing gradient
                    else:
                        if (not self.use_local_router_aux_loss) and (not self.use_global_router_aux_loss) :
                            self.output_act_buffers[mb].backward(self.output_grad_buffers[mb])
                        else:
                            self.get_ral_accounted_scalar_loss_for_backward(mb).backward()

                    self.log_memory_info(f"After B{mb}")
                    torch.xpu.synchronize()
                    print_debug(f"Rank {self.pmap.rank:2d} : B{mb}", debug)

                    # Sending gradient to previous stage
                    if not self.pmap.is_first_stage_rank:
                        self.send_grad_wrapper(mb, tag=(NMB+mb))
                        self.send_router_logits_grad_wrapper(mb, tag=(3*self.NMB+mb))
                        
                        print_debug(f"Rank {self.pmap.rank:2d} : SG{mb} {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
        elif self.pp_scheme == "1f1b":
            ################################# Warmup state ########################################
            for fwd_mb in range(min(self.pmap.pp_ind_reverse, NMB)):
                # Comms Forward
                if not self.pmap.is_first_stage_rank:
                    self.recv_act_wrapper(fwd_mb, tag=fwd_mb)
                    self.recv_router_logits_wrapper(fwd_mb, tag=(2*self.NMB+fwd_mb))
                    print_debug(f"Line : {get_linenumber()}, WU Rank {self.pmap.rank} : RA{fwd_mb} to buffer IAB{fwd_mb}, {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)

                # Forward compute
                if self.pmap.is_first_stage_rank:
                    output_dict = self.model(input_list[fwd_mb])
                    print_debug(f"Line : {get_linenumber()}, WU Rank {self.pmap.rank} : F{fwd_mb} IA{fwd_mb} -> OAB{fwd_mb}", debug)
                else:
                    output_dict = self.model(self.input_act_buffers[fwd_mb])
                    print_debug(f"Line : {get_linenumber()}, WU Rank {self.pmap.rank} : F{fwd_mb} IAB{fwd_mb} -> OAB{fwd_mb}", debug)
                self.output_act_buffers[fwd_mb] = output_dict["output"]
                self.output_router_logits_buffers[fwd_mb] = output_dict["router_logits"]
                self.log_memory_info(f"After F{fwd_mb}")
                
                torch.xpu.synchronize()

                # Comms Forward
                if not self.pmap.is_last_stage_rank:
                    self.send_act_wrapper(fwd_mb, tag=fwd_mb)
                    self.send_router_logits_wrapper(fwd_mb, tag=(2*self.NMB+fwd_mb))
                    print_debug(f"Line : {get_linenumber()}, WU Rank {self.pmap.rank} : SA{fwd_mb} 4m buffer OAB{fwd_mb}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)
            ########################################################################################

            ################################ Steady state ########################################
            for fwd_mb in range(self.pmap.pp_ind_reverse, NMB):
                bwd_mb = fwd_mb - self.pmap.pp_ind_reverse
                # Getting buffer id's
                fwd_buf_id = (fwd_mb % self.num_act_buffers) if self.reuse_buffers_in_1f1b else fwd_mb
                bwd_buf_id = (bwd_mb % self.num_act_buffers) if self.reuse_buffers_in_1f1b else bwd_mb

                # Comms 
                if not self.pmap.is_first_stage_rank:
                    self.recv_act_wrapper(fwd_buf_id, tag=fwd_mb)
                    self.recv_router_logits_wrapper(fwd_buf_id, tag=(2*NMB+fwd_mb))
                    print_debug(f"Line : {get_linenumber()}, SS Rank {self.pmap.rank} : RA{fwd_mb} to buffer IAB{fwd_buf_id}, {self.pmap.prev_stage_rank} -> {self.pmap.rank}", debug)
                
                if not only_forward:
                    # Comms
                    if not self.pmap.is_first_stage_rank:
                        bwd_prev_mb = bwd_mb - 1                    
                        bwd_prev_buf_id = (bwd_prev_mb % self.num_act_buffers) if self.reuse_buffers_in_1f1b else bwd_prev_mb

                        if bwd_prev_mb >= 0:
                            self.send_grad_wrapper(bwd_prev_buf_id, tag=(NMB+(bwd_prev_mb)))
                            self.send_router_logits_grad_wrapper(bwd_prev_buf_id, tag=(3*NMB+bwd_prev_mb))
                            if self.reuse_buffers_in_1f1b:
                                # Clearing previous input grad
                                self.input_act_buffers[(bwd_prev_buf_id-1)%self.num_act_buffers].grad = None #.mul_(0) # Clearing gradient
                            print_debug(f"Line : {get_linenumber()}, SS Rank {self.pmap.rank} : SG{bwd_prev_mb} 4m buffer IG{bwd_prev_buf_id} and cleared grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
                
                # Compute forward
                self.log_memory_info(f"Before F{fwd_mb}")
                if self.pmap.is_first_stage_rank:
                    output_dict = self.model(input_list[fwd_mb])
                    print_debug(f"Line : {get_linenumber()}, SS Rank {self.pmap.rank} : F{fwd_mb} IA{fwd_mb} -> OAB{fwd_buf_id}", debug)
                else:
                    output_dict = self.model(self.input_act_buffers[fwd_buf_id])
                    print_debug(f"Line : {get_linenumber()}, SS Rank {self.pmap.rank} : F{fwd_mb} IAB{fwd_buf_id} -> OAB{fwd_buf_id}", debug)
                self.output_act_buffers[fwd_buf_id] = output_dict["output"]
                self.output_router_logits_buffers[fwd_buf_id] = output_dict["router_logits"]
                self.log_memory_info(f"After F{fwd_mb}")

                if self.pmap.is_last_stage_rank:
                    self.log_memory_info(f"Before FL{fwd_mb}")
                    if output_logits:
                        logits_list[fwd_mb] = self.output_act_buffers[fwd_buf_id].detach().clone()
                    if labels != None:
                        # Calculate loss if labels are provided
                        logits = self.output_act_buffers[fwd_buf_id]
                        loss = self.loss_function(logits, labels_list[fwd_mb], self.vocab_size)
                        # Calculate auxiliary loss when applicable
                        aux_loss = self.calculate_router_aux_loss(fwd_buf_id)
                        if aux_loss is not None:
                            loss += (self.router_aux_loss_coef * aux_loss)

                        self.output_act_buffers[fwd_buf_id] = loss
                        loss_list[fwd_mb] = self.output_act_buffers[fwd_buf_id].detach().clone()
                        self.output_act_buffers[fwd_buf_id].div_(NMB*self.opt_grad_acc_steps) # Scaling loss by GAS
                    
                    self.log_memory_info(f"After FL{fwd_mb}")
                # Ensuring forward kernels are completed
                torch.xpu.synchronize()

                # Comms
                if not self.pmap.is_last_stage_rank:
                    self.send_act_wrapper(fwd_buf_id, tag=fwd_mb)
                    self.send_router_logits_wrapper(fwd_buf_id, tag=fwd_mb)
                    print_debug(f"Line : {get_linenumber()}, SS Rank {self.pmap.rank} : SA{fwd_mb} 4m buffer OAB{fwd_buf_id}, {self.pmap.rank} -> {self.pmap.next_stage_rank}", debug)

                if not only_forward:
                    # Comms
                    if not self.pmap.is_last_stage_rank:
                        self.recv_grad_wrapper(bwd_buf_id, tag=(NMB+bwd_mb))
                        self.recv_router_logits_grad_wrapper(bwd_buf_id, tag=(3*NMB+bwd_mb))
                        print_debug(f"Line : {get_linenumber()}, SS Rank {self.pmap.rank} : RG{bwd_mb} to buffer OGB{bwd_buf_id}, {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)

                    # Compute
                    self.log_memory_info(f"Before B{bwd_mb}")
                    if self.pmap.is_last_stage_rank:
                        self.output_act_buffers[bwd_buf_id].backward(output_grad_list[bwd_mb])
                        if self.use_global_router_aux_loss:
                            self.prev_router_logits_grad_buffers[bwd_buf_id].copy_(self.prev_router_logits_buffers[bwd_buf_id].grad)
                            self.prev_router_logits_buffers[bwd_buf_id].grad.zero_() # Clearing gradient
                        print_debug(f"Line : {get_linenumber()}, SS Rank {self.pmap.rank} : B{bwd_mb} OAB{bwd_buf_id}.backward(OG{bwd_mb})", debug)
                    else:
                        if (not self.use_local_router_aux_loss) and (not self.use_global_router_aux_loss) :
                            self.output_act_buffers[bwd_buf_id].backward(self.output_grad_buffers[bwd_buf_id])
                        else:
                            self.get_ral_accounted_scalar_loss_for_backward(bwd_buf_id).backward()
                        print_debug(f"Line : {get_linenumber()}, SS Rank {self.pmap.rank} : B{bwd_mb} OAB{bwd_buf_id}.backward(OGB{bwd_buf_id})", debug)
                    
                    self.log_memory_info(f"After B{bwd_mb}")
                    torch.xpu.synchronize()

                    # Comms (Taking care of final gradient send)
                    if fwd_mb == NMB - 1:
                        if not self.pmap.is_first_stage_rank:
                            self.send_grad_wrapper(bwd_buf_id, tag=(NMB+bwd_mb))
                            self.send_router_logits_grad_wrapper(bwd_buf_id, tag=(3*self.NMB+bwd_mb))
                            if self.reuse_buffers_in_1f1b:
                                self.input_act_buffers[(bwd_buf_id-1)%self.num_act_buffers].grad = None #.mul_(0) # Clearing gradient
                            print_debug(f"Line : {get_linenumber()}, SS Rank {self.pmap.rank} : SG{bwd_mb} 4m buffer IAB{bwd_buf_id} and cleared grad, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
                
            ########################################################################################
            
            ################################## Winddown state ######################################
            if not only_forward:
                for bwd_mb in range((NMB-self.pmap.pp_ind_reverse), NMB):
                    # Getting buffer ids
                    bwd_buf_id = (bwd_mb % self.num_act_buffers) if self.reuse_buffers_in_1f1b else bwd_mb

                    # Comms
                    if not self.pmap.is_last_stage_rank:
                        self.recv_grad_wrapper(bwd_buf_id, tag=(NMB+bwd_mb))
                        self.recv_router_logits_grad_wrapper(bwd_buf_id, tag=(3*NMB+bwd_mb))
                        print_debug(f"Line : {get_linenumber()}, WD Rank {self.pmap.rank} : RG{bwd_mb} to buffer OGB{bwd_buf_id}, {self.pmap.next_stage_rank} -> {self.pmap.rank}", debug)
                    
                    # Compute
                    self.log_memory_info(f"Before B{bwd_mb}")
                    if self.pmap.is_last_stage_rank:
                        self.output_act_buffers[bwd_buf_id].backward(output_grad_list[bwd_mb])
                        if self.use_global_router_aux_loss:
                            self.prev_router_logits_grad_buffers[bwd_buf_id].copy_(self.prev_router_logits_buffers[bwd_buf_id].grad)
                            self.prev_router_logits_buffers[bwd_buf_id].grad.zero_() # Clearing gradient
                        print_debug(f"Line : {get_linenumber()}, WD Rank {self.pmap.rank} : B{bwd_mb} OAB{bwd_buf_id}.backward(OG{bwd_mb})", debug)
                    else:
                        if (not self.use_local_router_aux_loss) and (not self.use_global_router_aux_loss):
                            self.output_act_buffers[bwd_buf_id].backward(self.output_grad_buffers[bwd_buf_id])
                        else:
                            self.get_ral_accounted_scalar_loss_for_backward(bwd_buf_id).backward()
                        print_debug(f"Line : {get_linenumber()}, WD Rank {self.pmap.rank} : B{bwd_mb} OAB{bwd_buf_id}.backward(OGB{bwd_buf_id})", debug)
                    
                    self.log_memory_info(f"After B{bwd_mb}")
                    torch.xpu.synchronize()

                    # Comms
                    if not self.pmap.is_first_stage_rank:
                        self.send_grad_wrapper(bwd_buf_id, tag=(NMB+bwd_mb))
                        self.send_router_logits_grad_wrapper(bwd_buf_id, tag=(3*self.NMB+bwd_mb))
                        print_debug(f"Line : {get_linenumber()}, WD Rank {self.pmap.rank} : SG{bwd_mb} 4m buffer IAB{bwd_buf_id}, {self.pmap.rank} -> {self.pmap.prev_stage_rank}", debug)
            ######################################################################################## 
        else:
            print(f"Pipeline scheme {pp_scheme} not supported")
            exit(-1)        
        print_rank_0(f"Completed forward {'' if only_forward else 'and backward'} computation", skip=not self.verbose)
        #####################################################################################

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
            for _ in self.input_act_buffers:
                _.grad = None
        
        # First entry is always the final output of the forward pass.
        return ParallelTrainerOutput(loss=loss, logits=logits)
        
        """
        if labels != None:
            return (loss, logits)
        else:
            return (logits, )
        """