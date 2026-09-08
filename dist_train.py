import os
import json
import argparse
import math
import time
import sys

from itertools import islice

import logging
logging.disable(logging.INFO)

if int(os.getenv("PALS_RANKID", "0")) == 0:
    print(f"Before torch/ipex imports.  (Timestamp : {time.ctime(time.time())})", flush=True)
import torch
import intel_extension_for_pytorch
if int(os.getenv("PALS_RANKID", "0")) == 0:
    print(f"After torch/ipex imports.   (Timestamp : {time.ctime(time.time())})", flush=True)

if int(os.getenv("PALS_RANKID", "0")) == 0:
    print(f"Before transformer imports. (Timestamp : {time.ctime(time.time())})", flush=True)
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version
if int(os.getenv("PALS_RANKID", "0")) == 0:
    print(f"After transformer imports.  (Timestamp : {time.ctime(time.time())})", flush=True)

if int(os.getenv("PALS_RANKID", "0")) == 0:
    print(f"Before optimus imports.     (Timestamp : {time.ctime(time.time())})", flush=True)
from optimus.mapper import ParallelDpPpEpTpMapper
from optimus.optimizers import MixedPrecisionAdamW, ShardedMixedPrecisionAdamW, ParamGroupShardedMixedPrecisionAdamW, SubShardedMixedPrecisionAdamW, ParamGroupSubShardedMixedPrecisionAdamW
from optimus.profilers import PclProfiler
from optimus.datasets import PclDataPrallelDataset, PclShardedDataParallelDataset
from optimus.config_utils import get_modified_config
from optimus.profilers import record_pcl_function
from optimus.utils import get_total_norm
from optimus.utils import no_init_weights
from optimus.utils import get_device, device_synchronize
if int(os.getenv("PALS_RANKID", "0")) == 0:
    print(f"After optimus imports.      (Timestamp : {time.ctime(time.time())})", flush=True)

# frameworks module has torch2.5, which does not have get_total_norm method
if not hasattr(torch.nn.utils, "get_total_norm"):
    torch.nn.utils.get_total_norm = get_total_norm

def synchronize():
    device_synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def get_learning_rate(step_id, train_steps, args):
    opt_step_id = step_id // args.opt_grad_acc_steps
    if args.lr_train_steps is None:
        optimizer_steps = train_steps // args.opt_grad_acc_steps
    else:
        optimizer_steps = args.lr_train_steps // args.opt_grad_acc_steps

    if opt_step_id < args.warmup_steps:
        # Linear warmup
        lr = args.min_lr + (args.lr - args.min_lr) * (opt_step_id) / float(max(1,args.warmup_steps))
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            # No warmup, cosine decay
            decay_ratio = (opt_step_id - args.warmup_steps) / (optimizer_steps - args.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = args.min_lr + coeff * (args.lr - args.min_lr)
        elif args.lr_schedule == "linear":
            # No warmup, linear decay
            decay_ratio = (opt_step_id - args.warmup_steps) / (optimizer_steps - args.warmup_steps)
            coeff =  (1.0 - decay_ratio)
            lr = args.min_lr + coeff * (args.lr - args.min_lr)
        else:
            raise ValueError(f"Invalid learning rate schedule: {args.lr_schedule}. Choose 'constant', 'cosine' or 'linear'.")
    return lr

if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(42) 

    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_choice", type=str, default="allenai/OLMo-1B-hf", help="Hugging face model name")
    parser.add_argument("--dtype", type=str, default="bf16", help="Data type to use for training")
    parser.add_argument("--use_activation_checkpointing", action="store_true", help="Use activation checkpointing in the model")
    parser.add_argument("--activation_checkpointing_level", type=int, default=0, help="Activation checkpointing level in a decoder block (0->None, 1->MoE 2->Attention, 4->Norms)")
    parser.add_argument("--use_fast_moe", action="store_true", help="Use custom ops in the model")
    parser.add_argument("--use_merged_mlp_in_fast_moe", action="store_true", help="Use merged mlp in fast moe")
    parser.add_argument("--use_triton_path_for_gemm_in_fast_moe", action="store_true", help="Use triton MoE kernels for GEMM in the model")
    parser.add_argument("--use_triton_path_for_nongemm_in_fast_moe", action="store_true", help="Use triton MoE kernels for non-GEMM in the model")

    # MoE specific
    parser.add_argument("--force_uniform_routing", action="store_true", help="Force uniform routing for MoE layers")
    parser.add_argument("--use_local_router_aux_loss", action="store_true", help="Use layer level router aux loss")

    # Initialization
    parser.add_argument("--use_om_cached_model", action="store_true", help="Use cached model generated from optimus scripts")
    parser.add_argument("--init_model_dir", type=str, default=None, help="Directory to load the model weights for initialization. This is needed for QAT and SAT experiments to initialize from the dense model weights.")
    parser.add_argument("--use_broadcast_for_model_init", action="store_true", help="Use broadcast for distributed initialization instead of loading from disk")
    parser.add_argument("--use_allreduce_in_broadcast_model_init", action="store_true", help="Use allreduce for distributed initialization instead of loading from disk or broadcast")

    # Parallelism
    parser.add_argument("--dist_backend", type=str, default="xccl", help="Distributed backend to use (xccl/ccl/mpi)")
    parser.add_argument("--use_hf_model", action="store_true", help="Use Hugging Face model")
    parser.add_argument("--data_parallelism", type=int, default=None, help="Data parallelism size")
    parser.add_argument("--pipeline_parallelism", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--virtual_pipeline_parallelism", type=int, default=1, help="Virtual pipeline parallelism size")
    parser.add_argument("--expert_parallelism", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--tensor_parallelism", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--use_sequence_parallelism_in_tp", action="store_true", help="When TP is used, sequence parallel can be used to parallelize norm computation")
    parser.add_argument("--pp_scheme", type=str, default="1f1b", choices=["gpipe","1f1b"], help="Scheme used for pipelining the micro batches")
    parser.add_argument("--use_pp_first", action="store_true", help="Whether to use pipeline parallelism first in the mapping")
    parser.add_argument("--reuse_buffers_in_1f1b", action="store_true", help="Whether to reuse buffers in 1f1b scheme")

    # Data
    parser.add_argument("--dataset_dir", type=str, default="datasets/HuggingFaceFW_fineweb-edu/preprocessed/olmo/", help="Path to the training data file")
    parser.add_argument("--num_data_files", type=int, default=-1, help="Number of data files to process")
    parser.add_argument("--use_sharded_dataset", action="store_true", help="Use sharded dataset")

    # Input
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--micro_batch_size", type=int, default=None, help="Micro batch size to be used pipeline parallelism")
    parser.add_argument("--context_size", type=int, default=4096, help="Input sequence length for training")

    # Optimizer
    parser.add_argument("--warmup_steps", type=int, default=2500, help="Number of warmup steps")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=4e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for AdamW")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon value")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "linear", "constant"], help="Learning rate schedule (constant/cosine/linear)")
    parser.add_argument("--lr_train_steps", type=int, default=None, help="Total training steps for learning rate schedule. If not provided, it will be calculated based on the dataset size and batch size.")

    parser.add_argument("--opt_use_pt_optimizer", action="store_true", help="Use AdamW with FP32 master weights")
    parser.add_argument("--opt_use_sharded_optimizer", action="store_true", help="Use sharded optimizer")
    parser.add_argument("--opt_use_pgsharded_optimizer", action="store_true", help="Use parameter group sharded optimizer (for large models)")
    parser.add_argument("--opt_use_subsharded_optimizer", action="store_true", help="Use sub-sharded optimizer (for large models)")
    parser.add_argument("--opt_use_pgsubsharded_optimizer", action="store_true", help="Use parameter group sub-sharded optimizer (for large models)")

    parser.add_argument("--opt_use_fp32_for_grad_acc", action="store_true", help="Use FP32 for gradient accumulation")
    parser.add_argument("--opt_use_allreduce_for_grad_acc", action="store_true", help="Use allreduce for gradient accumulation")
    parser.add_argument("--opt_use_allreduce_for_param_gather", action="store_true", help="Use allreduce for parameter gather")
    parser.add_argument("--opt_use_chunked_allreduce", action="store_true", help="Do allreduce ops in optimizer in chunks")

    parser.add_argument("--opt_disable_delayed_grad_clipping", action="store_true", help="Disable delayed gradient clipping")
    parser.add_argument("--opt_use_one_step_grad_clipping", action="store_true", help="Use one step gradient clipping")
    parser.add_argument("--opt_grad_acc_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Profiling
    parser.add_argument("--logging_steps", type=int, default=1, help="Print loss after these many steps")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--profile_modules", action="store_true", help="Enable module level profiling")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Checkpointing
    parser.add_argument("--exp_dir", type=str, default="experiment", help="Checkpoint directory")
    parser.add_argument("--enable_checkpointing", action="store_true", help="Enable checkpointing")
    parser.add_argument("--disable_optimizer_state_checkpointing", action="store_true", help="Disable optimizer state checkpointing to save memory")
    parser.add_argument("--disable_cold_checkpointing", action="store_true", help="Disable cold checkpointing and only save at the end of training")
    parser.add_argument("--dp_staggered_checkpointing", action="store_true", help="Stagger checkpointing across data parallel ranks to stagger the IO write across nodes")
    parser.add_argument("--checkpoint_steps", type=int, default=1000, help="Number of optimizer steps after which to save a checkpoint")

    # Caching
    parser.add_argument("--cached_models_dir", type=str, default="cached_models", help="Directory to store cached models")
    parser.add_argument("--om_cached_models_dir", type=str, default="parallel_models", help="Directory to store cached models generated from optimus scripts")

    # Features
    parser.add_argument("--disable_model_caching", action="store_true", help="Disable model caching to save memory")

    # Debug
    parser.add_argument("--log_memory_usage", action="store_true", help="Logs memory usage")
    parser.add_argument("--exit_steps", type=int, default=-1, help="Exit after these many steps")
    parser.add_argument("--enable_bad_grad_check", action="store_true", help="Check for NaN gradients")
    parser.add_argument("--skip_optimizer_step", action="store_true", help="Skip optimizer step for debugging purposes")
    parser.add_argument("--skip_optimizer_step_weight_update", action="store_true", help="Skip weight update in optimizer step for debugging purposes")
    parser.add_argument("--skip_backward_pass", action="store_true", help="Skip backward pass for debugging purposes")
    parser.add_argument("--dump_weight_gradients", action="store_true", help="Dumps the weight gradients")
    parser.add_argument("--log_all_ranks", action="store_true", help="Log all ranks' information")
    parser.add_argument("--enable_pp_debug", action="store_true", help="Enable pipeline parallelism debug")

    # Temporary
    parser.add_argument("--use_latest_trainer", action="store_true", help="Use latest trainer with vpp support")

    args = parser.parse_args()

    # Argument validation
    assert args.dtype in ["bf16", "fp32"], "Invalid data type"

    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size

    if (args.pipeline_parallelism > 1) or (args.expert_parallelism > 1) or (args.tensor_parallelism > 1):
        assert args.use_hf_model == False, "Huggingface model does not support parallelism"

    if int(os.getenv("PALS_RANKID", "0")) == 0:
        print(f"Setting up distributed training.      (Timestamp : {time.ctime(time.time())})", flush=True)

    # Initialize distributed training
    import os
    rank, world_size, local_rank, local_world_size = 0,1,0,1
    if int(os.getenv("PMI_SIZE", "1")) > 1:
        from optimus.dutils import setup_xpu_distributed
        rank, world_size, local_rank, local_world_size = setup_xpu_distributed(dist_backend=args.dist_backend)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if int(os.getenv("PALS_RANKID", "0")) == 0:
        print(f"Completed distributed training setup. (Timestamp : {time.ctime(time.time())})", flush=True)

    if args.data_parallelism == None:
        if rank == 0:
            print("Data parallelism size not provided. Calculating automatically.", flush=True)
        args.data_parallelism = world_size // (args.pipeline_parallelism * args.expert_parallelism * args.tensor_parallelism)
    assert world_size == (args.data_parallelism * args.pipeline_parallelism * args.expert_parallelism * args.tensor_parallelism), "WS = DP*PP*EP*TP"

    # Printing the arguments
    if rank == 0:
        print(args, flush=True)
        print(f"Model config choice             : {args.model_choice}", flush=True)
        print(f"Data type                       : {args.dtype}", flush=True)
        print(f"Distributed backend             : {args.dist_backend}", flush=True)
        print(f"Optimus model ?                 : {not args.use_hf_model}", flush=True)
        print(f"Init with om cached model ?     : {args.use_om_cached_model}", flush=True)
        print(f"Init with om trained model?     : {args.init_model_dir}", flush=True)
        print(f"Data parallelism size           : {args.data_parallelism}", flush=True)
        print(f"Pipeline parallelism size       : {args.pipeline_parallelism}", flush=True)
        print(f"Expert parallelism size         : {args.expert_parallelism}", flush=True)
        print(f"Tensor parallelism size         : {args.tensor_parallelism}", flush=True)
        print(f"Pipeline scheme                 : {args.pp_scheme}", flush=True)
        print(f"Use PP first in mapping         : {args.use_pp_first}", flush=True)
        if args.pp_scheme == "1f1b":
            print(f"Reuse buffers in 1f1b           : {args.reuse_buffers_in_1f1b}", flush=True)
        print("",flush=True)
        print(f"Preprocessed dataset path       : {args.dataset_dir}", flush=True)
        print(f"Using sharded dataset ?         : {args.use_sharded_dataset}", flush=True)
        print(f"Batch size                      : {args.batch_size}", flush=True)
        print(f"Context size                    : {args.context_size}", flush=True)
        print("",flush=True)
        print(f"Warmup steps                    : {args.warmup_steps}", flush=True)
        print(f"Learning rate                   : {args.lr}", flush=True)
        print(f"Minimum learning rate           : {args.min_lr}", flush=True)
        print(f"Learning rate schedule          : {args.lr_schedule}", flush=True)
        print(f"Gradient accumulation steps     : {args.opt_grad_acc_steps}", flush=True)
        print(f"Disable delayed grad clipping   : {args.opt_disable_delayed_grad_clipping}", flush=True)
        print(f"Use one step grad clipping      : {args.opt_use_one_step_grad_clipping}", flush=True)
        print("",flush=True)
        print(f"Activation checkpointing        : {args.use_activation_checkpointing}", flush=True)
        print(f"Activation checkpointing level  : {args.activation_checkpointing_level}", flush=True)
        print(f"Use fast moe path               : {args.use_fast_moe}", flush=True)
        print(f"Use merged mlp in fast moe      : {args.use_merged_mlp_in_fast_moe}", flush=True)
        print("",flush=True)
        print(f"Use pytorch optimizer ?         : {args.opt_use_pt_optimizer}", flush=True)
        print(f"Use optimus optimizer ?         : {not args.opt_use_pt_optimizer}", flush=True)
        print(f"Use optimus sharded opt ?       : {args.opt_use_sharded_optimizer}", flush=True)
        print(f"Use optimus pg sharded opt ?    : {args.opt_use_pgsharded_optimizer}", flush=True)
        print(f"Use optimus sub-sharded opt ?   : {args.opt_use_subsharded_optimizer}", flush=True)
        print(f"Use optimus pg sub-sharded opt? : {args.opt_use_pgsubsharded_optimizer}", flush=True)
        print(f"Use FP32 for grad acc           : {args.opt_use_fp32_for_grad_acc}", flush=True)
        print(f"Use allreduce for grad acc      : {args.opt_use_allreduce_for_grad_acc}", flush=True)
        print(f"Use allreduce for param gather  : {args.opt_use_allreduce_for_param_gather}", flush=True)
        print(f"Use chunked allreduce           : {args.opt_use_chunked_allreduce}", flush=True)
        print("",flush=True)
        print(f"Experiment directory            : {args.exp_dir}", flush=True)
        print(f"Enable checkpointing ?          : {args.enable_checkpointing}", flush=True)
        print(f"Disable optimizer state in ckpt : {args.disable_optimizer_state_checkpointing}", flush=True)
        print(f"Checkpoint steps                : {args.checkpoint_steps}", flush=True)
        checkpoint_tokens = args.checkpoint_steps * (world_size * args.opt_grad_acc_steps * args.batch_size * args.context_size)
        print(f"Checkpoint tokens               : {(checkpoint_tokens*1e-9):.2f} B", flush=True)

    pmap = ParallelDpPpEpTpMapper(
            data_parallelism=args.data_parallelism, 
            pipeline_parallelism=args.pipeline_parallelism,
            expert_parallelism=args.expert_parallelism,
            tensor_parallelism=args.tensor_parallelism, 
            rank=rank, 
            create_groups=True if world_size > 1 else False,
            use_pp_first=args.use_pp_first)

    if rank == 0:
    # if True:
        print(pmap, flush=True)

    # Input configuration
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = get_device()

    # Memory tracker
    from optimus.memory_profiler import MemoryTracker
    memory_tracker = None if not args.log_memory_usage else MemoryTracker(pmap)

    # Model
    if memory_tracker is not None:
        memory_tracker.log("Before model creation")
    if rank == 0:
        print(f"Creating model. (Timestamp : {time.ctime(time.time())})", flush=True)
    hf_model_choice = args.model_choice.split("--")[0]
    if args.use_hf_model:
        if transformers_version == "4.56.1":
            from optimus.utils import load_balancing_loss_func_4_56_1_mod
            from transformers.models.olmoe import modeling_olmoe 
            modeling_olmoe.load_balancing_loss_func = load_balancing_loss_func_4_56_1_mod

        config = AutoConfig.from_pretrained(hf_model_choice)
        config = get_modified_config(args.model_choice, config)
        if rank == 0:
            print(config, flush=True)
        model = AutoModelForCausalLM.from_config(config)
    else:
        assert hf_model_choice in ["allenai/OLMoE-1B-7B-0924", "allenai/OLMo-1B-hf", "allenai/OLMo-7B-hf", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "deepseek-ai/DeepSeek-V3"], f"{hf_model_choice} is not supported for parallel model training"
        ################################## Caching mechanism for faster run ####################################
        # Generated from hugging face model
        model_key = args.model_choice.replace('/','__')
        keys = ["NAL", "SKIPQKNORM"]
        for key in keys:
            if key in model_key:
                model_key = model_key.replace('--'+key,'')
                model_key = model_key.replace('-'+key,'')
        cached_model_key = f"{model_key}_pp{args.pipeline_parallelism}-ep{args.expert_parallelism}-tp{args.tensor_parallelism}-vpp{args.virtual_pipeline_parallelism}"
        if args.use_fast_moe and args.use_merged_mlp_in_fast_moe:
            cached_model_key += "_mmlp"
        cached_model_dir = os.path.join(args.cached_models_dir, cached_model_key)
        state_dict_path = os.path.join(cached_model_dir, f"model_{pmap.mp_ind}.pth")
        is_cached = os.path.exists(state_dict_path)     

        # Override to use OM cached model
        if args.use_om_cached_model:
            # Generated outside using optimus scripts with same initialization scheme as hugging face model
            om_cached_model_dir = os.path.join(args.om_cached_models_dir, cached_model_key)
            om_state_dict_path = os.path.join(om_cached_model_dir, f"model_{pmap.mp_ind}.pth")
            om_is_cached = os.path.exists(om_state_dict_path)
            assert om_is_cached == True, f"OM cached model not found at {om_state_dict_path}"
            state_dict_path = om_state_dict_path
            is_cached = om_is_cached

        if args.init_model_dir is not None:
            # Use this path for initializing the model weights
            init_state_dict_path = os.path.join(args.init_model_dir, "model_checkpoint.pth")
            if pmap.is_model_parallel:
                init_state_dict_path = os.path.join(args.init_model_dir, f"model_checkpoint_shard-{pmap.mp_ind}.pth")

            init_is_cached = os.path.exists(init_state_dict_path)
            assert init_is_cached == True, f"Initialization model not found at {init_state_dict_path}"
            state_dict_path = init_state_dict_path
            is_cached = init_is_cached

        # Synchronize before checking for cached model
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Hugging face config
        if (not is_cached) or args.disable_model_caching:
            # Hugging face model
            config_hf = AutoConfig.from_pretrained(hf_model_choice)
            config_hf = get_modified_config(args.model_choice, config_hf)
            model_hf = AutoModelForCausalLM.from_config(config_hf)
        #########################################################################################################
        if hf_model_choice == "allenai/OLMoE-1B-7B-0924":
            from optimus.models.olmoe.configuration_olmoe import OlmoeParallelConfig

            config = OlmoeParallelConfig(
                pipeline_parallelism=args.pipeline_parallelism,
                expert_parallelism=args.expert_parallelism,
                tensor_parallelism=args.tensor_parallelism,
                virtual_pipeline_parallelism=args.virtual_pipeline_parallelism,
                use_local_router_aux_loss=args.use_local_router_aux_loss,
                use_activation_checkpointing=args.use_activation_checkpointing,
                activation_checkpointing_level=args.activation_checkpointing_level,
                use_fast_moe=args.use_fast_moe,
                use_merged_mlp_in_fast_moe=args.use_merged_mlp_in_fast_moe,
                use_triton_path_for_gemm_in_fast_moe=args.use_triton_path_for_gemm_in_fast_moe,
                use_triton_path_for_nongemm_in_fast_moe=args.use_triton_path_for_nongemm_in_fast_moe,
                force_uniform_routing=args.force_uniform_routing
            )
            config = get_modified_config(args.model_choice, config)
            if rank == 0:
                print(config, flush=True)
            config.check_compatibility()
            if not args.use_latest_trainer:
                from optimus.models.olmoe.modeling_olmoe import OlmoeParallelForCausalLM
                with no_init_weights():
                    model = OlmoeParallelForCausalLM(config, pmap=pmap, memory_tracker=memory_tracker)
            else:
                from optimus.models.olmoe.modeling_olmoe_new import OlmoeParallelForCausalLM
                with no_init_weights():
                    model = OlmoeParallelForCausalLM(config, pmap=pmap, memory_tracker=memory_tracker)

        elif hf_model_choice in ["allenai/OLMo-1B-hf", "allenai/OLMo-7B-hf"]:
            from optimus.models.olmo.configuration_olmo  import OlmoParallelConfig
            from optimus.models.olmo.modeling_olmo import OlmoParallelForCausalLM
            
            config_hf_ref = AutoConfig.from_pretrained(hf_model_choice)
            config = OlmoParallelConfig(
                hidden_size=config_hf_ref.hidden_size,
                intermediate_size= config_hf_ref.intermediate_size,
                num_hidden_layers = config_hf_ref.num_hidden_layers,
                num_attention_heads = config_hf_ref.num_attention_heads,                
                # Parallel
                tensor_parallelism=args.tensor_parallelism,
                pipeline_parallelism=args.pipeline_parallelism,
                virtual_pipeline_parallelism=args.virtual_pipeline_parallelism)
            config = get_modified_config(args.model_choice, config)
            if rank == 0:
                print(config, flush=True)
            model = OlmoParallelForCausalLM(config, pmap=pmap)            
        elif hf_model_choice in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B"]:
            from optimus.models.llama.configuration_llama import LlamaParallelConfig
            from optimus.models.llama.modeling_llama import LlamaParallelForCausalLM

            config_hf_ref = AutoConfig.from_pretrained(hf_model_choice)
            config = LlamaParallelConfig(
                hidden_size=config_hf_ref.hidden_size,
                intermediate_size= config_hf_ref.intermediate_size,
                num_hidden_layers = config_hf_ref.num_hidden_layers,
                num_attention_heads = config_hf_ref.num_attention_heads,                
                num_key_value_heads = config_hf_ref.num_key_value_heads,
                # Parallel
                tensor_parallelism=args.tensor_parallelism,
                pipeline_parallelism=args.pipeline_parallelism,
                use_activation_checkpointing=args.use_activation_checkpointing,
                activation_checkpointing_level=args.activation_checkpointing_level
                )
            config = get_modified_config(args.model_choice, config)
            if rank == 0:
                print(config, flush=True)
            model = LlamaParallelForCausalLM(config, pmap=pmap)
        elif hf_model_choice == "deepseek-ai/DeepSeek-V3":
            from optimus.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3ParallelConfig
            from optimus.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ParallelForCausalLM

            config = DeepseekV3ParallelConfig(
                expert_parallelism=args.expert_parallelism,
                pipeline_parallelism=args.pipeline_parallelism,
                tensor_parallelism=args.tensor_parallelism,
                virtual_pipeline_parallelism=args.virtual_pipeline_parallelism,
                use_activation_checkpointing=args.use_activation_checkpointing,
                activation_checkpointing_level=args.activation_checkpointing_level,
                force_uniform_routing=args.force_uniform_routing
            ) # Default is DSV3
            config = get_modified_config(args.model_choice, config)
            if rank == 0:
                print(config, flush=True)
            # model = DeepseekV3ParallelForCausalLM(config)
            model = DeepseekV3ParallelForCausalLM(config, pmap=pmap, memory_tracker=memory_tracker)

        if (not is_cached) or args.disable_model_caching:
            with torch.no_grad():
                model.set_parameters_from_full_module(model_hf)
            # Cache the model
            if (pmap.dp_ind == 0) and (not args.disable_model_caching):
                print(f"Rank : {rank} Caching the model at {state_dict_path}", flush=True)
                os.makedirs(cached_model_dir, exist_ok=True)
                torch.save(model.state_dict(), state_dict_path)
        else:
            if pmap.dp_ind == 0:
                print(f"Loading cached model from {state_dict_path} (Timestamp : {time.ctime(time.time())})", flush=True)
            if not args.use_broadcast_for_model_init:
                model_state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True) # Load the model for debugging
                model_state_dict = model_state_dict["state_dict"] if "state_dict" in model_state_dict else model_state_dict
                model.load_state_dict(model_state_dict, strict=True)
                model = model.to(dtype).to(device)
            else:
                # Loading only on the first DP unit
                if pmap.dp_ind == 0:
                    model_state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
                    model_state_dict = model_state_dict["state_dict"] if "state_dict" in model_state_dict else model_state_dict
                    model.load_state_dict(model_state_dict, strict=True)
                else:                    
                    if args.use_allreduce_in_broadcast_model_init:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                param.data.fill_(0)
                
                model = model.to(dtype).to(device)

                if pmap.data_parallelism > 1:
                    # Global sync before broadcast
                    synchronize()
                    if rank == 0:
                        print(f"Started   broadcasting init model. (Timestamp : {time.ctime(time.time())})", flush=True)
                    # Broadcast the model to other ranks in the data parallel group
                    for name, param in model.named_parameters():
                        if not args.use_allreduce_in_broadcast_model_init:
                            torch.distributed.broadcast(param.data, src=pmap.mp_rank, group=pmap.dp_group)
                        else:
                            torch.distributed.all_reduce(param.data, group=pmap.dp_group)
                    # Global sync after broadcast
                    synchronize()
                    if rank == 0:
                        print(f"Completed broadcasting init model. (Timestamp : {time.ctime(time.time())})", flush=True)

    # Move model to device and dtype
    model = model.to(dtype).to(device)
    if memory_tracker is not None:
        memory_tracker.log("After model creation")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size = total_params / (1024 * 1024 * 1024) # Convert to GB
        model_size = (2*model_size) if args.dtype == "bf16" else (4*model_size)  # BF16 is 2 bytes, FP32 is 4 bytes
        print(f"Model parameters : {total_params}", flush=True)
        print(f"Model size : {model_size:.2f} GB", flush=True)
        print(f"Completed model creation. (Timestamp : {time.ctime(time.time())})", flush=True)

    # Profiling related
    profiler = None
    if args.profile:
        profiler = PclProfiler(rank=rank)
        profiler.register_timer("0train-step-block")
        import optimus.globals
        optimus.globals.profiler = profiler
    
    if args.profile_modules:
        assert args.profile == False, "Remove --profile flag when using --profile_modules"
        profiler = PclProfiler(rank=rank)
        profiler.register_timer("0train-step-block")
        from optimus.models.olmo.modeling_olmo import OlmoParallelAttention, OlmoParallelMLP
        from optimus.models.olmoe.modeling_olmoe import OlmoeParallelAttention, OlmoeParallelSparseMoeBlock, OlmoeParallelMLP
        include_filter_modules = (OlmoParallelAttention, OlmoParallelMLP)
        include_filter_modules += (OlmoeParallelAttention, OlmoeParallelSparseMoeBlock)
        profiler.setup_profiler_for_pytorch_model(model, include_filter_modules=include_filter_modules)

    if memory_tracker is not None:
        memory_tracker.log("Before optimizer creation")
    if rank == 0:
        print(f"Creating optimizer. (Timestamp : {time.ctime(time.time())})", flush=True)
    if not args.opt_use_pt_optimizer:
        kwargs = {
            "lr" : args.lr, 
            "betas" : (args.beta1, args.beta2),
            "eps" : args.eps,
            "weight_decay" : args.weight_decay, "warmup_steps" : args.warmup_steps,
            "dtype" : dtype, "device" : device,
            "num_shards" : pmap.data_parallelism, "shard_id" : pmap.dp_ind, "group" : pmap.dp_group,
            "opt_disable_delayed_grad_clipping" : args.opt_disable_delayed_grad_clipping,
            "profiler" : profiler,
            "tensor_parallelism" : pmap.tensor_parallelism, "tp_ind" : pmap.tp_ind, "tp_group" : pmap.tp_group,
            "expert_parallelism" : pmap.expert_parallelism, "ep_ind" : pmap.ep_ind, "ep_group" : pmap.ep_group, 
            "dpep_ind" : pmap.dpep_ind, "dpep_group" : pmap.dpep_group,
            "skip_optimizer_step" : args.skip_optimizer_step,
            "opt_use_chunked_allreduce" : args.opt_use_chunked_allreduce,
        }
        if not (args.opt_use_sharded_optimizer or args.opt_use_pgsharded_optimizer or args.opt_use_subsharded_optimizer or args.opt_use_pgsubsharded_optimizer):
            OptimizerClass = MixedPrecisionAdamW
            kwargs["opt_use_two_step_grad_clipping"] = (not args.opt_use_one_step_grad_clipping)
        else:
            if args.opt_use_sharded_optimizer:
                OptimizerClass = ShardedMixedPrecisionAdamW
                kwargs["disable_optimizer_state_checkpointing"] = args.disable_optimizer_state_checkpointing
                kwargs["skip_optimizer_step_weight_update"] = args.skip_optimizer_step_weight_update
            elif args.opt_use_pgsharded_optimizer:
                OptimizerClass = ParamGroupShardedMixedPrecisionAdamW
                kwargs["disable_optimizer_state_checkpointing"] = args.disable_optimizer_state_checkpointing
                kwargs["skip_optimizer_step_weight_update"] = args.skip_optimizer_step_weight_update
                # kwargs["verbose"] = args.verbose
            elif args.opt_use_subsharded_optimizer:
                OptimizerClass = SubShardedMixedPrecisionAdamW
            elif args.opt_use_pgsubsharded_optimizer:
                OptimizerClass = ParamGroupSubShardedMixedPrecisionAdamW
                kwargs["opt_use_fp32_for_grad_acc"] = args.opt_use_fp32_for_grad_acc
            kwargs["opt_use_allreduce_for_grad_acc"] = args.opt_use_allreduce_for_grad_acc
            kwargs["opt_use_allreduce_for_param_gather"] = args.opt_use_allreduce_for_param_gather
        optimizer = OptimizerClass(model.parameters(), model, **kwargs)
        # Using for debugging, so not passing to constructor
        optimizer.pmap = pmap
        optimizer.config = config
        optimizer.dump_weight_gradients = args.dump_weight_gradients
    else:
        # Setup distributed if needed
        if torch.distributed.is_initialized():
            if pmap.data_parallelism > 1:
                if rank == 0:
                    print("Setting up DDP model", flush=True)           
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = DDP(model, process_group=pmap.dp_group)
    
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), 
                            lr=args.lr, 
                            betas=(args.beta1, args.beta2),
                            eps=args.eps,
                            weight_decay=args.weight_decay)
    if memory_tracker is not None:
        memory_tracker.log("After optimizer creation")

    ######## Load from checkpoint ##################
    train_step_id = 0
    if args.enable_checkpointing:
        # Choose a valid checkpoint
        checkpoint = None

        # First checkpoint
        checkpoint1_valid = False
        checkpoint1_dir = os.path.join(args.exp_dir, "checkpoints", "checkpoint1")
        checkpoint1_success_marker_file_path = os.path.join(checkpoint1_dir, "completed")
        if os.path.isfile(checkpoint1_success_marker_file_path):
            checkpoint1_valid = True
        
        # Second checkpoint
        checkpoint2_valid = False
        checkpoint2_dir = os.path.join(args.exp_dir, "checkpoints", "checkpoint2")
        checkpoint2_success_marker_file_path = os.path.join(checkpoint2_dir, "completed")
        if os.path.isfile(checkpoint2_success_marker_file_path):
            checkpoint2_valid = True

        # Checkpoint choice logic
        if (checkpoint1_valid == True) and (checkpoint2_valid == True):
            # If both checkpoints are valid, choose the one with the higher step
            checkpoint1_step = int(open(checkpoint1_success_marker_file_path).readlines()[0])
            checkpoint2_step = int(open(checkpoint2_success_marker_file_path).readlines()[0])
            checkpoint = "checkpoint1" if (checkpoint1_step > checkpoint2_step) else "checkpoint2"
            if rank == 0:
                print(f"Both checkpoints are valid. Checkpoint 1 step is {checkpoint1_step}. Checkpoint 2 step is {checkpoint2_step}. Choosing latest checkpoint i.e, {checkpoint} for reading", flush=True)
        elif checkpoint1_valid == True:
            checkpoint = "checkpoint1"
            if rank == 0:
                print(f"Only checkpoint1 is valid. So choosing {checkpoint} for reading", flush=True)
        elif checkpoint2_valid == True:
            checkpoint = "checkpoint2"
            if rank == 0:
                print(f"Only checkpoint2 is valid. So choosing {checkpoint} for reading", flush=True)
        else:
            if rank == 0:
                print(f"Both checkpoints are invalid. Starting training from scratch", flush=True)
        
        if checkpoint is not None:
            checkpoint_dir = os.path.join(args.exp_dir, "checkpoints", checkpoint)
            model_checkpoint_file_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")
            if pmap.is_model_parallel:
                model_checkpoint_file_path = os.path.join(checkpoint_dir, f"model_checkpoint_shard-{pmap.mp_ind}.pth")

            is_optimizer_sharded = args.opt_use_sharded_optimizer or args.opt_use_pgsharded_optimizer or args.opt_use_subsharded_optimizer or args.opt_use_pgsubsharded_optimizer
            optimizer_checkpoint_file_path = os.path.join(checkpoint_dir, "optimizer_checkpoint.pth")
            if is_optimizer_sharded:
                optimizer_checkpoint_file_path = os.path.join(checkpoint_dir, f"optimizer_checkpoint_shard-{rank}.pth")
            
            if rank == 0:
                print(f"Loading model checkpoint from {checkpoint_dir}", flush=True)
            if not args.use_broadcast_for_model_init:
                model_checkpoint = torch.load(model_checkpoint_file_path, map_location="cpu", weights_only=True)
                model.load_state_dict(model_checkpoint['state_dict'])
                train_step_id = model_checkpoint['train_step_id'] # Override
            else:
                model_checkpoint_step_id = torch.zeros(1, dtype=torch.long, device=device)
                if pmap.dp_ind == 0:
                    model_checkpoint = torch.load(model_checkpoint_file_path, map_location="cpu", weights_only=True)
                    model_checkpoint_step_id[0] = model_checkpoint['train_step_id']
                    model.load_state_dict(model_checkpoint['state_dict'])
                else:                    
                    if args.use_allreduce_in_broadcast_model_init:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                param.data.fill_(0)
                        model_checkpoint_step_id.fill_(0)
                
                # Broadcast
                if pmap.data_parallelism > 0:
                    # Global sync before broadcast
                    synchronize()
                    if rank == 0:
                        print(f"Started   broadcasting model checkpoint. (Timestamp : {time.ctime(time.time())})", flush=True)
                    # Broadcast the model to other ranks in the data parallel group
                    if not args.use_allreduce_in_broadcast_model_init:
                        for name, param in model.named_parameters():
                            torch.distributed.broadcast(param.data, src=pmap.mp_rank, group=pmap.dp_group)
                        torch.distributed.broadcast(model_checkpoint_step_id, src=pmap.mp_rank, group=pmap.dp_group)
                    else:
                        for name, param in model.named_parameters():
                            torch.distributed.all_reduce(param.data, group=pmap.dp_group)
                        torch.distributed.all_reduce(model_checkpoint_step_id, group=pmap.dp_group)
                    # Global sync after broadcast
                    synchronize()
                    if rank == 0:
                        print(f"Completed broadcasting model checkpoint. (Timestamp : {time.ctime(time.time())})", flush=True)

                # Setting train step id (Override)
                train_step_id = model_checkpoint_step_id.item()

            if rank == 0:
                print(f"Loading optimizer checkpoint from {checkpoint_dir}", flush=True)
            optimizer_checkpoint = torch.load(optimizer_checkpoint_file_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_checkpoint['state_dict'])
    #################################################

    # Dataloader
    if rank == 0:
        print(f"Creating data loader. (Timestamp : {time.ctime(time.time())})", flush=True)
    st = time.time()
    if args.use_sharded_dataset:
        # train_dataset = PclShardedDataParallelDataset(args.dataset_dir, args.num_data_files, args.context_size, dp_size=pmap.data_parallelism, dp_ind=pmap.dp_ind)
        train_dataset = PclShardedDataParallelDataset(args.dataset_dir, args.num_data_files, args.context_size, dp_size=(pmap.data_parallelism * pmap.expert_parallelism), dp_ind=(pmap.dp_ind * pmap.expert_parallelism + pmap.ep_ind))
    else:
        train_dataset = PclDataPrallelDataset(args.dataset_dir, args.num_data_files, args.context_size, dp_size=(pmap.data_parallelism * pmap.expert_parallelism), dp_ind=(pmap.dp_ind * pmap.expert_parallelism + pmap.ep_ind))
        
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, prefetch_factor=4)
    train_dataset.set_base_dp_idx(train_step_id * args.batch_size) # Adjusting for checkpoint
    train_data_iterator = iter(train_dataloader)
    
    tr_st = time.time()
    train_steps = len(train_dataloader)
    if rank == 0:
        print(f"Training tokens   : {(train_steps * pmap.data_parallelism * pmap.expert_parallelism * args.batch_size * args.context_size)*1e-9:.2f} B", flush=True)
        print(f"Tokens processed  : {(train_step_id * pmap.data_parallelism * pmap.expert_parallelism * args.batch_size * args.context_size)*1e-9:.2f} B", flush=True)
        print(f"Tokens to process : {((train_steps - train_step_id) * pmap.data_parallelism * pmap.expert_parallelism * args.batch_size * args.context_size)*1e-9:.2f} B", flush=True)
        print(f"Completed data loader creation. (Timestamp : {time.ctime(time.time())})", flush=True)
    
    # Using trainer class for PP
    trainer = None
    if args.pipeline_parallelism > 1:
        if not args.use_latest_trainer:
            assert args.virtual_pipeline_parallelism == 1, "Old trainer does not support VPP"
            from optimus.training import ParallelTrainer
            trainer = ParallelTrainer(
                    config, model, pmap, 
                    args.batch_size, args.micro_batch_size, args.context_size, args.opt_grad_acc_steps,
                    dtype, device, args.pp_scheme, args.reuse_buffers_in_1f1b, memory_tracker=memory_tracker)
        else:
            from optimus.trainer import ParallelTrainer
            trainer = ParallelTrainer(
                    config, model, pmap, 
                    args.batch_size, args.micro_batch_size, args.context_size, args.opt_grad_acc_steps,
                    dtype, device, args.pp_scheme, args.reuse_buffers_in_1f1b, verbose=args.verbose)

    # Wait for all ranks before starting training
    synchronize()
    
    if rank == 0:
        print(f"Started training. (Timestamp : {time.ctime(time.time())})", flush=True)

    if memory_tracker is not None:
        memory_tracker.log("Before training")
    while train_step_id < train_steps:
        if (train_step_id == args.exit_steps):
            break

        if profiler is not None:
            profiler.start("0train-step-block")
        st = time.time()

        # Update the learning rate
        lr = get_learning_rate(train_step_id, train_steps, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Input data
        dl_st = time.time()
        with record_pcl_function("1train-step--0data-loader"):
            input_ids = next(train_data_iterator).to(device).long()
        dl_et = time.time()
 
        if trainer == None:
            if rank == 0 and args.verbose:
                print("Doing forward pass", flush=True)
            with record_pcl_function("1train-step--1forward"):
                output = model(input_ids, labels=input_ids)
            loss = output.loss
            scaled_loss = loss / args.opt_grad_acc_steps
            if rank == 0 and args.verbose:
                print("Doing backward pass", flush=True)
            with record_pcl_function("1train-step--2backward"):
                if not args.skip_backward_pass:
                    scaled_loss.backward()
                                    
        else:
            if rank == 0 and args.verbose:
                print("Doing trainer step", flush=True)
            output = trainer.step(input_ids, input_ids, debug=args.enable_pp_debug)
            loss = output.loss
            if rank == 0 and args.verbose:
                print("Completed trainer step", flush=True)
                
        # Calculate local gradient norm
        with record_pcl_function("1train-step--3local-gradnorm"):
            if rank == 0 and args.verbose:
                print("Calculating local grad norm", flush=True)
            local_grad_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
                
        # Calculate global loss
        with record_pcl_function("1train-step--4global-loss"):
            if rank == 0 and args.verbose:
                print("Calculating global loss", flush=True)
            """
            global_loss = None
            # Loss is available only on last stage ranks
            # TP ranks within a TP group have the same loss
            if pmap.is_last_stage_rank:
                global_loss = torch.tensor([loss.item()]).to(torch.float32).to(device)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(global_loss, group=pmap.dpep_group)
                    global_loss = global_loss / (pmap.data_parallelism * pmap.expert_parallelism)
            """
            global_loss = torch.tensor([0.0], dtype=torch.float32, device=device)
            if pmap.is_last_stage_rank:
                global_loss = torch.tensor([loss.item()], dtype=torch.float32, device=device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(global_loss)
                global_loss = global_loss / (pmap.data_parallelism * pmap.expert_parallelism * pmap.tensor_parallelism)

        # Soft node failure detection 
        is_soft_node_failure_detected = False
        with record_pcl_function("1train-step--5softnode-failure-check"):
            if rank == 0 and args.verbose:
                print("Checking for soft node failures", flush=True)

            soft_node_failure_status_list = torch.zeros(world_size, dtype=torch.float32, device=device)
            if ((loss is not None) and torch.isnan(loss)) or \
               ((loss is not None) and torch.isinf(loss)) or \
               (torch.isnan(local_grad_norm)) or (torch.isinf(local_grad_norm)):
                soft_node_failure_status_list[rank] = 1

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(soft_node_failure_status_list)
                torch.distributed.barrier()

            if (torch.sum(soft_node_failure_status_list) > 0):
                if rank == 0:
                    print(f"Softnode failure detected", flush=True)                    
                # Print information for ranks that have bad local loss or gradient
                if (soft_node_failure_status_list[rank] > 0):    
                    print(f"Node : {pmap.node}, Rank : {rank:4d}, dp_ind : {pmap.dp_ind:3d}, pp_ind : {pmap.pp_ind:3d}, ep_ind : {pmap.ep_ind:3d}, tp_ind : {pmap.tp_ind:2d}, Global loss : {global_loss.item():9.6f}, Local loss : {loss.item():9.6f}, Local Grad norm : {local_grad_norm.item():9.6f}", flush=True)
                is_soft_node_failure_detected = True

            # Only rank 0 writes the bad nodes file
            if is_soft_node_failure_detected:
                if (rank == 0):
                    job_id = os.getenv("PBS_JOBID", "0").split(".")[0]
                    launch_info_dir = os.path.join("launch_info", job_id)
                    soft_node_failure_file_path = os.path.join(launch_info_dir, "failed_soft_nodes.txt")

                    soft_nodes_failure_list = []
                    # Gather node list
                    for r in range(world_size):
                        if soft_node_failure_status_list[r] > 0:
                            node = pmap.rank_to_node_list[r]
                            if not node in soft_nodes_failure_list:
                                soft_nodes_failure_list.append(node)
                    with open(soft_node_failure_file_path, "a") as fh:
                        for n in soft_nodes_failure_list:
                            fh.write(f"{n}\n")

                # Synchronize to ensure bad nodes are written before exiting
                synchronize()

        # Exit on detection of soft node failure
        if is_soft_node_failure_detected:
            break

        ########### Debugging NaN gradients #################
        if args.enable_bad_grad_check:
            bad_grad_list = []
            for name,param in model.named_parameters():
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    bad_grad_list.append(name)
            
            if len(bad_grad_list) > 0:
                debug_dir = os.path.join(args.exp_dir, os.getenv("PBS_JOBID", "0"))
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"bad_grad_debug_rank{rank}-d{pmap.dp_ind}-e{pmap.ep_ind}-t{pmap.tp_ind}.txt")
                print(f"Dumping debug data at {debug_path}", flush=True)

                with open(debug_path, "w") as fh:
                    fh.write("Parameter, Gradient Norm\n")
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            param_grad_norm = torch.nn.utils.get_total_norm(p.grad)
                            fh.write(f"{n:45s}, {param_grad_norm.item():.6f}\n")
        #####################################################
        
        # Optimizer step
        grad_norm = local_grad_norm
        # Synchronization before doing optimizer step
        with record_pcl_function("1train-step--6jitter-before-opt"):
            if rank == 0 and args.verbose:
                print("Doing pre-optimizer step synchronization", flush=True)
            synchronize()

        if memory_tracker is not None:
            memory_tracker.log("Before optimizer step")    
        if ((train_step_id + 1) % args.opt_grad_acc_steps) == 0:
            if rank == 0 and args.verbose:
                print("Doing optimizer step", flush=True)
            if args.opt_use_pt_optimizer:
                with record_pcl_function("4opt-step_grad-clip"):
                    if args.opt_disable_delayed_grad_clipping or \
                        ((args.opt_disable_delayed_grad_clipping == False) and (train_step_id//args.opt_grad_acc_steps) >= args.warmup_steps):
                        if pmap.is_model_parallel and (rank == 0) and (train_step_id == 0):
                            print(f"Warning : Using local gradient clipping", flush=True)
                        if args.opt_use_one_step_grad_clipping:
                            # Single step gradient clipping
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        else:
                            # Two step gradient clipping
                            grad_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
                            torch.nn.utils.clip_grads_with_norm_(model.parameters(), 1.0, grad_norm)

                # Optimizer step
                with record_pcl_function("1train-step--7opt-step"):
                    if (not args.skip_optimizer_step):
                        optimizer.step()
            else:
                # Optimizer step
                with record_pcl_function("1train-step--7opt-step"):
                    if (not args.skip_optimizer_step):
                        grad_norm = optimizer.step()
            
            # Zero gradients
            with record_pcl_function("1train-step--8zero-grad"):
                optimizer.zero_grad()
        if memory_tracker is not None:
            memory_tracker.log("After optimizer step")
            
        # Synchronize
        if rank == 0 and args.verbose:
            print("Doing synchronization", flush=True)
        with record_pcl_function("1train-step--9jitter-end"):
            synchronize()

        if profiler is not None:
            profiler.stop("0train-step-block")

        # Iteration end timestamp
        et = time.time()
        iter_time = (et - st)
        dl_time = (dl_et - dl_st)
        
        if train_step_id % args.logging_steps == 0:
            step_print_str = f"Step {train_step_id:4d} / {train_steps}"
            step_print_str += f", Loss : {global_loss.item():9.6f}" if global_loss is not None else f", Loss : {0:9.6f}"
            step_print_str += f", Grad norm : {grad_norm:.6f}"
            step_print_str += f", LR : {lr:.8f}"
            step_print_str += f", Time : {iter_time:.4f} sec"
            step_print_str += f", DL time : {dl_time:.4f} sec"
            step_print_str += f", Tokens Processed : {((train_step_id+1) * (pmap.data_parallelism * args.expert_parallelism * args.batch_size) * args.context_size)}"
            step_print_str += f", Tokens/sec : {(((args.batch_size * args.context_size) / iter_time) / (pmap.pipeline_parallelism * pmap.tensor_parallelism)):.0f}"
            step_print_str += f", Timestamp : {time.time()}"
            
            tflops = 0
            if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
                if hasattr(model, "get_flops"):
                    tflops = (args.batch_size * model.get_flops(args.context_size)*1e-12)/iter_time
            else:
                if hasattr(model.module, "get_flops"):
                    tflops = (args.batch_size * model.module.get_flops(args.context_size)*1e-12)/iter_time
            step_print_str += f", Tflops : {tflops:.2f}"
            step_print_str += f", ETA : {((train_steps - (train_step_id+1))*iter_time)/(60*60):.2f} hr"
            step_print_str += f", Local Loss : {loss.item():9.6f}" if loss is not None else f", Local Loss : {0:9.6f}"
            step_print_str += f", Local Grad norm : {local_grad_norm.item():9.6f}"
            step_print_str += f", Rank : {rank} ({pmap.dp_ind},{pmap.pp_ind},{pmap.ep_ind},{pmap.tp_ind})"
            
            # if (rank == 0) or args.log_all_ranks:
            if (pmap.is_last_stage_rank and pmap.dp_ind == 0 and pmap.tp_ind == 0 and pmap.ep_ind == 0) or args.log_all_ranks:
                print(step_print_str, flush=True)
                if profiler != None:
                    profiler.print_timing_table(sort_by_name=True)

        def choose_checkpoint():
            # Choose a valid checkpoint
            checkpoint = None

            # First checkpoint
            checkpoint1_valid = False
            checkpoint1_dir = os.path.join(args.exp_dir, "checkpoints", "checkpoint1")
            checkpoint1_success_marker_file_path = os.path.join(checkpoint1_dir, "completed")
            if os.path.isfile(checkpoint1_success_marker_file_path):
                checkpoint1_valid = True
            
            # Second checkpoint
            checkpoint2_valid = False
            checkpoint2_dir = os.path.join(args.exp_dir, "checkpoints", "checkpoint2")
            checkpoint2_success_marker_file_path = os.path.join(checkpoint2_dir, "completed")
            if os.path.isfile(checkpoint2_success_marker_file_path):
                checkpoint2_valid = True

            # Choosing the checkpoint to write to.
            if (checkpoint1_valid == True) and (checkpoint2_valid == True):
                # Choose the oldest
                checkpoint1_step = int(open(checkpoint1_success_marker_file_path).readlines()[0])
                checkpoint2_step = int(open(checkpoint2_success_marker_file_path).readlines()[0])
                checkpoint = "checkpoint2" if (checkpoint1_step > checkpoint2_step) else "checkpoint1"
                if rank == 0:
                    print(f"Both checkpoints are valid. Checkpoint 1 step is {checkpoint1_step}. Checkpoint 2 step is {checkpoint2_step}. Choosing oldest checkpoint i.e, {checkpoint}", flush=True)
            elif checkpoint1_valid == True:
                # Choose the other i.e, checkpoint2
                checkpoint = "checkpoint2"
                if rank == 0:
                    print(f"Only checkpoint1 is valid. So choosing {checkpoint}", flush=True)
            elif checkpoint2_valid == True:
                # Choose the other i.e, checkpoint1
                checkpoint = "checkpoint1"
                if rank == 0:
                    print(f"Only checkpoint2 is valid. So choosing {checkpoint}", flush=True)
            else:
                # Starting fresh
                checkpoint = "checkpoint1"
                if rank == 0:
                    print(f"Beginning checkpointing with {checkpoint}", flush=True)
            return checkpoint

        def save_checkpoint(checkpoint=None, save_model_only_checkpoint=False):
            checkpoint = choose_checkpoint() if checkpoint is None else checkpoint
            
            # Create checkpoint directory on rank 0
            # Sync and remove status file on rank 0
            checkpoint_dir = os.path.join(args.exp_dir, "checkpoints", checkpoint)
            checkpoint_success_marker_file_path = os.path.join(checkpoint_dir, "completed")

            if rank == 0:
                print(f"Saving checkpoint to {checkpoint_dir}", flush=True)
                os.makedirs(checkpoint_dir, exist_ok=True)

            # Synchronizing to ensure checkpoint directory is created
            if torch.distributed.is_initialized():
                device_synchronize()
                torch.distributed.barrier()
            
            if rank == 0:
                if os.path.exists(checkpoint_success_marker_file_path):
                    # Remove the status file
                    os.remove(checkpoint_success_marker_file_path)
                    print(f"Removing previous checkpoint status file", flush=True)

            # Model checkpoint (First TP group will store the model checkpoint)
            if ((not args.dp_staggered_checkpointing) and pmap.dp_ind == 0) or (args.dp_staggered_checkpointing and (pmap.dp_ind == (pmap.mp_ind % pmap.data_parallelism))):
                model_checkpoint_file_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")
                if pmap.is_model_parallel:
                    model_checkpoint_file_path = os.path.join(checkpoint_dir, f"model_checkpoint_shard-{pmap.mp_ind}.pth")
                model_checkpoint = {
                    "train_step_id" : train_step_id+1, # Adding 1 because the step is already over
                    "state_dict" : model.state_dict()
                }
                print(f"Rank : {rank}. Saving model checkpoint {model_checkpoint_file_path}", flush=True)
                torch.save(model_checkpoint, model_checkpoint_file_path)
            
            # Synchronizing to ensure model checkpoint is written before writing optimizer checkpoint
            if torch.distributed.is_initialized():
                device_synchronize()
                torch.distributed.barrier()

            if not save_model_only_checkpoint:
                is_optimizer_sharded = args.opt_use_sharded_optimizer or args.opt_use_pgsharded_optimizer or args.opt_use_subsharded_optimizer or args.opt_use_pgsubsharded_optimizer

                # Optimizer checkpoint
                optimizer_checkpoint_file_path = os.path.join(checkpoint_dir, f"optimizer_checkpoint.pth")
                if is_optimizer_sharded:
                    optimizer_checkpoint_file_path = os.path.join(checkpoint_dir, f"optimizer_checkpoint_shard-{rank}.pth")

                if ((not is_optimizer_sharded) and rank == 0) or is_optimizer_sharded:
                    optimizer_checkpoint = {
                        "state_dict" : optimizer.state_dict()
                    }
                    if rank == 0:
                        print(f"Rank : {rank}. Saving optimizer checkpoint {optimizer_checkpoint_file_path} (Limited print to rank 0)", flush=True)
                    torch.save(optimizer_checkpoint, optimizer_checkpoint_file_path)
        
            # Synchronization ensures all ranks have saved the checkpoint
            if torch.distributed.is_initialized():
                device_synchronize()
                torch.distributed.barrier()
            
            if rank == 0:
                with open(checkpoint_success_marker_file_path, "w") as fh:
                    fh.write(f"{train_step_id+1}")
                print(f"Saved checkpoint to {checkpoint} successfully", flush=True)

        # Save checkpoint
        if args.enable_checkpointing and ((train_step_id+1) % (args.checkpoint_steps * args.opt_grad_acc_steps) == 0):
            if rank == 0:
                print(f"Saving checkpoint (hot)", flush=True)
            save_checkpoint()
            if not args.disable_cold_checkpointing:
                if rank == 0:
                    print(f"Saving model only checkpoint (cold)", flush=True)
                num_tokens_processed_in_billion = ((train_step_id+1) * (pmap.data_parallelism * pmap.expert_parallelism) * (args.batch_size * args.context_size) * 1e-9)
                checkpoint = f"step{train_step_id+1}-tokens{round(num_tokens_processed_in_billion):.0f}B"
                save_checkpoint(checkpoint=checkpoint, save_model_only_checkpoint=True)

        if (train_step_id == (train_steps -1)):
            if rank == 0:
                print("Saving final checkpoint", flush=True)
            save_checkpoint(checkpoint="main")

        # Advance step
        train_step_id += 1

        # # Reset profiler
        if profiler != None:
            profiler.reset()
        
        # End of step synchronization
        synchronize()

    tr_et = time.time()
    if rank == 0:
        print(f"Training time : {(tr_et - tr_st)/(60*60):.2f} hr", flush=True)

    # Final synchronization before exiting
    synchronize()

    # # Cleanup
    # if torch.distributed.is_initialized():
    #     torch.distributed.destroy_process_group()