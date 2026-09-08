#!/bin/bash

# Run arguments
FINAL=${FINAL:-0}
USE_ADVANCED_LAUNCHER=${USE_ADVANCED_LAUNCHER:-0}
NUM_REPEATS=${NUM_REPEATS:-1}
HOSTFILE_PATH=${HOSTFILE_PATH:-$PBS_NODEFILE}

# Model
MODEL_CHOICE=${MODEL_CHOICE:-allenai/OLMoE-1B-7B-0924}
DTYPE=${DTYPE:-bf16}
USE_HF_MODEL=${USE_HF_MODEL:-0}
ENABLE_SAC=${ENABLE_SAC:-0}
SAC_LEVEL=${SAC_LEVEL:-0}
USE_FAST_MOE=${USE_FAST_MOE:-0}
USE_MERGED_MLP_IN_FAST_MOE=${USE_MERGED_MLP_IN_FAST_MOE:-0}
USE_TRITON_PATH_FOR_GEMM_IN_FAST_MOE=${USE_TRITON_PATH_FOR_GEMM_IN_FAST_MOE:-0}
USE_TRITON_PATH_FOR_NONGEMM_IN_FAST_MOE=${USE_TRITON_PATH_FOR_NONGEMM_IN_FAST_MOE:-0}

# MoE specific
FORCE_UNIFORM_ROUTING=${FORCE_UNIFORM_ROUTING:-0}
USE_LOCAL_ROUTER_AUX_LOSS=${USE_LOCAL_ROUTER_AUX_LOSS:-0}

# Initialization
DISABLE_MODEL_CACHING=${DISABLE_MODEL_CACHING:-0}
USE_OM_CACHED_MODEL=${USE_OM_CACHED_MODEL:-0}
INIT_MODEL_DIR=${INIT_MODEL_DIR:-null}
USE_BROADCAST_FOR_MODEL_INIT=${USE_BROADCAST_FOR_MODEL_INIT:-1}
USE_ALLREDUCE_IN_BROADCAST_MODEL_INIT=${USE_ALLREDUCE_IN_BROADCAST_MODEL_INIT:-0}

# Model parallel
PP=${PP:-1}
EP=${EP:-1}
TP=${TP:-1}
VPP=${VPP:-1}
USE_SP_IN_TP=${USE_SP_IN_TP:-0}
PP_SCHEME=${PP_SCHEME:-gpipe}
REUSE_BUFFERS_IN_1F1B=${REUSE_BUFFERS_IN_1F1B:-1}
USE_PP_FIRST=${USE_PP_FIRST:-0}

# Input
BATCH_SIZE=${BATCH_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

# Scaling
NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-12}

# Dataset
SUPPORTED_DATASETS=("allenai/dolma" "allenai/OLMoE-mix-0924" "HuggingFaceFW/fineweb-edu" "Salesforce/wikitext")
DATASET_CHOICE=${DATASET_CHOICE:-allenai/OLMoE-mix-0924}
NUM_DATA_FILES=${NUM_DATA_FILES:--1}
TOKENIZER=${TOKENIZER:-olmo}
USE_SHARDED_DATASET=${USE_SHARDED_DATASET:-0}
CONTEXT_SIZE=${CONTEXT_SIZE:-2048}

# Optimizer
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
USE_PT_OPTIMIZER=${USE_PT_OPTIMIZER:-0}
USE_OM_OPTIMIZER=${USE_OM_OPTIMIZER:-0}
USE_OM_SHARDED_OPTIMIZER=${USE_OM_SHARDED_OPTIMIZER:-0}
USE_OM_SUBSHARDED_OPTIMIZER=${USE_OM_SUBSHARDED_OPTIMIZER:-0}
USE_OM_PGSHARDED_OPTIMIZER=${USE_OM_PGSHARDED_OPTIMIZER:-0}
USE_OM_PGSUBSHARDED_OPTIMIZER=${USE_OM_PGSUBSHARDED_OPTIMIZER:-0}
OPT_USE_ALLREDUCE_FOR_GRAD_ACC=${OPT_USE_ALLREDUCE_FOR_GRAD_ACC:-0}
OPT_USE_ALLREDUCE_FOR_PARAM_GATHER=${OPT_USE_ALLREDUCE_FOR_PARAM_GATHER:-0}
OPT_USE_CHUNKED_ALLREDUCE=${OPT_USE_CHUNKED_ALLREDUCE:-0}
OPT_USE_FP32_FOR_GRAD_ACC=${OPT_USE_FP32_FOR_GRAD_ACC:-0}
DISABLE_DELAYED_GRAD_CLIPPING=${DISABLE_DELAYED_GRAD_CLIPPING:-0}
USE_ONE_STEP_GRAD_CLIPPING=${USE_ONE_STEP_GRAD_CLIPPING:-0}

# Learning rate related
LR=${LR:-null}
MIN_LR=${MIN_LR:-null}
WARMUP_STEPS=${WARMUP_STEPS:-null}
LR_TRAIN_STEPS=${LR_TRAIN_STEPS:-null}
LR_SCHEDULE=${LR_SCHEDULE:-null}
WEIGHT_DECAY=${WEIGHT_DECAY:-null}

# Checkpointing and profiling
ENABLE_CHECKPOINTING=${ENABLE_CHECKPOINTING:-1}
DISABLE_COLD_CHECKPOINTING=${DISABLE_COLD_CHECKPOINTING:-0}
DP_STAGGERED_CHECKPOINTING=${DP_STAGGERED_CHECKPOINTING:-0}
DISABLE_OPTIMIZER_STATE_CHECKPOINTING=${DISABLE_OPTIMIZER_STATE_CHECKPOINTING:-0}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-1000}
DEBUG=${DEBUG:-0}
PAPER=${PAPER:-0}
PROFILE=${PROFILE:-0}
PROFILE_MODULES=${PROFILE_MODULES:-0}

# Debug options
LOG_MEMORY_USAGE=${LOG_MEMORY_USAGE:-0}
DUMP_WEIGHT_GRADIENTS=${DUMP_WEIGHT_GRADIENTS:-0}
ENABLE_BAD_GRAD_CHECK=${ENABLE_BAD_GRAD_CHECK:-0}
SKIP_OPTIMIZER_STEP=${SKIP_OPTIMIZER_STEP:-0}
SKIP_OPTIMIZER_STEP_WEIGHT_UPDATE=${SKIP_OPTIMIZER_STEP_WEIGHT_UPDATE:-0}
SKIP_BACKWARD_PASS=${SKIP_BACKWARD_PASS:-0}
LOG_ALL_RANKS=${LOG_ALL_RANKS:-0}
ENABLE_PP_DEBUG=${ENABLE_PP_DEBUG:-0}

VERBOSE=${VERBOSE:-0}
EXIT_STEPS=${EXIT_STEPS:--1}
ENABLE_TB_LOGGING=${ENABLE_TB_LOGGING:-0}

# TEMPORARY
USE_LATEST_TRAINER=${USE_LATEST_TRAINER:-0}

# Experiment id
JOB_ID=${PBS_JOBID:-0}

# Compatibility checks
if [[ ! " ${SUPPORTED_DATASETS[@]} " =~ " ${DATASET_CHOICE} " ]]; then
    echo "Error: Unsupported dataset choice. Supported datasets are: ${SUPPORTED_DATASETS[@]}"
    exit 1
fi

OPTIMIZER_SUM=$(( USE_PT_OPTIMIZER + USE_OM_OPTIMIZER + USE_OM_SHARDED_OPTIMIZER + USE_OM_PGSHARDED_OPTIMIZER + USE_OM_SUBSHARDED_OPTIMIZER + USE_OM_PGSUBSHARDED_OPTIMIZER ))
if [ $OPTIMIZER_SUM -ne 1 ]; then
    echo "Error: Exactly one of USE_PT_OPTIMIZER, USE_OM_OPTIMIZER, USE_OM_SHARDED_OPTIMIZER, USE_OM_PGSHARDED_OPTIMIZER, USE_OM_SUBSHARDED_OPTIMIZER, and USE_OM_PGSUBSHARDED_OPTIMIZER must be set to 1."
    exit 1
fi

if [ $PP -ne 1 ] && [ $USE_HF_MODEL -eq 1 ]; then
    echo "Error: PP can only be used with USE_HF_MODEL set to 0."
    exit 1
fi

if [ $EP -ne 1 ] && [ $USE_HF_MODEL -eq 1 ]; then
    echo "Error: EP can only be used with USE_HF_MODEL set to 0."
    exit 1
fi

if [ $TP -ne 1 ] && [ $USE_HF_MODEL -eq 1 ]; then
    echo "Error: TP can only be used with USE_HF_MODEL set to 0."
    exit 1
fi

# Derived variables
NUM_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))
DP=$(( NUM_GPUS / (PP * EP * TP)  ))
GLOBAL_BATCH_SIZE=$(( DP * EP * BATCH_SIZE ))

# Directory name should not have slashes
MODEL_KEY=$(echo $MODEL_CHOICE | cut -d'/' -f2)
DATASET_KEY=$(echo $DATASET_CHOICE | tr '/' '_')

# Directories
# STORAGE_DIR="/lus/flare/projects/$PROJECT/$USER"
STORAGE_DIR="."
DATASET_DIR="$STORAGE_DIR/datasets/${DATASET_KEY}/preprocessed/${TOKENIZER}/"
CACHED_MODELS_DIR="$STORAGE_DIR/cached_models"
OM_CACHED_MODELS_DIR="$STORAGE_DIR/parallel_models"
EXPERIMENT_DUMP_DIR="$STORAGE_DIR/pretrain_parallel_models"

cmd_args="--model_choice $MODEL_CHOICE --dataset_dir $DATASET_DIR --dtype $DTYPE --batch_size ${BATCH_SIZE} --context_size $CONTEXT_SIZE --opt_grad_acc_steps $GRAD_ACC_STEPS "
cmd_args+=" --data_parallelism $DP --pipeline_parallelism $PP --expert_parallelism $EP --tensor_parallelism $TP --virtual_pipeline_parallelism $VPP"

# If LR_TRAIN_STEPS is set to null, it will be calculated based on the dataset size and batch size. Otherwise, use the provided value.
if [[ $LR != "null" ]]; then
    cmd_args+=" --lr $LR"
fi
if [[ $MIN_LR != "null" ]]; then
    cmd_args+=" --min_lr $MIN_LR"
fi
if [[ $WARMUP_STEPS != "null" ]]; then
    cmd_args+=" --warmup_steps $WARMUP_STEPS"
fi
if [[ $LR_SCHEDULE != "null" ]]; then
    cmd_args+=" --lr_schedule $LR_SCHEDULE"
fi
if [[ $WEIGHT_DECAY != "null" ]]; then
    cmd_args+=" --weight_decay $WEIGHT_DECAY"
fi
if [[ $LR_TRAIN_STEPS != "null" ]]; then
    cmd_args+=" --lr_train_steps $LR_TRAIN_STEPS"
fi

exp_name="${MODEL_KEY}"

if [[ $DATASET_CHOICE == "allenai/dolma" ]]; then
    exp_name+="_dolma"
elif [[ $DATASET_CHOICE == "allenai/OLMoE-mix-0924" ]]; then
    exp_name+="_olmoemix0924"
elif [[ $DATASET_CHOICE == "HuggingFaceFW/fineweb-edu" ]]; then
    exp_name+="_finewebedu"
elif [[ $DATASET_CHOICE == "Salesforce/wikitext" ]]; then
    exp_name+="_wikitext"
else
    echo "Unsupported dataset choice: $DATASET_CHOICE"
    exit 1
fi

if [[ $USE_HF_MODEL -eq 1 ]]; then
    cmd_args+=" --use_hf_model"
    exp_name+="_hf"
else
    exp_name+="_om"

    if [[ $USE_OM_CACHED_MODEL -eq 1 ]]; then
        cmd_args+=" --use_om_cached_model --om_cached_models_dir $OM_CACHED_MODELS_DIR"
        exp_name+="_omcached"
    fi

    if [[ $INIT_MODEL_DIR != "null" ]]; then
        cmd_args+=" --init_model_dir $INIT_MODEL_DIR"
        exp_name+="_finetune"
    fi

    if [[ $MODEL_CHOICE == *"OLMoE"* ]]; then
        if [[ $USE_LOCAL_ROUTER_AUX_LOSS -eq 1 ]]; then
            cmd_args+=" --use_local_router_aux_loss"
            exp_name+="_lral"
        else
            exp_name+="_gral"
        fi

        if [[ $FORCE_UNIFORM_ROUTING -eq 1 ]]; then
            cmd_args+=" --force_uniform_routing"
            exp_name+="_fur"
        fi

        if [[ $USE_FAST_MOE -eq 1 ]]; then
            cmd_args+=" --use_fast_moe"
            exp_name+="_fastmoe"

            if [[ $USE_MERGED_MLP_IN_FAST_MOE -eq 1 ]]; then
                cmd_args+=" --use_merged_mlp_in_fast_moe"
                exp_name+="-mmlp"
            fi

            if [[ $USE_TRITON_PATH_FOR_GEMM_IN_FAST_MOE -eq 1 ]]; then
                cmd_args+=" --use_triton_path_for_gemm_in_fast_moe"
                exp_name+="-tritongemm"
            fi

            if [[ $USE_TRITON_PATH_FOR_NONGEMM_IN_FAST_MOE -eq 1 ]]; then
                cmd_args+=" --use_triton_path_for_nongemm_in_fast_moe"
                exp_name+="-tritonnongemm"
            fi
        fi
    fi

    if [[ $MODEL_CHOICE == *"DeepSeek"* ]]; then
        if [[ $FORCE_UNIFORM_ROUTING -eq 1 ]]; then
            cmd_args+=" --force_uniform_routing"
            exp_name+="_fur"
        fi
    fi
    
    if [[ $ENABLE_SAC -eq 1 ]]; then
        cmd_args+=" --use_activation_checkpointing --activation_checkpointing_level $SAC_LEVEL"
        exp_name+="_ac${SAC_LEVEL}"
    fi
fi

exp_name+="_${DTYPE}_n${NUM_NODES}xg${GPUS_PER_NODE}"
exp_name+="_gbs${GLOBAL_BATCH_SIZE}-bs${BATCH_SIZE}-cs${CONTEXT_SIZE}-gas${GRAD_ACC_STEPS}"

# Parallelism
exp_name+="_dp${DP}-pp${PP}-ep${EP}-tp${TP}"

# Experiment name backward compatibility
if [[ $VPP -ne 1 ]]; then
    exp_name+="-vpp${VPP}"
fi

# TP options
if [[ $TP -ne 1 ]]; then
    if [[ $USE_SP_IN_TP -eq 1 ]]; then
        cmd_args+=" --use_sequence_parallelism_in_tp"
        exp_name+="_sp${USE_SP_IN_TP}"
    fi
fi

# PP options
if [[ $PP -ne 1 ]]; then
    cmd_args+=" --pp_scheme $PP_SCHEME --micro_batch_size ${MICRO_BATCH_SIZE}"
    exp_name+="-ps${PP_SCHEME}-mbs${MICRO_BATCH_SIZE}"
    # 1f1b options
    if [[ $PP_SCHEME == "1f1b" ]]; then
        if [[ $REUSE_BUFFERS_IN_1F1B -eq 1 ]]; then
            cmd_args+=" --reuse_buffers_in_1f1b"    
        fi
        exp_name+="-rb${REUSE_BUFFERS_IN_1F1B}"
    fi

    if [[ $USE_PP_FIRST -eq 1 ]]; then
        cmd_args+=" --use_pp_first"
        exp_name+="-ppfirst"
    fi
fi

# Optimizer related
if [ $USE_PT_OPTIMIZER -eq 1 ]; then
    cmd_args+=" --opt_use_pt_optimizer"
    exp_name+="_ptopt-ddp"
else
    if [ $USE_OM_OPTIMIZER -eq 1 ]; then
        exp_name+="_omopt-ddp"
    fi

    if [ $USE_OM_SHARDED_OPTIMIZER -eq 1 ]; then
        cmd_args+=" --opt_use_sharded_optimizer"
        exp_name+="_omopt-shard"
    fi

    if [ $USE_OM_SUBSHARDED_OPTIMIZER -eq 1 ]; then
        cmd_args+=" --opt_use_subsharded_optimizer"
        exp_name+="_omopt-subshard"
    fi

    if [ $USE_OM_PGSHARDED_OPTIMIZER -eq 1 ]; then
        cmd_args+=" --opt_use_pgsharded_optimizer"
        exp_name+="_omopt-pgshard"
    fi

    if [ $USE_OM_PGSUBSHARDED_OPTIMIZER -eq 1 ]; then
        cmd_args+=" --opt_use_pgsubsharded_optimizer"
        exp_name+="_omopt-pgsubshard"
    fi

    if [ $OPT_USE_FP32_FOR_GRAD_ACC -eq 1 ]; then
        cmd_args+=" --opt_use_fp32_for_grad_acc"
        exp_name+="-gafp32"
    fi

    if [ $OPT_USE_ALLREDUCE_FOR_GRAD_ACC -eq 1 ]; then
        cmd_args+=" --opt_use_allreduce_for_grad_acc"
        exp_name+="-gaar"
    fi

    if [ $OPT_USE_ALLREDUCE_FOR_PARAM_GATHER -eq 1 ]; then
        cmd_args+=" --opt_use_allreduce_for_param_gather"
        exp_name+="-pgar"
    fi

    if [ $OPT_USE_CHUNKED_ALLREDUCE -eq 1 ]; then
        cmd_args+=" --opt_use_chunked_allreduce"
        exp_name+="-arch"
    fi

fi

if [ $DISABLE_DELAYED_GRAD_CLIPPING -eq 1 ]; then
    cmd_args+=" --opt_disable_delayed_grad_clipping"
    exp_name+="-gcfromstart"
fi

if [ $USE_ONE_STEP_GRAD_CLIPPING -eq 1 ]; then
    cmd_args+=" --opt_use_one_step_grad_clipping"
    exp_name+="-onestepgc"
fi

# Debug related
if [ $SKIP_OPTIMIZER_STEP -eq 1 ]; then
    cmd_args+=" --skip_optimizer_step"
    exp_name+="_skipoptstep"
fi

if [ $SKIP_OPTIMIZER_STEP_WEIGHT_UPDATE -eq 1 ]; then
    cmd_args+=" --skip_optimizer_step_weight_update"
    exp_name+="_skipoptstepwupdate"
fi

if [ $LOG_MEMORY_USAGE -eq 1 ]; then
    cmd_args+=" --log_memory_usage"
fi

if [ $DUMP_WEIGHT_GRADIENTS -eq 1 ]; then
    cmd_args+=" --dump_weight_gradients"
fi

if [ $SKIP_BACKWARD_PASS -eq 1 ]; then
    cmd_args+=" --skip_backward_pass"
    exp_name+="_skipbwd"
fi

if [ $DISABLE_MODEL_CACHING -eq 1 ]; then
    cmd_args+=" --disable_model_caching"
fi

if [ $PROFILE -eq 1 ]; then
    cmd_args+=" --profile"
    exp_name+="_profile"
fi

if [ $PROFILE_MODULES -eq 1 ]; then
    cmd_args+=" --profile_modules"
    exp_name+="_profilemods"
fi

# Experiment segregation
if [ $FINAL -eq 1 ]; then
    exp_name+="_final"
else
    if [ $DEBUG -eq 1 ]; then
        exp_name+="_debug"
    elif [ $PAPER -eq 1 ]; then
        exp_name+="_paper"
    else
        exp_name+="_temp"
    fi
fi

if [ $NUM_DATA_FILES -ne -1 ]; then
    cmd_args+=" --num_data_files $NUM_DATA_FILES"
fi

if [ $USE_SHARDED_DATASET -eq 1 ]; then
    cmd_args+=" --use_sharded_dataset"
fi

if [ $VERBOSE -eq 1 ]; then
    cmd_args+=" --verbose"
fi

if [ $ENABLE_TB_LOGGING -eq 1 ]; then
    cmd_args+=" --enable_tb_logging"
fi

if [ $ENABLE_BAD_GRAD_CHECK -eq 1 ]; then
    cmd_args+=" --enable_bad_grad_check"
fi

if [ $LOG_ALL_RANKS -eq 1 ]; then
    cmd_args+=" --log_all_ranks"
fi

if [ $ENABLE_PP_DEBUG -eq 1 ]; then
    cmd_args+=" --enable_pp_debug"
fi

if [ $ENABLE_CHECKPOINTING -eq 1 ]; then
    cmd_args+=" --enable_checkpointing"
    cmd_args+=" --checkpoint_steps $CHECKPOINT_STEPS"

    if [ $DISABLE_OPTIMIZER_STATE_CHECKPOINTING -eq 1 ]; then
        cmd_args+=" --disable_optimizer_state_checkpointing"
    fi

    if [ $DISABLE_COLD_CHECKPOINTING -eq 1 ]; then
        cmd_args+=" --disable_cold_checkpointing"
    fi

    if [ $DP_STAGGERED_CHECKPOINTING -eq 1 ]; then
        cmd_args+=" --dp_staggered_checkpointing"
    fi
fi

if [ $EXIT_STEPS -ne -1 ]; then
    cmd_args+=" --exit_steps $EXIT_STEPS"
fi

if [ $USE_BROADCAST_FOR_MODEL_INIT -eq 1 ]; then
    cmd_args+=" --use_broadcast_for_model_init"
fi

if [ $USE_ALLREDUCE_IN_BROADCAST_MODEL_INIT -eq 1 ]; then
    cmd_args+=" --use_allreduce_in_broadcast_model_init"
fi

if [ $USE_LATEST_TRAINER -eq 1 ]; then
    cmd_args+=" --use_latest_trainer"
fi

# Final experiment directory
exp_dir=$EXPERIMENT_DUMP_DIR/$exp_name
cmd_args+=" --exp_dir $exp_dir"

# Cached model directory
cmd_args+=" --cached_models_dir $CACHED_MODELS_DIR"

# Create necessary directories
mkdir -p $CACHED_MODELS_DIR
mkdir -p $EXPERIMENT_DUMP_DIR
mkdir -p $exp_dir
mkdir -p ${exp_dir}/checkpoints
mkdir -p ${exp_dir}/checkpoints/checkpoint1
mkdir -p ${exp_dir}/checkpoints/checkpoint2

# Use the same nodes instead of exiting
for ((i=1; i<=NUM_REPEATS; i++)); do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    log_path=$exp_dir/log_${JOB_ID}_${TIMESTAMP}.txt
    
    dist_prefix=""
    if [ $NUM_GPUS -gt 1 ]; then
        if [ $USE_ADVANCED_LAUNCHER -eq 1 ]; then
            dist_prefix="bash launch_dist_advanced.sh $GPUS_PER_NODE $NUM_NODES"
        else
            dist_prefix="HOSTFILE_PATH=$HOSTFILE_PATH bash launch_dist.sh $GPUS_PER_NODE $NUM_NODES"
        fi
    fi
    cmd="$dist_prefix python dist_train.py $cmd_args | tee $log_path"

    echo $cmd
    if [ $i -gt 1 ]; then
        sleep 60
    fi
    eval $cmd
done
