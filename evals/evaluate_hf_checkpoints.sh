#!/bin/bash

exp_dump_dir=/lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models

# Experiment details and tasks
#########################################################################
# exp_name=OLMo-1B-hf_om_bf16_n128xg12_gbs1536-bs1-cs4096-gas1_dp1536-pp1-ep1-tp1_omopt-shard_final
# tasks="arc_easy,arc_challenge,hellaswag,piqa,boolq,sciq,winogrande,openbookqa,mmlu"
# extra_model_args=""
#########################################################################

##########################################################################
# exp_name=OLMoE-1B-7B-0924--SE1.125_om_gral_bf16_n128xg12_gbs1536-bs1-cs4096-gas1_dp128-pp1-ep12-tp1_omopt-pgshard_final
# tasks="arc_easy,arc_challenge,hellaswag,piqa,boolq,sciq,winogrande,openbookqa,mmlu"
# extra_model_args=""
##########################################################################

# 100B
exp_name=OLMoE-1B-7B-0924--SA3-SH1.5-SE2.25-SC0.5-SI2-L36_om_omcached_gral_ac1_bf16_n256xg12_gbs6144-bs8-cs4096-gas1_dp64-pp4-ep12-tp1-ps1f1b-mbs1-rb1_omopt-pgshard-pgar_final
tasks="arc_easy,arc_challenge,hellaswag,piqa,boolq,sciq,winogrande,openbookqa,mmlu"
extra_model_args="parallelize=True"

# Get all checkpoints
hf_checkpoints_dir=$exp_dump_dir/$exp_name/hf_checkpoints
output_dir=$exp_dump_dir/$exp_name/results
checkpoints=($(ls ${hf_checkpoints_dir} | grep -E "^(step|main)" | sort -V))

# Later use these for parallel processing
rank=0
world_size=1
device="xpu"
# device_id=0
# device="xpu:$device_id"
# echo "Rank: $rank, World Size : $world_size, Device ID: $device_id, Device: $device"

# Access elements in a loop
checkpoint_count=${#checkpoints[@]}
for ((i=0; i<$checkpoint_count; i=i+1))
do
    # Only process if i % world_size == 0
    if (( i % world_size != rank )); then
        continue
    fi

    cp="${checkpoints[$((i))]}"
    pretrained=$hf_checkpoints_dir/$cp
    output_path=${output_dir}/$cp

    if [ ! -d "$output_path" ]; then
        echo "Processing Checkpoint $cp (Rank : $rank)"
        python -m lm_eval \
            --model hf \
            --model_args pretrained=$pretrained,dtype=bfloat16,$extra_model_args \
            --tasks $tasks \
            --device $device \
            --batch_size 32 \
            --output_path $output_path
    else
        echo "Checkpoint $cp already processed. Skipping."
    fi
done
