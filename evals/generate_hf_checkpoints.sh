#!/bin/bash

# Change these appropriately
exp_dump_dir="../pretrain_parallel_models"
exp_dir=OLMoE-1B-7B-0924_olmoemix0924_om_omcached_gral_fastmoe-mmlp_bf16_n256xg12_gbs3072-bs1-cs2048-gas1_dp3072-pp1-ep1-tp1_omopt-shard_final

om_checkpoints_dir=${exp_dump_dir}/${exp_dir}/checkpoints/
hf_checkpoints_dir=${exp_dump_dir}/${exp_dir}/hf_checkpoints/

# Create HF checkpoints directory if it doesn't exist
mkdir -p ${hf_checkpoints_dir}

# List all checkpoints
checkpoints=($(ls ${om_checkpoints_dir} | grep -E "^(step|main)" | sort -V))

for checkpoint in "${checkpoints[@]}"; do
    om_checkpoint_dir=${om_checkpoints_dir}/${checkpoint}
    hf_checkpoint_dir=${hf_checkpoints_dir}/${checkpoint}

    if [ ! -f ${om_checkpoint_dir}/completed ]; then
        echo "Checkpoint ${checkpoint} is not fully saved yet. Skipping."
    else
        if [ -f ${hf_checkpoint_dir}/completed ]; then
            echo "Checkpoint ${checkpoint} already converted. Skipping."
        else
            echo "Processing $checkpoint"

            # Run the conversion script
            python convert_olmo_om_checkpoint_to_hf.py --use_init_empty_weights \
                --model_choice allenai/OLMo-1B-hf \
                --om_checkpoint_dir ${om_checkpoint_dir} \
                --hf_checkpoint_dir ${hf_checkpoint_dir} 

            # python convert_olmoe_om_checkpoint_to_hf.py --use_init_empty_weights \
            #     --model_choice allenai/OLMoE-1B-7B-0924 \
            #     --om_checkpoint_dir ${om_checkpoint_dir} \
            #     --hf_checkpoint_dir ${hf_checkpoint_dir}

            # python convert_olmoe_om_checkpoint_to_hf.py \
            #     --model_choice allenai/OLMoE-1B-7B-0924--SE1.125 \
            #     --om_checkpoint_dir ${om_checkpoint_dir} \
            #     --hf_checkpoint_dir ${hf_checkpoint_dir} \
            #     --use_init_empty_weights

            # python convert_olmoe_ppep_om_checkpoint_to_hf.py \
            #     --model_choice allenai/OLMoE-1B-7B-0924--SA3-SH1.5-SE2.25-SC0.5-SI2-L36 \
            #     --om_checkpoint_dir ${om_checkpoint_dir} \
            #     --hf_checkpoint_dir ${hf_checkpoint_dir} \
            #     --use_init_empty_weights

            # Save status file
            touch ${hf_checkpoint_dir}/completed
        fi
    fi
done