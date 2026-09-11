#!/bin/bash


ckpt="checkpoint_hf"
tasks="arc_easy,arc_challenge,hellaswag,piqa,boolq,sciq,winogrande,openbookqa,mmlu"

python -m lm_eval \
    --model hf \
    --model_args pretrained=$ckpt,dtype=bfloat16 \
    --tasks $tasks \
    --device xpu:0 \
    --batch_size 32 \
    --output_path results