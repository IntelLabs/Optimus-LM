# Optimus-LM
![GitHub License](https://img.shields.io/github/license/IntelLabs/Optimus-LM)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/Optimus-LM/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/Optimus-LM)
<!-- UNCOMMENT AS NEEDED
[![Unit Tests](https://github.com/IntelLabs/ConvAssist/actions/workflows/run_unittests.yaml/badge.svg?branch=covassist-cleanup)](https://github.com/IntelLabs/ConvAssist/actions/workflows/run_unittests.yaml)
[![pytorch](https://img.shields.io/badge/PyTorch-v2.4.1-green?logo=pytorch)](https://pytorch.org/get-started/locally/)
![python-support](https://img.shields.io/badge/Python-3.12-3?logo=python)
-->

# About
Large scale LLM pretraining of MoE models on the Aurora super computer. (Tech report : https://arxiv.org/abs/2604.00785)

# Environment setup (Aurora)
```
# Standard
module load frameworks
source setenv_aurora_essential.sh

# Custom (Required only for FastMoE path)
module load frameworks
source setenv_aurora_essential.sh
python3 -m venv fastmoe --system-site-packages
source fastmoe/bin/activate
cd pytorch_xpu_extensions/pcl.xpu.customops
python setup.py install

# Basic compute and comms checks
python tests/test_compute.py
bash launch_dist.sh 12 1 python tests/test_comms.py

# Setup soft links (storage_dir is used for storing cached/serial/parallel models, experiments, and job logs)
bash link_folders.sh /lus/flare/projects/Aurora_deployment/$USER/storage_dir
or
bash link_folders.sh /lus/flare/projects/Intel-Aurora/$USER/storage_dir

```

# Data preparation 
```
# If you have access to datasets folder, then create a datasets softlink and ignore next steps
ln -s /lus/flare/projects/Aurora_deployment/dvooturi/datasets datasets
or
ln -s /lus/flare/projects/Intel-Aurora/dvooturi/datasets datasets

# Download dataset
mkdir -p datasets
mkdir -p datasets/allenai_OLMoE-mix-0924
python scripts/generate_urls.py datasets/allenai_OLMoE-mix-0924
bash scripts/download.sh datasets/allenai_OLMoE-mix-0924

# For deepseek, use "--model_choice deepseek-ai/DeepSeek-V3"
# Data tokenization (Quick check) (Add NUM_DATA_FILES=1 as prefix to the training run command)
python scripts/tokenize_data.py --num_files 1 --verbose --model_choice allenai/OLMoE-1B-7B-0924
python scripts/generate_metadata.py --num_files 1 --model_choice allenai/OLMoE-1B-7B-0924

# Data tokenization (Full dataset) (Required)
bash launch_dist.sh 8 256 python scripts/tokenize_data.py allenai/OLMoE-1B-7B-0924 --num_files 2903 --verbose --model_choice allenai/OLMoE-1B-7B-0924
python scripts/generate_metadata.py --num_files 2903 --model_choice allenai/OLMoE-1B-7B-0924

# Token shuffling (Full dataset) (Optional) (If you do, then you can use USE_SHARDED_DATASET=1 for efficient data loading)
bash launch_dist.sh 8 256 python scripts/generate_shards.py --num_files 2903 --model_choice allenai/OLMoE-1B-7B-0924

```

# OLMo Pretraining (Standard)
```
# Get single node (Change -A appropriately)
qsub -l select=1 -l walltime=01:00:00 -A Intel-Aurora -q debug -I -l filesystems=flare

# Run
EXIT_STEPS=32 MODEL_CHOICE=allenai/OLMo-1B-hf DATASET_CHOICE=allenai/OLMoE-mix-0924 USE_OM_SHARDED_OPTIMIZER=1 bash run_experiment.sh
```

# OLMoE Pretraining (Standard)
```

# Setup environment
qsub -l select=8 -l walltime=01:00:00 -A Intel-Aurora -q debug-scaling -I -l filesystems=flare
module load frameworks
source setenv_aurora_essential.sh

# Generate serial and parallel models
bash launch_dist.sh 12 1 python generate_serial_model.py --model_choice allenai/OLMoE-1B-7B-0924--SE1.125 
python generate_parallel_model.py --model_choice allenai/OLMoE-1B-7B-0924--SE1.125
bash launch_dist.sh 12 1 python generate_parallel_model.py --model_choice allenai/OLMoE-1B-7B-0924--SE1.125 --expert_parallelism 12
bash launch_dist.sh 12 4 python generate_parallel_model.py --model_choice allenai/OLMoE-1B-7B-0924--SE1.125 --expert_parallelism 12 --pipeline_parallelism 4

# Single node (DP=12 vs EP=12)
EXIT_STEPS=32 NUM_NODES=1 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=1 USE_OM_SHARDED_OPTIMIZER=1 bash run_experiment.sh
EXIT_STEPS=32 NUM_NODES=1 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 

# Multi node (DP96 vs DP8-EP12 vs DP2-PP4-EP12)
EXIT_STEPS=32 NUM_NODES=8 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=1 USE_OM_SHARDED_OPTIMIZER=1 bash run_experiment.sh
EXIT_STEPS=32 NUM_NODES=8 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 
EXIT_STEPS=32 NUM_NODES=8 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 PP=4 PP_SCHEME=1f1b BATCH_SIZE=4 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 
```

# OLMoE Pretraining (Standard vs FastMoE)
```
# Get nodes
qsub -l select=8 -l walltime=01:00:00 -A Intel-Aurora  -q debug-scaling -I -l filesystems=flare
module load frameworks
source fastmoe/bin/activate
source setenv_aurora_essential.sh

# Generate FastMoE parallel models. (Run serial and baseline models generations from previous section if you haven't done)
python generate_parallel_model.py --model_choice allenai/OLMoE-1B-7B-0924--SE1.125 --use_merged_mlp_in_fast_moe
bash launch_dist.sh 12 1 python generate_parallel_model.py --model_choice allenai/OLMoE-1B-7B-0924--SE1.125 --expert_parallelism 12 --use_merged_mlp_in_fast_moe
bash launch_dist.sh 12 4 python generate_parallel_model.py --model_choice allenai/OLMoE-1B-7B-0924--SE1.125 --expert_parallelism 12 --pipeline_parallelism 4 --use_merged_mlp_in_fast_moe

# Single node (DP12) (baseline, fastmoe)
EXIT_STEPS=32 NUM_NODES=1 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 USE_OM_SHARDED_OPTIMIZER=1 bash run_experiment.sh
EXIT_STEPS=32 NUM_NODES=1 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 USE_FAST_MOE=1 USE_MERGED_MLP_IN_FAST_MOE=1 USE_OM_SHARDED_OPTIMIZER=1 bash run_experiment.sh

# Single node (EP12) (baseline, fastmoe)
EXIT_STEPS=32 NUM_NODES=1 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 
EXIT_STEPS=32 NUM_NODES=1 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 USE_FAST_MOE=1 USE_MERGED_MLP_IN_FAST_MOE=1 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 

# Multi node (DP8-EP12) (baseline, fastmoe)
EXIT_STEPS=32 NUM_NODES=8 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 
EXIT_STEPS=32 NUM_NODES=8 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 USE_FAST_MOE=1 USE_MERGED_MLP_IN_FAST_MOE=1 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 

# Multi node (DP2-PP4-EP12) (baseline, fastmoe)
EXIT_STEPS=32 NUM_NODES=8 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 PP=4 PP_SCHEME=1f1b BATCH_SIZE=4 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 
EXIT_STEPS=32 NUM_NODES=8 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 PP=4 PP_SCHEME=1f1b BATCH_SIZE=4 USE_FAST_MOE=1 USE_MERGED_MLP_IN_FAST_MOE=1 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 

```

# OLMoE model scaling
For demonstration, we used allenai/OLMoE-1B-7B-0924--SE1.125 as the model choice. You can change the tag (the part after the last '--') to scale the model. The following are the tags that can be used to scale the model:

'L'  =  Overrides number of layers  
'SA' = Scales attention heads  
'SH' = Scales hidden size  
'SE' = Scales experts  
'SC' = Scales num_experts_per_token  
'SI' = Scales intermediate size  
'SM' = SA U SH  
'S'  = SA U SH U SE U SI  

Paper variants :  
Mula-7B-A1B    : allenai/OLMoE-1B-7B-0924  
Mula-20B-A2B   : allenai/OLMoE-1B-7B-0924--SE1.5-L32  
Mula-100B-A7B  : allenai/OLMoE-1B-7B-0924--SA1.5-SH1.5-SE2.25-SI1.5-L48  
Mula-220B-A10B : allenai/OLMoE-1B-7B-0924--SA1.5-SH1.5-SE3.75-SI1.5-L64  

# DeepSeek-V3 Trillion parameter model pretraining
DeepSeek-V3--N264-L96 is a 1.1 Trillion parameter model with 58 Billion active parameters. It is obtained from the original DeepSeek-V3 by increasing the number of experts from 256 to 264 and the number of layers from 61 to 96 respectively. For dataset, refer to "Data preparation" section above. It is recommended to test the small debug model first to validate the setup.

```

# Debug model (Setup environment)
qsub -l select=8 -l walltime=01:00:00 -A Intel-Aurora -q debug-scaling -I -l filesystems=flare
module load frameworks
source setenv_aurora_essential.sh

# Debug model (Generation : Serial, Parallel_PP4-EP12)
bash launch_dist.sh 12 4 python generate_dsv3_serial_model.py --model_choice deepseek-ai/DeepSeek-V3--L32-DL1-H2048-I10944-MI1408-N72-K6-SE2-A16
bash launch_dist.sh 12 4 python generate_dsv3_parallel_model.py --model_choice deepseek-ai/DeepSeek-V3--L32-DL1-H2048-I10944-MI1408-N72-K6-SE2-A16 --pipeline_parallelism 4 --expert_parallelism 12 

# Debug model (Run with DP2-PP4-EP12 parallelization config) (Add FORCE_UNIFORM_ROUTING=1 if it gets stuck. Add USE_SHARDED_DATASET=1 if you only have sharded data available)
EXIT_STEPS=32 NUM_NODES=8 USE_OM_CACHED_MODEL=1 EP=12 PP=4 BATCH_SIZE=4 PP_SCHEME=1f1b MODEL_CHOICE=deepseek-ai/DeepSeek-V3--L32-DL1-H2048-I10944-MI1408-N72-K6-SE2-A16 TOKENIZER=deepseek USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh

# DSV3-1Trillion model (Setup environment)
qsub -l select=24 -l walltime=01:00:00 -A Intel-Aurora -q debug-scaling -I -l filesystems=flare
module load frameworks
source setenv_aurora_essential.sh

# DSV3-1Trillion model (Generation : Serial, Parallel_PP24-EP12)
bash launch_dist.sh 12 24 python generate_dsv3_serial_model.py --model_choice deepseek-ai/DeepSeek-V3--N264-L96
bash launch_dist.sh 12 24 python generate_dsv3_parallel_model.py --model_choice deepseek-ai/DeepSeek-V3--N264-L96 --pipeline_parallelism 24 --expert_parallelism 12 

# DSV3-1Trillion model (Check on 24 nodes with DP1-PP24-EP12 without optimizer)
SKIP_OPTIMIZER_STEP=1 NUM_NODES=24 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=deepseek-ai/DeepSeek-V3--N264-L96 TOKENIZER=deepseek EP=12 PP=24 BATCH_SIZE=24 PP_SCHEME=1f1b ENABLE_SAC=1 SAC_LEVEL=7 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh

# DSV3-1Trillion model training on 192 nodes with DP8-PP24-EP12 config.
qsub -A Intel-Aurora submit_dsv3-1trillion_job.pbs

```
