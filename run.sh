rm -rf ./pretrain_parallel_models/*_temp

EXIT_STEPS=32 NUM_NODES=1 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=1 USE_OM_SHARDED_OPTIMIZER=1 bash run_experiment.sh
EXIT_STEPS=32 NUM_NODES=1 USE_OM_CACHED_MODEL=1 MODEL_CHOICE=allenai/OLMoE-1B-7B-0924--SE1.125 DATASET_CHOICE=allenai/OLMoE-mix-0924 EP=12 USE_OM_PGSHARDED_OPTIMIZER=1 bash run_experiment.sh 

python compare_losses.py --config_path compare_config.json