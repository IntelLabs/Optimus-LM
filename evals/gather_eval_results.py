import os
import json
import argparse

import numpy as np


benchmark_metric_map = {
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "boolq": "acc,none",
    "sciq": "acc,none",
    "winogrande": "acc,none",
    "openbookqa": "acc_norm,none",
    "mmlu": "acc,none"
}

benchmarks = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "boolq", "sciq", "winogrande", "openbookqa", "mmlu"]

parser = argparse.ArgumentParser(description="Gather evaluation results from multiple checkpoints")
parser.add_argument('--results_dir_list', nargs='+', type=str, default=["results/checkpoint_hf"], help='List of result directories')

args = parser.parse_args()

json_result_paths = []
for results_dir in args.results_dir_list:
    # List files in results_dir and find the latest checkpoint
    checkpoint_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    print(results_dir, checkpoint_files)
    assert len(checkpoint_files) == 1, "Expected exactly one JSON result file in the results directory."
    json_result_path = os.path.join(results_dir, checkpoint_files[0])
    json_result_paths.append(json_result_path)

# Get data
for benchmark in benchmarks:
    print(f"{benchmark:20s} & ", end="")
    for i,json_result_path in enumerate(json_result_paths):
        with open(json_result_path, 'r') as f:
            results = json.load(f)["results"]
        # print(f"{results[benchmark].get(benchmark_metric_map[benchmark]):.4f} ", end="")
        print(f"{results[benchmark].get(benchmark_metric_map[benchmark])*100:.1f} ", end="")
        if i != len(json_result_paths) - 1:
            print(" & ", end="")
        else:
            print("\\\\ \hline", end="")
    print()

# Get average
print(f"{'Average':20s} & ", end="")
for i,json_result_path in enumerate(json_result_paths):
    with open(json_result_path, 'r') as f:
        results = json.load(f)["results"]
    metric_values = []
    for benchmark in benchmarks:
        metric_value = results[benchmark].get(benchmark_metric_map[benchmark])
        if metric_value is not None:
            metric_values.append(metric_value)
    average_metric = np.mean(metric_values) if metric_values else 0.0
    print(f"{average_metric*100:.1f} ", end="")
    if i != len(json_result_paths) - 1:
        print(" & ", end="")
    else:        
        print("\\\\", end="")
print()


# python gather_eval_results.py --results_dir_list /lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models/OLMo-1B-hf_olmoemix0924_om_omcached_bf16_n256xg12_gbs3072-bs1-cs2048-gas1_dp3072-pp1-ep1-tp1_omopt-shard_final/results/main/__lus__flare__projects__Aurora_deployment__dvooturi__pretrain_parallel_models__OLMo-1B-hf_olmoemix0924_om_omcached_bf16_n256xg12_gbs3072-bs1-cs2048-gas1_dp3072-pp1-ep1-tp1_omopt-shard_final__hf_checkpoints__main/ /lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models/OLMoE-1B-7B-0924_olmoemix0924_om_omcached_gral_fastmoe-mmlp_bf16_n256xg12_gbs3072-bs1-cs2048-gas1_dp3072-pp1-ep1-tp1_omopt-shard_final/results/main/__lus__flare__projects__Aurora_deployment__dvooturi__pretrain_parallel_models__OLMoE-1B-7B-0924_olmoemix0924_om_omcached_gral_fastmoe-mmlp_bf16_n256xg12_gbs3072-bs1-cs2048-gas1_dp3072-pp1-ep1-tp1_omopt-shard_final__hf_checkpoints__main/ /lus/flare/projects/Aurora_deployment/dvooturi/hf_evals/allenai__OLMoE-1B-7B-0924/main/__lus__flare__projects__Aurora_deployment__dvooturi__hf_checkpoints__allenai__OLMoE-1B-7B-0924__main/