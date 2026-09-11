import os
import math
import json

from collections import OrderedDict
import matplotlib.pyplot as plt

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

# tasks="arc_easy,arc_challenge,hellaswag,piqa,boolq,sciq,winogrande,openbookqa,mmlu"

# pcl results
def get_data(results_dir, inner_prefix):
    checkpoints = os.listdir(results_dir)
    step_checkpoints = [cp for cp in checkpoints if cp.startswith("step")] # Filter steps
    step_checkpoints = sorted(step_checkpoints, key=lambda x: int(x.split("-")[0][4:]))

    data = OrderedDict()
    for cp in step_checkpoints:
        inner_result_dir = os.path.join(results_dir, cp, inner_prefix+cp)
        json_result_path = os.path.join(inner_result_dir, os.listdir(inner_result_dir)[-1])
        with open(json_result_path, 'r') as f:
            data[cp] = json.load(f)["results"]
    
    return data


data_list = []
label_list = []

### Dense(1e-5) 
# exp_dir = "/lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models/OLMo-1B-hf_om_bf16_n128xg12_gbs1536-bs1-cs4096-gas1_dp1536-pp1-ep1-tp1_omopt-shard-rs_final"
# data_list.append(get_data(os.path.join(exp_dir, "results"), os.path.join(exp_dir, "hf_checkpoints/").replace("/", "__")))
# label_list.append("Intel-Aurora/OLMo-1B (eps=1e-5)")

# ### Dense(1-8)
# exp_dir = "/lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models/OLMo-1B-hf_om_bf16_n128xg12_gbs1536-bs1-cs4096-gas1_dp1536-pp1-ep1-tp1_omopt-shard_final"
# data_list.append(get_data(os.path.join(exp_dir, "results"), os.path.join(exp_dir, "hf_checkpoints/").replace("/", "__")))
# label_list.append("Intel-Aurora/OLMo-1B")

# ### Dense(AllenAI)
# data_list.append(get_data("results_allenai__OLMo-1B-hf/", "__lus__flare__projects__Aurora_deployment__dvooturi__hf_checkpoints__allenai__OLMo-1B-hf__"))
# label_list.append("Allenai/OLMo-1B")

# benchmarks = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "boolq", "sciq", "winogrande", "openbookqa", "mmlu"]

### MoE (OLMoE-100B-A4B)
# exp_dir = "/lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models/OLMoE-1B-7B-0924--SA3-SH1.5-SE2.25-SC0.5-SI2-L36_om_omcached_gral_ac1_bf16_n256xg12_gbs6144-bs8-cs4096-gas1_dp64-pp4-ep12-tp1-ps1f1b-mbs1-rb1_omopt-pgshard-pgar_final"
# data_list.append(get_data(os.path.join(exp_dir, "results"), "__tmp__"))
# label_list.append("Intel-Aurora/OLMoE-100B-A4B")

### MoE (OLMoE-1B-7B--SE1.125)
# exp_dir = "/lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models/OLMoE-1B-7B-0924--SE1.125_om_gral_bf16_n128xg12_gbs1536-bs1-cs4096-gas1_dp128-pp1-ep12-tp1_omopt-pgshard_final"
# data_list.append(get_data(os.path.join(exp_dir, "results"), os.path.join(exp_dir, "hf_checkpoints/").replace("/", "__")))
# label_list.append("Intel-Aurora/OLMoE-8B-A1B")

# ### Moe (AllenAI)
# data_list.append(get_data("results_allenai__OLMoE-1B-7B-0924/", "__lus__flare__projects__Aurora_deployment__dvooturi__hf_checkpoints__allenai-OLMoE-1B-7B-0924__"))
# label_list.append("Allenai/OLMoE-1B-7B-0924")


# exp_dir = "/lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models/OLMo-1B-hf_olmoemix0924_om_omcached_bf16_n256xg12_gbs3072-bs1-cs2048-gas1_dp3072-pp1-ep1-tp1_omopt-shard_final"
# data_list.append(get_data(os.path.join(exp_dir, "results"), os.path.join(exp_dir, "hf_checkpoints/").replace("/", "__")))
# label_list.append("Mula-1B")
# label_list.append("Intel-Aurora/OLMo-1B")

data_list.append(get_data("/lus/flare/projects/Aurora_deployment/dvooturi/hf_evals/allenai__OLMoE-1B-7B-0924/", "__lus__flare__projects__Aurora_deployment__dvooturi__hf_checkpoints__allenai__OLMoE-1B-7B-0924__"))
label_list.append("Allenai/OLMoE-1B-7B-0924")

exp_dir = "/lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models/OLMoE-1B-7B-0924_olmoemix0924_om_omcached_gral_fastmoe-mmlp_bf16_n256xg12_gbs3072-bs1-cs2048-gas1_dp3072-pp1-ep1-tp1_omopt-shard_final"
data_list.append(get_data(os.path.join(exp_dir, "results"), os.path.join(exp_dir, "hf_checkpoints/").replace("/", "__")))
label_list.append("Mula-7B-A1B")


# print(data_list[-1])


# benchmarks = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "mmlu"]
benchmarks = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "boolq", "sciq", "winogrande", "openbookqa", "mmlu"]


# Create 2x4 grid of subplots
fig, axes = plt.subplots(3, math.ceil(len(benchmarks)/3), figsize=(15, 8))  
axes = axes.flatten()
for i,benchmark in enumerate(benchmarks):
    axes[i].set_title(benchmark)

    for id, data in enumerate(data_list):
        label = label_list[id]
        x = [int(cp.split("-tokens")[1][:-1]) for cp in data]
        y = [data[cp][benchmark].get(benchmark_metric_map[benchmark]) for cp in data]
        # print(x, y)
        axes[i].plot(x, y, label=label)
        
    axes[i].set_xlabel('Tokens(B)')
    axes[i].set_ylabel('Accuracy')
    # axes[i].legend(loc='lower right')
    axes[i].grid(True)

# Get legend handles and labels from one axis
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave 5% space at bottom

# plt.tight_layout()
plt.savefig(f"results.png")
plt.clf()


# Generate individual plots for each benchmark
for benchmark in benchmarks:
    plt.figure(figsize=(8, 6))
    for id, data in enumerate(data_list):
        label = label_list[id]
        x = [int(cp.split("-tokens")[1][:-1]) for cp in data]
        y = [data[cp][benchmark].get(benchmark_metric_map[benchmark]) for cp in data]
        plt.plot(x, y, label=label)
    
    # plt.title(benchmark)
    plt.xlabel('Training Tokens (B)')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"plots/{benchmark}_results.png")
    plt.clf()

# Generate average plot across all benchmarks
plt.figure(figsize=(8, 6))
for id, data in enumerate(data_list):
    label = label_list[id]
    x = [int(cp.split("-tokens")[1][:-1]) for cp in data]
    y_all = []
    for cp in data:
        y_vals = []
        for benchmark in benchmarks:
            metric_value = data[cp][benchmark].get(benchmark_metric_map[benchmark])
            if metric_value is not None:
                y_vals.append(metric_value)
        y_all.append(np.mean(y_vals) if y_vals else 0)
    plt.plot(x, y_all, label=label)

# plt.title("Average Benchmark Performance")
plt.xlabel('Training Tokens (B)')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(f"plots/average_benchmark_results.png")
plt.clf()