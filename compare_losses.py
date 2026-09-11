import re
import os
import sys
import math
import time
import json
import subprocess

import numpy as np
import matplotlib.pyplot as plt

info = []

# Step  103 / 23493657, Loss :  6.965205, Grad norm : 0.955142, LR : 0.00004742, Time : 0.7239 sec, DL time : 0.0003 sec, Tokens Processed : 10223616, Tokens/sec : 2829, Timestamp : 1753767088.0519562, Tflops : 47.05, ETA : 4724.25 hr, Local Loss :  7.485284, Local Grad norm :  2.765625, Rank : 0 (0,0)


keys = ["step_id", "loss_value", "grad_norm", "lr", "step_time", "tokens_processed", "tokens_per_sec", "time_stamp", "tflops", "world_size"]

global_info = []

def gather_data_from_log(log_file_path):
    """
    Extracts step data from the log file.
    """
    log_step_data = {key: [] for key in keys}
    
    # Get world_size
    world_size = 1
    fh = open(log_file_path, 'r')
    for line in fh:
        if "PMI_SIZE" in line:
            world_size = int(re.search(r'PMI_SIZE\s*=\s*(\d+)', line).group(1))
            break
    fh.close()
    
    with open(log_file_path, 'r') as file:
        for line in file:
            if "PMI_SIZE" in line:
                world_size = int(re.search(r'PMI_SIZE\s*=\s*(\d+)', line).group(1))
                log_step_data["world_size"].append(world_size)
            match = re.search(r'Step\s*(\d+).*Loss\s*:\s*(\d*\.\d+).*Grad norm\s*:\s*([+-]?\d*\.\d*).*LR\s*:\s*(\d*\.\d+).*Time\s*:\s*(\d*\.\d+).*Tokens Processed\s*:\s*(\d+).*Tokens/sec\s*:\s*(\d+)', line)
            # match_ts = re.search(r'Step\s*(\d+).*Loss\s*:\s*(\d*\.\d+).*LR\s*:\s*(\d*\.\d+).*Time\s*:\s*(\d*\.\d+).*Tokens Processed\s*:\s*(\d+).*Tokens/sec\s*:\s*(\d+).*Timestamp\s*:\s*(\d+\.\d+)', line)
            if match:
                step_id = int(match.group(1))
                loss_value = float(match.group(2))
                grad_norm = float(match.group(3))
                lr = float(match.group(4))
                step_time = float(match.group(5))
                tokens_processed = int(match.group(6))
                tokens_per_sec = int(match.group(7))
                time_stamp = None if "Timestamp" not in line else float(re.search(r'Timestamp\s*:\s*(\d+\.\d+)', line).group(1))
                tflops = 0 if "Tflops" not in line else float(re.search(r'Tflops\s*:\s*(\d+\.\d+)', line).group(1))

                log_step_data["step_id"].append(step_id)
                log_step_data["grad_norm"].append(grad_norm)
                log_step_data["loss_value"].append(loss_value)
                log_step_data["lr"].append(lr)
                log_step_data["step_time"].append(step_time)
                log_step_data["tokens_processed"].append(tokens_processed)
                log_step_data["tokens_per_sec"].append(tokens_per_sec)
                log_step_data["time_stamp"].append(time_stamp)
                log_step_data["tflops"].append(tflops)
                log_step_data["world_size"].append(world_size)
    
    # print(log_step_data)

    sorted_log_step_data = {key: [] for key in keys}
    # Sort the data by step_id
    sorted_indices = np.argsort(log_step_data["step_id"])
    for key in keys:
        sorted_log_step_data[key] = [log_step_data[key][i] for i in sorted_indices]

    return sorted_log_step_data

# Function to extract step information from log files
def get_step_info_list(log_file_paths):
    local_info = []
    verbose = True
    all_logs_step_data = {key: [] for key in keys}
    for log_file_path in log_file_paths:
        job_id = log_file_path.split("/")[-1].split(".")[0].split("_")[-1]
        log_step_data = gather_data_from_log(log_file_path)
        for key in all_logs_step_data:
            all_logs_step_data[key].extend(log_step_data[key])

        if len(log_step_data["step_id"]) > 0:
            if verbose:
                print(f"{len(log_step_data['step_id'])}, (s,e) : ({log_step_data['step_id'][0]}, {log_step_data['step_id'][-1]}) ({log_step_data['tokens_processed'][0]*1e-9:.0f}, {log_step_data['tokens_processed'][-1]*1e-9:.0f}), {log_file_path} , ./LLM-pretraining.e{job_id} , ./job_logs/LLM-pretraining.e{job_id} ")
            if len(log_step_data["step_id"]) > 1:
                tokens_processed = (log_step_data["tokens_processed"][-1] - log_step_data["tokens_processed"][0])
                if log_step_data["time_stamp"][-1] is None:
                    time_taken = 6 * 60 * 60
                else:
                    time_taken = (log_step_data["time_stamp"][-1] - log_step_data["time_stamp"][0])
                tps_per_gpu = (tokens_processed / time_taken)/log_step_data["world_size"][0]
                if verbose:
                    time_str = f"{int(time_taken//3600):02d}:{int((time_taken%3600)//60):02d}"
                    print(f"(Tokens (B), Time (hh:mm), Tokens/sec/GPU) : ({tokens_processed/1e9:.2f}, {time_str}, {tps_per_gpu:.0f})")
            if verbose:
                print(log_step_data["step_id"][-1], log_step_data["loss_value"][-1], log_step_data["grad_norm"][-1])
                # global_info.append(f"{log_step_data['step_id'][-1]}")
                local_info.append(f"{len(log_step_data['step_id'])}")

            info.append(f"{len(log_step_data['step_id'])} {log_file_path}")
        else:
            if verbose:
                print(f"0 steps in {log_file_path} , ./LLM-pretraining.e{job_id} , ./job_logs/LLM-pretraining.e{job_id} ")

    if verbose:
        print("*")
        if len(local_info) > 0:
            global_info.append(local_info)
    return all_logs_step_data

def filter_data(step_data, plot_kwargs):
    indices = []
    for i,step_id in enumerate(step_data["step_id"]):
        if plot_kwargs["start_step"] is not None and step_id < plot_kwargs["start_step"]:
            continue
        if plot_kwargs["end_step"] is not None and step_id > plot_kwargs["end_step"]:
            continue

        if plot_kwargs["start_tokens"] is not None and (step_data["tokens_processed"][i])*1e-9 < plot_kwargs["start_tokens"]:
            continue

        if plot_kwargs["end_tokens"] is not None and (step_data["tokens_processed"][i])*1e-9 > plot_kwargs["end_tokens"]:
            continue
        indices.append(i)
    
    filtered_step_data = {}
    for key in step_data:
        filtered_step_data[key] = [step_data[key][i] for i in indices]
    
    return filtered_step_data

# Plot functions
def plot_loss_vs_step(step_data_list, labels, save_path, kwargs):
    x_key = "step_id"
    y_key = "loss_value"
    ########## Common step code #####################################
    enable_marker = False
    marker_list = ["x", ".", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "D", "d"]
    marker_index = 0
    for step_data, label in zip(step_data_list, labels):
        if len(step_data[x_key]) == 0:
            continue
        marker = None if not enable_marker else marker_list[marker_index % len(marker_list)]
        plt.plot(step_data[x_key], step_data[y_key], label=label, marker=marker)
        marker_index += 1
    plt.grid()
    #################################################################

    if kwargs["max_loss"] is not None:
        plt.ylim(top=kwargs["max_loss"])
    if kwargs["min_loss"] is not None:
        plt.ylim(bottom=kwargs["min_loss"])

    plt.xlabel('Train step id')
    plt.ylabel('Loss')
    if not kwargs["skip_label"]:
        # plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 3), 1))
        # plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 5), 1))
    
        if kwargs["label_inside_plot"]:
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
        else:
            plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))

    plt.savefig(save_path, bbox_inches="tight")

    # Clear the current figure
    plt.clf()

def plot_loss_vs_tokens(step_data_list, labels, save_path, kwargs):
    plot_in_billions = True
    y_key = "loss_value"
    ##################################################################
    x_key = "tokens_processed"
    for step_data, label in zip(step_data_list, labels):
        if len(step_data[x_key]) == 0:
            continue
        x_values = step_data[x_key]
        if plot_in_billions:
            x_values = [x / 1e9 for x in x_values]
        y_values = step_data[y_key]

        plt.plot(x_values, y_values, label=label)
    plt.grid()
    ###################################################################

    if kwargs["max_loss"] is not None:
        plt.ylim(top=kwargs["max_loss"])
    if kwargs["min_loss"] is not None:
        plt.ylim(bottom=kwargs["min_loss"])

    plt.xlabel('Training Tokens' if not plot_in_billions else 'Training Tokens (B)')
    plt.ylabel('Loss')
    if not kwargs["skip_label"]:
        if kwargs["label_inside_plot"]:
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
        else:
            plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
        
    plt.savefig(save_path, bbox_inches="tight")

    # Clear the current figure
    plt.clf()

def plot_loss_vs_time(step_data_list, labels, save_path, kwargs):
    y_key = "loss_value"
    ########## Common step code #####################################
    x_key = "time_stamp"

    # Between jobs there will be wait times
    # Within a job what is the time spent on doing useful work.


    # Get min_val
    min_val = None
    for step_data, label in zip(step_data_list, labels):
        if len(step_data[x_key]) == 0:
            continue
        if None in step_data[x_key]:
            # There should not be None values in the data.
            break
        min_val = min(step_data[x_key]) if min_val is None else min(min_val, min(step_data[x_key]))
    
    if min_val is None:
        print("No valid timestamp data found in the logs. Skipping loss vs time plot.")
        return

    for step_data, label in zip(step_data_list, labels):
        timestamp_data = [(x - min_val)/(3600) for x in step_data[x_key]]  # Convert to hours
        if len(step_data[x_key]) == 0:
            continue
        good_time = (step_data[x_key][-1] - step_data[x_key][0]) / 3600  # Convert to hours
        few_steps_time = 0
        # few_steps_time = (step_data[x_key][32] - step_data[x_key][0]) / 60
        job_time_utilization = good_time/6*100
        # plt.plot(step_data[x_key], step_data[y_key], label=label)
        # plt.plot(timestamp_data, step_data[y_key], label=f"{label.split('_')[0]} ({good_time:.2f} hr, Utilization : {job_time_utilization:.2f}%) ")
        if label != "":
            plt.plot(timestamp_data, step_data[y_key], label=f"{label} ({good_time:.2f} hr)")
        else:
            plt.plot(timestamp_data, step_data[y_key], label=f"{label}")
        # plt.plot(timestamp_data, step_data[y_key], label=f"{label} ({good_time:.2f} hr) ({few_steps_time:.3f} min)")
    plt.grid()
    #################################################################
    
    if kwargs["max_loss"] is not None:
        plt.ylim(top=kwargs["max_loss"])
    if kwargs["min_loss"] is not None:
        plt.ylim(bottom=kwargs["min_loss"])

    # plt.title("Loss vs Wallclock Time (OLMo-1B Nodes=1024 TP=4)")

    plt.xlabel('Wall clock Time (hr)')
    plt.ylabel('Loss')
    if not kwargs["skip_label"]:
        # plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
        plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
    plt.savefig(save_path, bbox_inches="tight")

    # Clear the current figure
    plt.clf()

# Plot functions
def plot_gradnorm_vs_step(step_data_list, labels, save_path, kwargs):
    x_key = "step_id"
    y_key = "grad_norm"
    ########## Common step code #####################################
    for step_data, label in zip(step_data_list, labels):
        if len(step_data[x_key]) == 0:
            continue
        x_values = step_data[x_key]
        y_values = step_data[y_key]
        x_values_new = []
        y_values_new = []
        for i,y in enumerate(y_values):
            if y > 0:
                x_values_new.append(x_values[i])
                y_values_new.append(y_values[i])

        plt.plot(x_values_new, y_values_new, label=label)
    plt.grid()
    #################################################################
    plt.yscale('log')
    # plt.ylim(top=30)
    
    
    plt.xlabel('Step id')
    plt.ylabel('Grad norm')
    if not kwargs["skip_label"]:
        # plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
        plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
    plt.savefig(save_path, bbox_inches="tight")

    # Clear the current figure
    plt.clf()

def plot_gradnorm_vs_tokens(step_data_list, labels, save_path, kwargs):
    plot_in_billions = False
    y_key = "grad_norm"
    ########## Common step code #####################################
    x_key = "tokens_processed"
    for step_data, label in zip(step_data_list, labels):
        if len(step_data[x_key]) == 0:
            continue
        x_values = step_data[x_key]
        if plot_in_billions:
            x_values = [x / 1e9 for x in x_values]
        y_values = step_data[y_key]

        plt.plot(x_values, y_values, label=label)
    plt.grid()
    #################################################################
    plt.yscale('log')
    
    plt.xlabel('Tokens processed (B)' if plot_in_billions else 'Tokens processed')
    plt.ylabel('Grad norm')
    if not kwargs["skip_label"]:
        # plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
        plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
    plt.savefig(save_path, bbox_inches="tight")

    # Clear the current figure
    plt.clf()

def plot_tflops_vs_step(step_data_list, labels, save_path, kwargs):
    y_key = "tflops"
    averages = []
    marker = "."
    ########## Common step code #####################################
    x_key = "step_id"
    for step_data, label in zip(step_data_list, labels):
        if len(step_data[x_key]) == 0:
            continue
        x_values = step_data[x_key]
        y_values = step_data[y_key]

        avg_tflops = (sum(y_values[1:]) / len(y_values[1:])) if y_values else 0
        
        helper_str = f"(Avg: {avg_tflops:5.2f})"
        plt.plot(x_values, y_values, marker, label=f"{label} {helper_str}")

        # Custom
        averages.append(avg_tflops)

    plt.grid()
    #################################################################
    
    for avg_tflops in averages:
        plt.axhline(y=avg_tflops, color='red', linestyle='--')

    plt.xlabel('Step id')
    plt.ylabel('Tflops')   
    if not kwargs["skip_label"]:
        plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
        # plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 5), 1))
    plt.savefig(save_path, bbox_inches="tight")

    # Clear the current figure
    plt.clf()


def plot_tps_vs_step(step_data_list, labels, save_path, kwargs):
    y_key = "tokens_per_sec"
    averages = []
    marker = "."
    ########## Common step code #####################################
    x_key = "step_id"
    for step_data, label in zip(step_data_list, labels):
        if len(step_data[x_key]) == 0:
            continue
        x_values = step_data[x_key]
        y_values = step_data[y_key]

        avg_tps = 0
        if len(y_values) > 1:
            if step_data["time_stamp"][-1] != None and step_data["time_stamp"][0] != None:
                run_tokens = step_data["tokens_processed"][-1] - step_data["tokens_processed"][0]
                run_time = step_data["time_stamp"][-1] - step_data["time_stamp"][0]
                avg_tps = int((run_tokens / run_time) / step_data["world_size"][0])
            else:
                avg_tps = -1

        helper_str = f"(Avg: {avg_tps})"
        """
        if "TP" in label:
            tokens = label.split("_")
            tp_token = [token for token in tokens if "TP" in token][0]
            tp = int(tp_token.split("-")[1].replace("TP",""))
            helper_str += f" {avg_tps * tp}"
        """
        plt.plot(x_values, y_values, marker, label=f"{label} {helper_str}")

        # Custom
        averages.append(avg_tps)
    plt.grid()
    #################################################################
    
    for avg_tps in averages:
        plt.axhline(y=avg_tps, color='red', linestyle='--')
    
    plt.xlabel('Step id')
    plt.ylabel('Tokens/sec/GPU')   
    if not kwargs["skip_label"]:
        # plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 3), 1))
        plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
    plt.savefig(save_path, bbox_inches="tight")

    # Clear the current figure
    plt.clf()
    
def plot_lr_vs_step(step_data_list, labels, save_path, kwargs):
    y_key = "lr"
    ########## Common step code #####################################
    x_key = "step_id"
    for step_data, label in zip(step_data_list, labels):
        if len(step_data[x_key]) == 0:
            continue
        plt.plot(step_data[x_key], step_data[y_key], label=label)
    plt.grid()
    #################################################################
    
    plt.ylim(bottom=0)  # Set the lower y-limit to 0
    plt.xlabel('Step id')
    plt.ylabel('LR')
    if not kwargs["skip_label"]:
        # plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
        plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), borderaxespad=0., ncol=max(math.ceil(len(step_data_list) / 12), 1))
    plt.savefig(save_path, bbox_inches="tight")

    # Clear the current figure
    plt.clf()

# Main function
if __name__ == "__main__":
    # There are two cases
    # 1. Single experiment, multiple runs are plotted separately.
    # 2. Multiple experiments, All runs in the experiment are plotted as a single plot

    import argparse

    parser = argparse.ArgumentParser(description='Plot training loss and other metrics.')
    parser.add_argument('--num_logs', type=int, default=None, help='Number of log files to consider from the last')
    parser.add_argument('--min_loss', type=float, default=None, help='Minimum loss value for y-axis')
    parser.add_argument('--max_loss', type=float, default=None, help='Maximum loss value for y-axis')
    parser.add_argument('--start_step', type=int, default=None, help='Starting step for plotting')
    parser.add_argument('--end_step', type=int, default=None, help='Ending step for plotting')
    parser.add_argument('--start_tokens', type=int, default=None, help='Starting tokens for plotting in billions')
    parser.add_argument('--end_tokens', type=int, default=None, help='Ending tokens for plotting in billions')
    parser.add_argument('--skip_label', type=int, default=None, help='Skip label in the plot legend')
    parser.add_argument("--label_inside_plot", type=int, default=None, help="Whether to include labels in the plot or not")
    parser.add_argument('--plot_dir', type=str, default=".", help='Directory to save plots')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the config file')
    parser.add_argument('--analysis_choice', type=int, default=None, help='Choice of analysis from config file')
    args = parser.parse_args()

    exp_dump_dir = "."
    # exp_dump_dir = "/lus/flare/projects/Intel-Aurora/dvooturi/"
    plot_name = "training"

    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            json_config = json.load(f)
            # Config
            args.analysis_choice = json_config['analysis_choice'] if args.analysis_choice is None else args.analysis_choice
            config = json_config[f"analysis{args.analysis_choice}"]

            # Arguments
            args.min_loss = config.get("min_loss", None) if args.min_loss is None else args.min_loss
            args.max_loss = config.get("max_loss", None) if args.max_loss is None else args.max_loss
            args.start_step = config.get("start_step", None) if args.start_step is None else args.start_step
            args.end_step = config.get("end_step", None) if args.end_step is None else args.end_step
            args.start_tokens = config.get("start_tokens", None) if args.start_tokens is None else args.start_tokens
            args.end_tokens = config.get("end_tokens", None) if args.end_tokens is None else args.end_tokens
            args.num_logs = config.get("num_logs", None) if args.num_logs is None else args.num_logs
            args.skip_label = config.get("skip_label", False) if args.skip_label is None else (args.skip_label == 1)
            args.label_inside_plot = config.get("label_inside_plot", False) if args.label_inside_plot is None else (args.label_inside_plot == 1)
            args.plot_dir = config.get("plot_dir", args.plot_dir)

            exp_group_id = config.get("exp_group_id", "")
            exp_info_list = config.get(f"exp_info_list{exp_group_id}")

            if len(exp_info_list) == 0:
                # Take all folders that end with _temp and plot
                dump_dir = os.path.join(exp_dump_dir, f"pretrain_parallel_models")
                # Get all directories ending with _temp and sort by their timestamps (modification time)
                temp_dirs = [
                    f for f in os.listdir(dump_dir)
                    if f.endswith("_temp") and os.path.isdir(os.path.join(dump_dir, f))
                ]
                temp_dirs_sorted = sorted(
                    temp_dirs,
                    key=lambda d: os.path.getmtime(os.path.join(dump_dir, d))
                )
                exp_info_list = [[f] for f in temp_dirs_sorted]
                for i,e in enumerate(exp_info_list):
                    print(f'["{e[0]}"]', end="")
                    if i != (len(exp_info_list)-1):
                        print(",", end="")
                    print()
                # print(exp_info_list)
                exit

        all_log_file_paths = []
        all_labels = []
        if len(exp_info_list) > 1:
            for exp_info in exp_info_list:
                exp_dir1 = os.path.join(exp_dump_dir, f"pretrain_parallel_models_archive", exp_info[0])
                exp_dir2 = os.path.join(exp_dump_dir, f"pretrain_parallel_models", exp_info[0])
                exp_dir = exp_dir1 if os.path.exists(exp_dir1) else exp_dir2

                log_file_paths = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith('.txt')]
                log_file_paths = sorted(log_file_paths)
                if len(exp_info) > 1:
                    label = exp_info[1] if exp_info[1] is not None else f"{exp_info[0]}"
                else:
                    exp_name = exp_info[0]
                    """
                    tokens = exp_name.split("_")
                    # print(tokens)
                    model_name, dtype = tokens[0], tokens[1]
                    num_nodes, gpus_per_node = int(tokens[2].split("x")[0][1:]), int(tokens[2].split("x")[1][1:])
                    num_gpus = num_nodes * gpus_per_node
                    context_size, gas = int(tokens[3].split("-")[0][1:]), int(tokens[3].split("-")[1][3:])
                    model_type = tokens[5]
                    tensor_parallelism = int(tokens[6].split("-")[0].replace("tp", ""))
                    optimizer_info = tokens[7]
                    data_parallelism = num_gpus // tensor_parallelism
                    
                    extra = "skipopt," if "skipoptstep" in exp_name else ""
                    extra += f"{dtype},{tokens[4].split('-')[1]}"
                    # label = f"{model_name}_{model_type}_N{num_nodes}G{gpus_per_node}_DP{data_parallelism}-TP{tensor_parallelism}_C{context_size}_GAS{gas}_{optimizer_info}_({extra})"
                    label = f"{model_name}_N{num_nodes}_DP{data_parallelism}-TP{tensor_parallelism}_C{context_size}_GAS{gas}_{optimizer_info}_({extra})"
                    # label = f"{model_name}_N{num_nodes}_DP{data_parallelism}-TP{tensor_parallelism}_C{context_size}_GAS{gas}_{optimizer_info}"
                    """

                    # # TEMP
                    label = exp_name

                all_log_file_paths.append(log_file_paths)
                all_labels.append(label)
        else:
            exp_info = exp_info_list[0]
            exp_dir1 = os.path.join(exp_dump_dir, f"pretrain_parallel_models_archive", exp_info[0])
            exp_dir2 = os.path.join(exp_dump_dir, f"pretrain_parallel_models", exp_info[0])
            exp_dir = exp_dir1 if os.path.exists(exp_dir1) else exp_dir2

            # info_tuple = [(os.path.join(exp_dir, f), f.split(".")[0].split("_")[1]) for f in os.listdir(exp_dir) if f.endswith('.txt')]
            info_tuple = [(os.path.join(exp_dir, f), f.split(".")[0].split("_")[1] + "_" + f.split(".")[-2].replace("gov_","")) for f in os.listdir(exp_dir) if f.endswith('.txt')]
            info_tuple = sorted(info_tuple, key=lambda x: x[1])
            log_file_paths, labels = zip(*info_tuple)
            log_file_paths, labels = list(log_file_paths), list(labels)

            all_log_file_paths = [[_]for _ in log_file_paths]
            all_labels = labels
    else:

        # Override
        # All logs in logs folder
        logs_dir, args.plot_dir = "./logs", "." # Default
        all_log_file_paths = [[_] for _ in sorted([os.path.join(logs_dir, file) for file in os.listdir(logs_dir) if file.endswith(".txt") ]) ]
        all_labels = [_[0].split("/")[-1].split(".txt")[0] for _ in all_log_file_paths]
        all_labels = [label.replace("log_","") for label in all_labels]

        # # Select only a few logs based on string
        # filter_key = "dp6-tp2"
        # all_log_file_paths = [_ for _ in all_log_file_paths if filter_key in _[0]]
        # all_labels = [_ for _ in all_labels if filter_key in _]


        print("All log file paths:", all_log_file_paths)
        print("All labels:", all_labels)

        # Filter
        # filter_keys = ["log_fp32_nl1_bs1_1stepgc_dp1_pt", "log_fp32_nl1_bs1_2stepgc_dp1_om-stage1", "log_fp32_nl1_bs1_2stepgc_dp1_tp2-d0-moe1_om-stage1"]
        # all_log_file_paths = [_ for _ in all_log_file_paths if all(key in _[0] for key in filter_keys)]

    if args.num_logs is not None:
        if len(exp_info_list) > 1:
            all_log_file_paths = [log_file_paths[-args.num_logs:] for log_file_paths in all_log_file_paths]
        else:
            all_log_file_paths = all_log_file_paths[-args.num_logs:]
            all_labels = labels[-args.num_logs:]

    # Get step data from log files
    step_data_list = [get_step_info_list(log_file_paths) for log_file_paths in all_log_file_paths]

    plot_kwargs = {
        "min_loss": args.min_loss,
        "max_loss": args.max_loss,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "start_tokens" : args.start_tokens,
        "end_tokens" : args.end_tokens,
        "skip_label": args.skip_label,
        "label_inside_plot": args.label_inside_plot
        }

    # Filter data based on plot_kwargs
    step_data_list = [filter_data(step_data, plot_kwargs) for step_data in step_data_list]

    if args.plot_dir != "." and not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    # Print global info
    print("Global Info:")
    if os.path.exists("info.txt"):
        prev_lines = open("info.txt", 'r').readlines()
        print([int(_.strip()) for _ in prev_lines])
    fh = open("info.txt", 'w')
    cur_lines = []
    for g in global_info:
        print_str = g[-1] + "\n"
        fh.write(print_str)
        cur_lines.append(print_str)
    fh.close()
    print([int(_.strip()) for _ in cur_lines])

    loss_vs_step_plot_path = os.path.join(args.plot_dir, f"{plot_name}_loss-step.png")
    plot_loss_vs_step(step_data_list, all_labels, loss_vs_step_plot_path, plot_kwargs)
    print("Loss vs step plot saved at ", loss_vs_step_plot_path)

    gradnorm_vs_step_plot_path = os.path.join(args.plot_dir, f"{plot_name}_gradnorm.png")
    plot_gradnorm_vs_step(step_data_list, all_labels, gradnorm_vs_step_plot_path, plot_kwargs)
    print("Grad norm vs step plot saved at ", gradnorm_vs_step_plot_path)

    loss_vs_tokens_plot_path = os.path.join(args.plot_dir, f"{plot_name}_loss-tokens.png")
    plot_loss_vs_tokens(step_data_list, all_labels, loss_vs_tokens_plot_path, plot_kwargs)
    print("Loss vs tokens plot saved at ", loss_vs_tokens_plot_path)

    gradnorm_vs_tokens_plot_path = os.path.join(args.plot_dir, f"{plot_name}_gradnorm-tokens.png")
    plot_gradnorm_vs_tokens(step_data_list, all_labels, gradnorm_vs_tokens_plot_path, plot_kwargs)
    print("Grad norm vs tokens plot saved at ", gradnorm_vs_tokens_plot_path)

    loss_vs_time_plot_path = os.path.join(args.plot_dir, f"{plot_name}_loss-time.png")
    plot_loss_vs_time(step_data_list, all_labels, os.path.join(args.plot_dir, f"{plot_name}_loss-time.png"), plot_kwargs)
    print("Loss vs time plot saved at ", loss_vs_time_plot_path)

    lr_vs_step_plot_path = os.path.join(args.plot_dir, f"{plot_name}_lr.png")
    plot_lr_vs_step(step_data_list, all_labels, lr_vs_step_plot_path, plot_kwargs)
    print("LR vs step plot saved at ", lr_vs_step_plot_path)
    
    # Performance related
    tps_vs_step_plot_path = os.path.join(args.plot_dir, f"{plot_name}_tps.png")
    plot_tps_vs_step(step_data_list, all_labels, tps_vs_step_plot_path, plot_kwargs)
    print("tps vs step plot", tps_vs_step_plot_path)

    tflops_vs_step_plot_path = os.path.join(args.plot_dir, f"{plot_name}_tflops.png")
    plot_tflops_vs_step(step_data_list, all_labels, tflops_vs_step_plot_path, plot_kwargs)
    print("Tflops vs step plot saved at ", tflops_vs_step_plot_path)