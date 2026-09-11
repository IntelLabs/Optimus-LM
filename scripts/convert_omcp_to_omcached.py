import os

import torch

# # 8B-1B
# exp_name = "OLMoE-1B-7B-0924--SE1.125_om_gral_bf16_n128xg12_gbs1536-bs1-cs4096-gas1_dp128-pp1-ep12-tp1_omopt-pgshard_final"
# checkpoint = "step247000-tokens1554B"
# model_name = "allenai__OLMoE-1B-7B-0924--SE1.125_pp1-ep12-tp1"

# 100B-4B
exp_name = "OLMoE-1B-7B-0924--SA3-SH1.5-SE2.25-SC0.5-SI2-L36_om_omcached_gral_ac1_bf16_n256xg12_gbs6144-bs8-cs4096-gas1_dp64-pp4-ep12-tp1-ps1f1b-mbs1-rb1_omopt-pgshard-pgar_final"
checkpoint = "step61500-tokens1548B"
model_name = "allenai__OLMoE-1B-7B-0924--SA3-SH1.5-SE2.25-SC0.5-SI2-L36_pp4-ep12-tp1"

exp_dump_dir = "/lus/flare/projects/Aurora_deployment/dvooturi/pretrain_parallel_models"
om_cached_models_dir = "/lus/flare/projects/Intel-Aurora/dvooturi/parallel_models/"
exp_dir = os.path.join(exp_dump_dir, exp_name)
om_model_checkpoint_dir = os.path.join(exp_dir, "checkpoints", checkpoint)
om_model_cached_dir = os.path.join(om_cached_models_dir, model_name)

print(f"OM model checkpoint dir : {om_model_checkpoint_dir}")
print(f"OM model cached dir     : {om_model_cached_dir}")

print(f"cp {exp_dir}/*.txt {exp_dir.replace('Aurora_deployment','Intel-Aurora')}/")
exit(-1)

files = os.listdir(om_model_checkpoint_dir)
model_shards = [f for f in files if f.startswith("model_checkpoint_shard")]
model_shards = sorted(model_shards, key=lambda x: int(x.split("-")[-1].replace(".pth", "")))

# Create directory if it does not exists
if not os.path.exists(om_model_cached_dir):
    os.makedirs(om_model_cached_dir)

for model_shard in model_shards:
    print(f"Converting {model_shard}")
    checkpoint_model_shard_path = os.path.join(om_model_checkpoint_dir, model_shard)
    checkpoint_model_state_dict = torch.load(checkpoint_model_shard_path, map_location="cpu", weights_only=True)["state_dict"]

    cached_model_shard = model_shard.replace("_checkpoint_shard-", "_")
    cached_model_shard_path = os.path.join(om_model_cached_dir, cached_model_shard)
    torch.save(checkpoint_model_state_dict, cached_model_shard_path)

    

"""
for model_shard in model_shards:
    # print(f"Converting {model_shard}")
    shard_id = int(model_shard.split("-")[-1].replace(".pth", ""))
    pp_ind = shard_id // 12
    ep_ind = shard_id % 12
    # if ep_ind != 0:
    #     continue

    model_shard_path = os.path.join(om_model_checkpoint_dir, model_shard)
    model_state_dict = torch.load(model_shard_path, map_location="cpu")["state_dict"]

    # keys = [key for key in model_state_dict.keys() if "norm" in key]    
    keys = model_state_dict.keys()

    for key in keys:
        tensor = model_state_dict[key]
        global_key = key
        if "layers" in global_key:
            layer_ind = int(global_key.split(".")[2])
            new_layer_ind = layer_ind + pp_ind * (36 // 4)
            global_key = global_key.replace(f"layers.{layer_ind}.", f"layers.{new_layer_ind}.")
        
        if "experts" in global_key:
            expert_ind = int(global_key.split(".")[5])
            new_expert_ind = expert_ind + ep_ind * (144 // 12)
            global_key = global_key.replace(f"experts.{expert_ind}.", f"experts.{new_expert_ind}.")

        print(f"{global_key:50s}, {torch.min(tensor).item():.2e}, {torch.max(tensor).item():.2e}, {torch.mean(tensor).item():.2e}")


    # exit(-1)

    
"""