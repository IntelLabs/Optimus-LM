import os

from huggingface_hub import list_repo_refs
from huggingface_hub import snapshot_download

hf_model_choice = "allenai/OLMoE-1B-7B-0924"
hf_checkpoints_dump_dir = "/lus/flare/projects/Aurora_deployment/dvooturi/hf_checkpoints/"
hf_checkpoints_dir = os.path.join(hf_checkpoints_dump_dir, hf_model_choice.replace("/","__"))

if not os.path.exists(hf_checkpoints_dir):
    os.makedirs(hf_checkpoints_dir, exist_ok=True)

# All checkpoints
out = list_repo_refs(hf_model_choice)
checkpoints = [b.name for b in out.branches]

# Filtered checkpoints
filtered_checkpoints = []
step_checkpoints = [cp for cp in checkpoints if cp.startswith("step")]
step_checkpoints = sorted(step_checkpoints, key=lambda x: int(x.split("-")[0][4:]))

for cp in step_checkpoints:
    step_number = int(cp.split("-")[0][4:])
    # if (step_number == 1000) or (step_number % 10000 == 0):
    if True:
        filtered_checkpoints.append(cp)
filtered_checkpoints = (["main"] + filtered_checkpoints) if "main" in checkpoints else filtered_checkpoints

with open("checkpoints_allenai.txt", "w") as f:
    for cp in filtered_checkpoints:
        f.write(f"{cp}\n")

# Download checkpoints
for revision in filtered_checkpoints:
    local_dir = os.path.join(hf_checkpoints_dir, revision)
    snapshot_download(
        repo_id=hf_model_choice,
        revision=revision,
        local_dir=local_dir
    )