# LM Evaluation Harness Setup and Usage

## Installation

Clone the LM Evaluation Harness repository:

```bash
bash ../build.sh # Skip if conda env is already created
conda activate omxpu
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Usage

### Convert OM Checkpoint to HuggingFace Format

```bash
export PYTHONPATH=$PYTHONPATH:$PWD  (From repo main directory)
python convert_olmo_om_checkpoint_to_hf.py --om_checkpoint_dir <path_to_om_checkpoint_dir>
python convert_olmoe_om_checkpoint_to_hf.py --model_choice allenai/OLMoE-1B-7B-0924--SE1.125 --om_checkpoint_dir <path_to_om_checkpoint_dir>
python convert_olmoe_ppep_om_checkpoint_to_hf.py --model_choice allenai/OLMoE-1B-7B-0924--SA3-SH1.5-SE2.25-SC0.5-SI2-L36 --om_checkpoint_dir <path_to_om_checkpoint_dir> --use_init_empty_weights

```

### Run Evaluation

```bash
bash simple_eval.sh
```

## Notes

- Replace `<path_to_om_checkpoint_dir>` with the actual path to your OM checkpoint directory
- Ensure you have the necessary dependencies installed before running the evaluation