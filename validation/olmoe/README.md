# Run
```
export PYTHONPATH=$PYTHONPATH:$PWD

python validation/olmoe/test_OlmoeAttention.py
python validation/olmoe/test_OlmoeSparseMoeBlock.py
python validation/olmoe/test_OlmoeDecoderLayer.py
python validation/olmoe/test_OlmoeModel.py
python validation/olmoe/test_OlmoeForCausalLM.py

bash launch_dist.sh 2 1 python validation/olmoe/test_OlmoeExpertParallelSparseMoeBlock.py
```

