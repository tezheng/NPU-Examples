# BERT Quantization

## Prepare Environment

Create `conda` environment and install dependencies with `poetry`.

```
conda create -f environment.yml -n {conda_env_name}
conda activate {conda_env_name}
poetry install
```

## Quantize BERT

Quantize `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` with Olive.

```
olive run --config ./llmlingua2_bert_qdq_npu_meetingbank.json
```

## Evaluation

Evalute the quantized model on the test dataset.

```
python ./eval.py
```
