# MobileNet

Quantize [MobileNet V2](https://huggingface.co/google/mobilenet_v2_1.4_224) with [Olive](https://github.com/microsoft/Olive) using [ONNX QDQ](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#onnx-quantization-representation-format).

Dependencies:

- [Olive](https://github.com/microsoft/Olive)
- [onnxruntime-qnn](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#pre-built-packages-windows-only)

## Set up Environment with Conda and Poetry

```bash
# create a new conda environment with environment.yml
conda env create -f environment.yml -n {env_name}
conda activate {env_name}

# install dependencies with poetry
cd /path/to/repo/models/icl/mobilenet
poetry install
```

### Login Hugging Face

```bash
poetry run huggingface-cli login
```

## Quantize MobileNet V2

### Run the whole workflow

```bash
cd /path/to/repo/models/icl/mobilenet

# or `python . --config workflow.json`
olive run --config workflow.json
```

### Debugging

Set breakpoints in `__main__.py`, select `Python Debugger: MobileNetV2` in the debug panel then press `F5`.

> Debug Configuration
>
> ```json
> {
>   "name": "Python Debugger: MobileNetV2",
>   "type": "debugpy",
>   "request": "launch",
>   "program": "${workspaceFolder}/models/icl/mobilenet",
>   "cwd": "${workspaceFolder}/models/icl/mobilenet",
>   "justMyCode": false,
>   "args": ["--config", "workflow.json"],
>   "console": "integratedTerminal"
> }
> ```

## TODO

- [] Packaging
- [] Support QNN profiling
- [] Enable `evaluate_input_model` with multiple `accelerator`
- [] Generate EPContext model, aka "JIT"
- [] Resume from checkpoint
- [] Adopt `IsolatedORTSystem`
- [] Implement ImagenetDataset
- [] Add NPU support in `OrtSessionParamsTuning`
- [] auto-opt
