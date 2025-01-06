# MobileNet

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

- [] Enable `evaluate_input_model` with multiple `accelerator`
- [] Add JIT pass
- [] Resume from checkpoint
- [] Adopt `IsolatedORTSystem`
- [] Implement ImagenetDataset
- [] auto-opt
