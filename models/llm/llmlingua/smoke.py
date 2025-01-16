from datasets import load_dataset

from prompt_compressor import PromptCompressor
# from model import QNPUModule

from time import time

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="cpu"
)


# model_path = "./outputs/microsoft/llmlingua-2-xlm-roberta-large-meetingbank/model/model.wu8au16.onnx"
# hook = QNPUModule(model_path)
# llm_lingua.model = hook

# Download the original prompt and dataset
dataset = load_dataset("huuuyeah/meetingbank", split="test")
context = dataset[0]["transcript"]

time_start = time()

# 2000 Compression
compressed_prompt = llm_lingua.compress_prompt(
    context,
    rate=0.33,
    force_tokens=["!", ".", "?", "\n"],
    drop_consecutive=True,
)
print(f"Time taken: {time() - time_start})")

print(f"Original Prompt:\n{context}")
print(f"Compressed Prompt:\n{compressed_prompt}")
# print(compressed_prompt)
