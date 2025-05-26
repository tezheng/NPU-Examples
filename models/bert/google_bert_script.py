from collections import OrderedDict
from typing import Optional, Union

import torch
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm.auto import tqdm
from transformers import (
    AutoModelForNextSentencePrediction,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BertModel,
)
from transformers.modeling_outputs import ModelOutput

from olive.common.utils import format_data
from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry
from olive.hardware.accelerator import Device
from olive.model import ONNXModelHandler


class SimpleBert(torch.nn.Module):
    def __init__(self, model: BertModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embeddings = model.embeddings
        self.encoder = model.encoder
        self.pooler = model.pooler

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        **kwargs,
    ) -> ModelOutput:
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        sequence_output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
        )[0]

        pooled_output = self.pooler(sequence_output) if self.pooler else None

        return ModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )


def load_bert_nsp_model(model_name: str) -> torch.nn.Module:
    return AutoModelForNextSentencePrediction.from_pretrained(model_name).eval()


def load_bert_qa_model(model_name: str) -> torch.nn.Module:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()
    model.bert = SimpleBert(model.bert)
    return model


@Registry.register_post_process()
def bert_scl_post_process(outputs) -> torch.Tensor:
    """Post-processing for Sequence Classification task."""
    if isinstance(outputs, torch.Tensor):
        return outputs.argmax(dim=-1)
    if isinstance(outputs, (OrderedDict, dict)):
        if "logits" in outputs:
            return outputs["logits"].argmax(dim=-1)
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
    raise ValueError(f"Unsupported output type: {type(outputs)}")


@Registry.register_dataset()
def dataset_to_nsp_dataset(
    data_name: str,
    subset: str,
    split: str,
    input_cols: list[str],
    label_col: str,
    max_samples: Optional[int],
):
    from wikitext import create_nsp_dataset

    return create_nsp_dataset(
        dataset=data_name,
        subset=subset,
        split=split,
        sent_cols=input_cols,
        label_col=label_col,
        max_samples=max_samples,
    )


def create_4d_mask(
    mask: torch.Tensor,
    input_shape: Union[torch.Size, tuple[int, int]],
    masked_value: float = -50.0,
) -> torch.Tensor:
    # (batch_size, num_heads, seq_len, head_dim)
    batch_sz, seq_len = input_shape
    expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.float()
    return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)


@Registry.register_pre_process()
def tokenize_dataset_causal_mask(
    dataset,
    model_name: str,
    input_cols: list[str],
    label_col: str = "label",
    max_samples: Optional[int] = None,
    max_length: int = 512,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_inputs(sample, indices):
        encoded_input = tokenizer(
            *[sample[input_col] for input_col in input_cols],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        batch_sz = encoded_input.input_ids.shape[0]
        input_ids = encoded_input.input_ids
        attention_mask = create_4d_mask(
            encoded_input.attention_mask,
            (batch_sz, max_length),
        )
        token_type_ids = (
            encoded_input.token_type_ids
            if "token_type_ids" in encoded_input
            else torch.zeros(max_length).expand(batch_sz, -1)
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            **(
                {label_col: sample.get(label_col, indices)}
                if label_col is not None
                else {}
            ),
        }

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    tokenized_datasets = dataset.map(
        generate_inputs,
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
    )
    tokenized_datasets.set_format("torch", output_all_columns=True)

    return BaseDataset(tokenized_datasets, label_col)


def eval_squad(
    model: ONNXModelHandler,
    device: Device,
    execution_providers: str,
    dataset_config: dict[str, str],
    model_name: str,
    max_samples: Optional[int] = None,
) -> dict[str, Union[float, int]]:
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    sample_queue, result_queue = Queue(maxsize=10), Queue(maxsize=10)

    dataset = load_dataset(
        path=dataset_config["data_name"],
        split=dataset_config["split"],
    )
    if max_samples is not None:
        dataset = dataset.take(min(max_samples, len(dataset)))

    def data_thread_func():
        io_config = model.io_config
        input_ids_index = io_config["input_names"].index("input_ids")
        input_ids_shape = io_config["input_shapes"][input_ids_index]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for sample in tqdm(dataset, position=0, desc="Loading Data"):
            encoded_input = tokenizer(
                sample["question"],
                sample["context"],
                padding="max_length",
                max_length=input_ids_shape[1],
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            inputs = format_data(
                {
                    "input_ids": encoded_input.input_ids,
                    "attention_mask": create_4d_mask(
                        encoded_input.attention_mask, input_ids_shape
                    ),
                },
                io_config,
            )
            sample_queue.put((inputs, encoded_input.offset_mapping, sample))

        # Sentinel value to indicate end of data
        sample_queue.put((None, None, None))

    def inference_thread_func():
        sess = model.prepare_session(
            device=device,
            execution_providers=execution_providers,
        )
        with tqdm(total=len(dataset), position=1, desc="Inferencing") as pbar:
            while True:
                inputs, offset_mapping, sample = sample_queue.get()
                if inputs is None:
                    result_queue.put((None, None, None))
                    break  # Exit if sentinel value is received

                pred = model.run_session(session=sess, inputs=inputs)
                result_queue.put((pred, offset_mapping, sample))
                pbar.update(1)

    def post_process_thread_func():
        predictions, references = [], []
        with tqdm(total=len(dataset), position=2, desc="Post Processing") as pbar:
            while True:
                pred, offset_mapping, sample = result_queue.get()
                if pred is None:
                    break  # Exit if sentinel value is received

                start_index, end_index = pred[0].argmax(-1), pred[1].argmax(-1)
                answer_start, answer_end = (
                    offset_mapping[:, start_index, 0].squeeze(),
                    offset_mapping[:, end_index, 1].squeeze(),
                )
                predictions.append(
                    {
                        "id": sample["id"],
                        "prediction_text": sample["context"][answer_start:answer_end],
                    }
                )
                references.append(
                    {
                        "id": sample["id"],
                        "answers": {
                            "answer_start": sample["answers"]["answer_start"],
                            "text": sample["answers"]["text"],
                        },
                    }
                )
                pbar.update(1)

        return predictions, references

    with ThreadPoolExecutor(max_workers=3) as executor:
        data_future = executor.submit(data_thread_func)
        inference_future = executor.submit(inference_thread_func)
        post_process_future = executor.submit(post_process_thread_func)

        data_future.result()
        inference_future.result()
        predictions, references = post_process_future.result()

    results = load_metric("squad").compute(
        predictions=predictions,
        references=references,
    )

    return (
        {"f1": results["f1"], "exact_match": results["exact_match"]}
        if results
        else {"f1": float("nan"), "exact_match": float("nan")}
    )
