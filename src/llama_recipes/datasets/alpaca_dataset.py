# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset

# a map containing keys and tuples

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):

        # ann is a json : loaded from a dataset file
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    # method to get a single element at certain index
    
    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        # get an element from ann which is a json
        ann = self.ann[index]

        # if something has input
        if ann.get("input", "") == "":
            # extract from the json and the json have keys that can be filled into the formatted string
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        # join the formatted prompt and the output which could be the label
        example = prompt + ann["output"]

        # encode the prompt string to tensor using the tokenizer
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )

        # encode the example joint stirng to a sequence of tokens
        example = self.tokenizer.encode(example)

        # put an end to the sequence of tokens
        example.append(self.tokenizer.eos_token_id)

        # convert the sequence of tokens to tensor
        example = torch.tensor(
            example, dtype=torch.int64
        )

        # pad the sequence to a specific length
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        # copy the example string
        labels = copy.deepcopy(example)

        # convert the prompt part into -1
        labels[: len(prompt)] = -1
        # get the indices - true/false where the element is 0
        example_mask = example.ge(0)
        # do thd same in labels
        label_mask = labels.ge(0)
        # convert the 0 indices to 0 in data
        example[~example_mask] = 0
        # convert the 0 indices to IGNORE_INDEX in labels
        labels[~label_mask] = IGNORE_INDEX
        
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
