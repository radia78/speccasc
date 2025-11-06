"""
Script to run testing for a given config (task, language model, and generation method)
and should output a csv file to 'results/[decoding-strategy]-[task]-[model-name]'
"""

from datasets import load_dataset, load_dataset_builder
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
import pandas as pd
import time
import json

# Test configuration
dsname = "openai/gsm8k"
small_model_name = None
large_model_name = "google/gemma-3-1B"
decoding_method = "greedy"
task = "math"
sample_size = 500
shuffle_seed = 10
max_len = 80

# Load the preamble/prompt and experiment artifacts
if task.split("-")[0].strip() == "math":
    preamble = "Solve the following problem: "
    ds = load_dataset(dsname, split='test', streaming=True).shuffle(seed=shuffle_seed).batch(sample_size)
    metric = load('exact_match')

# Load the model artifacts
if small_model_name is None:
    draft_model = AutoModelForCausalLM.from_pretrained(large_model_name)
    tokenizer = AutoTokenizer.from_pretrained(large_model_name)

else:
    draft_model = AutoModelForCausalLM.from_pretrained(small_model_name)
    verifier = AutoModelForCausalLM.from_pretrained(large_model_name)
    tokenizer = AutoTokenizer.from_pretrained(small_model_name)

# Lazy load the ds -  let the model process and if its finished processing time it - evaluate the score
for batch in ds:
    answers = []
    scores = []
    for sample in batch:
        question, _ = sample
        token_input = tokenizer.encode(preamble + question, return_tensors='pt')
        t0 = time.time()
        output = draft_model.generate(**token_input, max_length=max_len)
        t1 = time.time()
        answers.append(output[0])

    decoded_strs = tokenizer.batch_decode(answers, skip_special_tokens=True)
    metric