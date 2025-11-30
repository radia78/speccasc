"""
Script to run testing for a given config (task, language model, and generation method)
and should output a csv file to 'results/[decoding-strategy]-[task]-[model-name]'
"""
from datasets import load_dataset, load_dataset_builder
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import GSM8K_PROMPT, WMT_DE_EN_PROMPT, CNN_DM_PROMPT
import pandas as pd
import time, tqdm, torch, os

# TODO: Add argument parsing on this line

def load_benchmark(
        benchmark_dataset, 
        shuffle_seed=10,
        sample_size=5,
    ):
    if benchmark_dataset == 'gsm8k':
        add_preamble = lambda sample: {'question': GSM8K_PROMPT.format(question=sample['question']), 'answer': sample['answer']}
        ds = load_dataset('openai/gsm8k', 'main', split='test').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)

        return ds['question']
    
    if benchmark_dataset == 'wmt_de_en':
        add_preamble = lambda sample: {'question': WMT_DE_EN_PROMPT.format(de=sample['de']), 'answer': sample['en']}
        ds = load_dataset('wmt/wmt19', 'de-en', split='test').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)
        
        return ds['question']
    
    if benchmark_dataset == 'cnn_dm':
        add_preamble = lambda sample: {'question': CNN_DM_PROMPT.format(article=sample['article']), 'answer': sample['highlights']}
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split='test').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)

        return ds['question']

# Load the preamble/prompt and experiment artifacts
def run_eval(
        model_id,
        tokenizer,
        forward_func,
        benchmark_name,
        num_trials,
        device,
        run_name,
):
    questions = load_benchmark(benchmark_name)
    responses = []
    wall_time = []
    for _ in range(num_trials):
        batch = next(iter(questions))
        for question in tqdm.tqdm(batch):
            token_input = tokenizer(
                question, 
                return_tensors='pt', 
                return_attention_mask=True,
            ).to(device)
            torch.cpu.synchronize()
            t0 = time.time()
            output = forward_func(token_input)
            t1 = time.time()
            torch.cpu.synchronize()
            wall_time.append(t1-t0)
            responses.append(output)
    
    decoded_strs = tokenizer.batch_decode(responses, skip_special_tokens=True)
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    if not os.path.exists('results'):
        os.makedirs(path)

    pd.DataFrame(
        {'response': decoded_strs, 'wall_time': wall_time}
    ).to_csv(
        os.path.join(path, f"{run_name}-{model_id}-{benchmark_name}.csv"), 
        index=False
    )