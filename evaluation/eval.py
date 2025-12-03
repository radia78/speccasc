"""
Script to run testing for a given config (task, language model, and generation method)
and should output a csv file to 'results/[decoding-strategy]-[task]-[model-name]'
"""
from datasets import load_dataset
# Import the stop string to prevent endless generation
from prompts import (
    GSM8K_PROMPT, 
    Q_A_STOP_STRINGS,
    WMT_DE_EN_PROMPT,
    WMT_DE_EN_STOP_STRINGS, 
    CNN_DM_PROMPT, 
    CNN_DM_STOP_STRINGS,
    NATURAL_QA_PROMPT, 
    TRIVIA_QA_PROMPT, 
    WEBQUESTIONS_PROMPT, 
    SQUAD_2_PROMPT,
    MBPP_PROMPT,
    MBPP_STOP_STRINGS
)
from transformers import StopStringCriteria, StoppingCriteriaList
import pandas as pd
import time, tqdm, torch, os

# TODO: Add argument parsing on this line

def load_benchmark(
        benchmark_dataset, 
        tokenizer,
        shuffle_seed=10,
        sample_size=5,
    ):
    if benchmark_dataset == 'gsm8k':
        add_preamble = lambda sample: {'question': GSM8K_PROMPT.format(question=sample['question']), 'answer': sample['answer']}
        ds = load_dataset('openai/gsm8k', 'main', split='test').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)
        stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, Q_A_STOP_STRINGS)])

        return ds['question'], stopping_criteria
    
    if benchmark_dataset == 'wmt_de_en':
        add_preamble = lambda sample: {'question': WMT_DE_EN_PROMPT.format(de=sample['de']), 'answer': sample['en']}
        ds = load_dataset('wmt/wmt19', 'de-en', split='test').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)
        stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, WMT_DE_EN_STOP_STRINGS)])
        
        return ds['question'], stopping_criteria
    
    if benchmark_dataset == 'cnn_dm':
        add_preamble = lambda sample: {'question': CNN_DM_PROMPT.format(article=sample['article']), 'answer': sample['highlights']}
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split='test').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)
        stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, CNN_DM_STOP_STRINGS)])

        return ds['question'], stopping_criteria
    
    if benchmark_dataset == 'natural_qa':
        add_preamble = lambda sample: {'question': NATURAL_QA_PROMPT.format(query=sample['query']), 'answer': sample['answer']}
        ds = load_dataset("sentence-transformers/natural-questions", split='train').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)
        stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, Q_A_STOP_STRINGS)])

        return ds['question'], stopping_criteria
    
    if benchmark_dataset == 'trivia_qa':
        add_preamble = lambda sample: {'question': TRIVIA_QA_PROMPT.format(question=sample['question']), 'answer': sample['answer']['value']}
        ds = load_dataset("mandarjoshi/trivia_qa", "rc", split='test').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)
        stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, Q_A_STOP_STRINGS)])
    
    if benchmark_dataset == 'webquestions':
        add_preamble = lambda sample: {'question': WEBQUESTIONS_PROMPT.format(question=sample['question']), 'answer': sample['answers']}
        ds = load_dataset("stanfordnlp/web_questions", split='test').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)
        stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, Q_A_STOP_STRINGS)])

        return ds['question'], stopping_criteria
    
    if benchmark_dataset == 'squad_2':
        add_preamble = lambda sample: {'question': SQUAD_2_PROMPT.format(context=sample['context'], question=sample['question']), 'answer': sample['answers']['text']}
        ds = load_dataset("rajpurkar/squad_v2", split='validation').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)
        stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, Q_A_STOP_STRINGS)])

        return ds['question'], stopping_criteria

    if benchmark_dataset == "mbpp":
        add_preamble = lambda sample: {'question': MBPP_PROMPT.format(context=sample['context'], question=sample['question']), 'answer': sample['answers']['text']}
        ds = load_dataset("Muennighoff/mbpp", 'sanitized', split='test').shuffle(shuffle_seed).map(add_preamble).batch(batch_size=sample_size)
        stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, MBPP_STOP_STRINGS)])

        return ds['question'], stopping_criteria

# Load the preamble/prompt and experiment artifacts
def run_eval(
    model_id,
    tokenizer,
    forward_func,
    benchmark_name,
    dataset,
    num_trials,
    device,
    run_name,
):
    responses = []
    wall_time = []
    for _ in range(num_trials):
        batch = next(iter(dataset))
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