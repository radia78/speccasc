"""Generate answers with greedy decoding for a single model
Run the following: python3 --model_path='google/gemma3-1B-it'
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval import run_eval, load_benchmark
import torch, jsonargparse, functools

@torch.inference_mode()
def baseline_forward(
    inputs, 
    model, 
    max_new_tokens, 
    do_sample=False
):
    output_ids = model.generate(
        **inputs,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
    )
    return output_ids[0][inputs.input_ids.shape[-1]:]

if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', 
        action=jsonargparse.ActionConfigFile, 
        help='Path to config file'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--model-id',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--bench-name',
        type=str,
        default='gsm8k'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=320
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=3
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu'
    )

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=getattr(torch, args.dtype), 
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    forward_func = functools.partial(
        baseline_forward,
        model=model,
        max_new_tokens=args.max_tokens,
        do_sample=False
    )
    run_eval(
        model_id=args.model_id,
        tokenizer=tokenizer,
        forward_func=forward_func,
        benchmark_name=args.bench_name,
        num_trials=args.num_trials,
        device=args.device,
        run_name='greedy'
    )