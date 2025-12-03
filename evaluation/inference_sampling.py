import functools
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval import run_eval, load_benchmark
import torch, jsonargparse

"""

This script generates samples from a causal language model using specified sampling parameters
and evaluates the outputs on a benchmark dataset.
Script to run sampling inference for a given config (task, language model, and sampling parameters)
and should output a csv file to 'results/sampling-[task]-[model-name]'

Can use to compare run_name="sampling", "greedy", "beamsearch" in eval.py

"""

@torch.inference_mode()
def sampling_forward(
        inputs,
        model,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        stopping_criteria,
):
    gen_kwargs = {
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "stopping_criteria": stopping_criteria,
    }

    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if top_k is not None:
        gen_kwargs["top_k"] = top_k

    output_ids = model.generate(
        **inputs,
        **gen_kwargs,
    )

    return output_ids[0][inputs.input_ids.shape[-1]:]


if __name__ == "__main__":

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("-c", action=jsonargparse.ActionConfigFile)
    parser.add_argument("-mp", type=str, required=True) # model path
    parser.add_argument("-mi", type=str, required=True) # model id
    parser.add_argument("-bn", type=str, default="gsm8k") # bench name
    parser.add_argument("-mt", type=int, default=320) # max tokens
    parser.add_argument("-ntt", type=int, default=5) # num trials
    parser.add_argument("-dt", type=str, default="bfloat16") # torch dtype name e.g float16, bfloat16, float32
    parser.add_argument("-d", type=str, default="cpu") # device
    parser.add_argument("-t", type=float, default=0.7) # temperature
    parser.add_argument("-tp", type=float, default=None) # top p
    parser.add_argument("-tk", type=int, default=None) # top k
    parser.add_argument("-rn", type=str, default="sampling") # run name
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.mp,
        dtype=getattr(torch, args.dt),
        device_map=args.d,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.mp)
    benchmark_data, stopping_criteria = load_benchmark(args.bn)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    forward_func = functools.partial(
        sampling_forward,
        model=model,
        max_new_tokens=args.mt,
        temperature=args.t,
        top_p=args.tp,
        top_k=args.tk,
        stopping_criteria=stopping_criteria,
    )

    run_eval(
        model_id=args.mi,
        tokenizer=tokenizer,
        forward_func=forward_func,
        benchmark_name=args.bn,
        dataset=benchmark_data,
        num_trials=args.ntt,
        device=args.d,
        run_name=args.rn,
    )