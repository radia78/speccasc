"""
speculative decoding is expord via the
assistant_model argument to the generate function

This script will load the target model, loads a smaller assistant model, calls the function

"""


import functools
import torch
import jsonargparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import run_eval,load_benchmark


@torch.inference_mode()
def speculative_forward(
        inputs,
        model,
        assistant_model,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        num_assistant_tokens,
        num_assistant_tokens_schedule,
        assistant_confidence_threshold,
        stopping_criteria,
):
    gen_kwargs = {
        "assistant_model": assistant_model,
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "num_assistant_tokens": num_assistant_tokens,
        "num_assistant_tokens_schedule": num_assistant_tokens_schedule,
        "assistant_confidence_threshold": assistant_confidence_threshold,
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
    parser.add_argument("-amp", type=str, required=True) # assistant model path
    parser.add_argument("-mid", type=str, required=True) # model id
    parser.add_argument("-bn", type=str, default="gsm8k") # bench name
    parser.add_argument("-mt", type=int, default=320) # max tokens
    parser.add_argument("-ntt", type=int, default=5) # num trials
    parser.add_argument("-dt", type=str, default="bfloat16") # torch dtype name e.g float16, bfloat16, float32
    parser.add_argument("-d", type=str, default="cpu") # device
    parser.add_argument("-t", type=float, default=0.8) # temperature
    parser.add_argument("-tp", type=float, default=0.95) # top p
    parser.add_argument("-tk", type=int, default=50) # top k
    parser.add_argument("-nat", type=int, default=16) # num assistant tokens
    parser.add_argument("-nats", type=str, default="constant") # num assistant tokens schedule
    parser.add_argument("-act", type=float, default=0.8) # assistant confidence threshold
    parser.add_argument("-rn", type=str, default="speculative") # run name
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.mp,
        dtype=getattr(torch, args.dt),
        device_map=args.d,
    )

    assistant_model = AutoModelForCausalLM.from_pretrained(
        args.amp,
        dtype=getattr(torch, args.dt),
        device_map=args.d,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.mp)
    benchmark_data, stopping_criteria = load_benchmark(args.bn, tokenizer)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    forward_func = functools.partial(
        speculative_forward,
        model=model,
        assistant_model=assistant_model,
        max_new_tokens=args.mt,
        temperature=args.t,
        top_p=args.tp,
        top_k=args.tk,
        num_assistant_tokens=args.nat,
        num_assistant_tokens_schedule=args.nats,
        assistant_confidence_threshold=args.act,
        stopping_criteria=stopping_criteria,
    )

    run_eval(
        model_id=args.mid,
        tokenizer=tokenizer,
        forward_func=forward_func,
        benchmark_name=args.bn,
        dataset=benchmark_data,
        num_trials=args.ntt,
        device=args.d,
        run_name=args.rn,
    )


