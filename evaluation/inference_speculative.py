"""
speculative decoding is expord via the
assistant_model argument to the generate function

This script will load the target model, loads a smaller assistant model, calls the function

"""



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
):
    gen_kwargs = {
        "assistant_model": assistant_model,
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "num_assistant_tokens": num_assistant_tokens,
        "num_assistant_tokens_schedule": num_assistant_tokens_schedule,
        "assistant_confidence_threshold": assistant_confidence_threshold,
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
    parser.add_argument("-am", type=str, required=True) # assistant model id
    parser.add_argument("-bn", type=str, default="gsm8k") # bench name
    parser.add_argument("-mt", type=int, default=320) # max tokens
    parser.add_argument("-ntt", type=int, default=5) # num trials
    parser.add_argument("-dt", type=str, default="bfloat16") # torch dtype name e.g float16, bfloat16, float32
    parser.add_argument("-d", type=str, default="cpu") # device
    parser.add_argument("-rn", type=str, default="speculative") # run name
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.mp,
        dtype=getattr(torch, args.dt),
        device_map=args.d,
    )
