import argparse
import subprocess
import os
import itertools

def run(cmd: list):
    print("\n=== Running: " + " ".join(str(c) for c in cmd) + "\n")
    subprocess.run(cmd, check=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run_sampling(model, model_id, bench, device, dtype, max_tokens, num_trials, temp, top_p, top_k):
    script = "evaluation/inference_sampling.py"
    run_name = f"sampling_t{temp}_p{top_p}_k{top_k}"
    cmd = [
        "python3", script,
        "-mp", model,
        "-mi", model_id,
        "-bn", bench,
        "-d", device,
        "-dt", dtype,
        "-mt", str(max_tokens),
        "-ntt", str(num_trials),
        "-t", str(temp),
        "-tp", str(top_p),
        "-tk", str(top_k),
        "-rn", run_name
    ]
    run(cmd)


def run_speculative(
    model,
    assistant,
    model_id,
    bench,
    device,
    dtype,
    max_tokens,
    num_trials,
    num_assist,
    conf_thresh,
    schedule,
    temp,
    top_p,
    top_k,
):
    script = "evaluation/inference_speculative.py"
    run_name = (
        f"spec_nt{num_assist}_th{conf_thresh}_{schedule}_"
        f"t{temp}_p{top_p}_k{top_k}"
    )

    cmd = [
        "python3", script,
        "-mp", model,
        "-amp", assistant,
        "-mid", model_id,
        "-bn", bench,
        "-d", device,
        "-dt", dtype,
        "-mt", str(max_tokens),
        "-ntt", str(num_trials),
        "-nat", str(num_assist),
        "-act", str(conf_thresh),
        "-nats", schedule,
        "-t", str(temp),
        "-tp", str(top_p),
        "-tk", str(top_k),
        "-rn", run_name
    ]
    run(cmd)

def run_spec_casc(
    model, assistant, model_id, bench, device, dtype, max_tokens, num_trials,
    temp, top_p, top_k, alpha, deferral, num_assist, schedule, conf_thresh
):
    script = "evaluation/inference_speculative_cascades.py"
    run_name = f"spec_casc_a{alpha}_df{deferral}_t{temp}_p{top_p}_k{top_k}"

    cmd = [
        "python3", script,
        "-mp", model,
        "-amp", assistant,
        "-mid", model_id,
        "-bn", bench,
        "-d", device,
        "-dt", dtype,
        "-mt", str(max_tokens),
        "-ntt", str(num_trials),
        "-t", str(temp),
        "-tp", str(top_p),
        "-tk", str(top_k),
        "-alpha", str(alpha),
        "-df", deferral if deferral else "none",
        "-nat", str(num_assist),
        "-nats", schedule,
        "-act", str(conf_thresh),
        "-rn", run_name,
    ]
    run(cmd)


def run_spec_ensemble(
    model, model_id, bench, device, dtype, max_tokens, num_trials,
    temp, top_p, top_k, epsilon, beta, num_assist, schedule, conf_thresh
):
    script = "evaluation/inference_speculative_ensemble.py"
    run_name = f"spec_ens_eps{epsilon}_b{beta}_t{temp}_p{top_p}_k{top_k}"

    cmd = [
        "python3", script,
        "-mp", model,
        "-mid", model_id,
        "-bn", bench,
        "-d", device,
        "-dt", dtype,
        "-mt", str(max_tokens),
        "-ntt", str(num_trials),
        "-t", str(temp),
        "-tp", str(top_p),
        "-tk", str(top_k),
        "-nat", str(num_assist),
        "-nats", schedule,
        "-act", str(conf_thresh),
        "-eps", str(epsilon),
        "-beta", str(beta),
        "-rn", run_name
    ]
    run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--bench", type=str, default="gsm8k")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--num-trials", type=int, default=3)

    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--assistants", nargs="+", default=[])

    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["sampling", "speculative", "cascades", "ensemble", "all"],
        help="Which experiment to run.",
    )

    args = parser.parse_args()

    if len(args.assistants) == 1 and len(args.models) > 1:
        args.assistants = args.assistants * len(args.models)

    assert len(args.assistants) in (0, len(args.models))

    temperatures = [0.7, 1.0]
    top_ps = [0.9, 0.95]
    top_ks = [20, 50]

    speculative_nt = [5, 10, 20]
    speculative_th = [0.3, 0.4]
    speculative_schedules = ["constant", "heuristic"]

    ensure_dir("results")

    def do_sampling(model, model_id, assistant):
        for temp, p, k in itertools.product(temperatures, top_ps, top_ks):
            run_sampling(
                model=model, model_id=model_id, bench=args.bench,
                device=args.device, dtype=args.dtype,
                max_tokens=args.max_tokens, num_trials=args.num_trials,
                temp=temp, top_p=p, top_k=k
            )

    def do_speculative(model, model_id, assistant):
        if not assistant:
            print("Speculative requires an assistant model. Skipping.")
            return
        for nt, th, sched, temp, p, k in itertools.product(
            speculative_nt, speculative_th, speculative_schedules,
            temperatures, top_ps, top_ks
        ):
            run_speculative(
                model=model, assistant=assistant, model_id=model_id,
                bench=args.bench, device=args.device, dtype=args.dtype,
                max_tokens=args.max_tokens, num_trials=args.num_trials,
                num_assist=nt, conf_thresh=th, schedule=sched,
                temp=temp, top_p=p, top_k=k
            )

    def do_cascades(model, model_id, assistant):
        if not assistant:
            print("Cascades requires an assistant model. Skipping.")
            return
        for nt, temp in itertools.product([5, 10], temperatures):
            run_spec_casc(
                model=model, assistant=assistant, model_id=model_id,
                bench=args.bench, device=args.device, dtype=args.dtype,
                max_tokens=args.max_tokens, num_trials=args.num_trials,
                temp=temp, top_p=0.95, top_k=50,
                alpha=0.3, deferral=None, num_assist=nt,
                schedule="constant", conf_thresh=0.8,
            )

    def do_ensemble(model, model_id, assistant):
        for nt, temp in itertools.product([5, 10], temperatures):
            run_spec_ensemble(
                model=model, model_id=model_id, bench=args.bench,
                device=args.device, dtype=args.dtype,
                max_tokens=args.max_tokens, num_trials=args.num_trials,
                temp=temp, top_p=0.95, top_k=50,
                epsilon=0.25, beta=0.4,
                num_assist=nt, schedule="constant", conf_thresh=0.8,
            )

    DISPATCH = {
        "sampling": do_sampling,
        "speculative": do_speculative,
        "cascades": do_cascades,
        "ensemble": do_ensemble,
        "all": lambda m, mid, a: [
            do_sampling(m, mid, a),
            do_speculative(m, mid, a),
            do_cascades(m, mid, a),
            do_ensemble(m, mid, a),
        ],
    }

    # -----------------------------
    # Main Loop
    # -----------------------------
    for idx, model in enumerate(args.models):
        model_id = model.split("/")[-1]
        assistant = args.assistants[idx] if args.assistants else None

        print(f"\n==============================")
        print(f" Running model: {model_id}")
        print(f" Experiment: {args.experiment}")
        print(f"==============================")

        DISPATCH[args.experiment](model, model_id, assistant)

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--bench", type=str, default="gsm8k")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--num-trials", type=int, default=3)

    parser.add_argument("--models", nargs="+", required=True,
                        help="List of main models (e.g., google/gemma-2-2b-it)")
    parser.add_argument("--assistants", nargs="+", default=[],
                        help="List of assistant models (same length or broadcast 1-to-many)")

    args = parser.parse_args()

    if len(args.assistants) == 1 and len(args.models) > 1:
        args.assistants = args.assistants * len(args.models)

    assert len(args.assistants) in (0, len(args.models)), \
        "assistants must be empty or match #models or broadcast to it."

    temperatures = [0.7, 1.0]
    top_ps = [0.9, 0.95]
    top_ks = [20, 50]

    speculative_nt = [5, 10, 20]
    speculative_th = [0.3, 0.4]
    speculative_schedules = ["constant", "heuristic"]

    ensure_dir("experiments")

    for idx, model in enumerate(args.models):

        model_id = model.split("/")[-1]
        assistant = args.assistants[idx] if args.assistants else None

        print(f"\n==============================")
        print(f" Running model: {model_id}")
        print(f"==============================")

        # ----------------------------
        # 1. Sampling experiments
        # ----------------------------
        for temp, p, k in itertools.product(temperatures, top_ps, top_ks):
            run_sampling(
                model=model,
                model_id=model_id,
                bench=args.bench,
                device=args.device,
                dtype=args.dtype,
                max_tokens=args.max_tokens,
                num_trials=args.num_trials,
                temp=temp,
                top_p=p,
                top_k=k
            )

        # ----------------------------
        # 2. Speculative decoding (if assistant provided)
        # ----------------------------
        if assistant:
            for nt, th, sched, temp, p, k in itertools.product(
                speculative_nt,
                speculative_th,
                speculative_schedules,
                temperatures,
                top_ps,
                top_ks
            ):
                run_speculative(
                    model=model,
                    assistant=assistant,
                    model_id=model_id,
                    bench=args.bench,
                    device=args.device,
                    dtype=args.dtype,
                    max_tokens=args.max_tokens,
                    num_trials=args.num_trials,
                    num_assist=nt,
                    conf_thresh=th,
                    schedule=sched,
                    temp=temp,
                    top_p=p,
                    top_k=k
                )
        # ----------------------------
        # 3. Speculative Cascades
        # ----------------------------
        if assistant:
            for nt, temp in itertools.product([5, 10], temperatures):
                run_spec_casc(
                    model=model,
                    assistant=assistant,
                    model_id=model_id,
                    bench=args.bench,
                    device=args.device,
                    dtype=args.dtype,
                    max_tokens=args.max_tokens,
                    num_trials=args.num_trials,
                    temp=temp,
                    top_p=0.95,
                    top_k=50,
                    alpha=0.3,
                    deferral=None,
                    num_assist=nt,
                    schedule="constant",
                    conf_thresh=0.8,
                )

        # ----------------------------
        # 4. Speculative Ensemble
        # ----------------------------
        for nt, temp in itertools.product([5, 10], temperatures):
            run_spec_ensemble(
                model=model,
                model_id=model_id,
                bench=args.bench,
                device=args.device,
                dtype=args.dtype,
                max_tokens=args.max_tokens,
                num_trials=args.num_trials,
                temp=temp,
                top_p=0.95,
                top_k=50,
                epsilon=0.25,
                beta=0.4,
                num_assist=nt,
                schedule="constant",
                conf_thresh=0.8,
            )


## Commands

# Sampling only (no speculative):
# python3 run_experiments.py --models google/gemma-2-2b-it --experiment sampling --device cpu

# Sampling + Speculative:
# python3 run_experiments.py --models google/gemma-2-9b-it --assistants google/gemma-2-2b-it  --experiment speculative --device cpu

# Multiple models:
# python3 run_experiments.py --models google/gemma-2-1b-it google/gemma-2-4b-it --assistants google/gemma-2-1b-it google/gemma-2-2b-it --experiment speculative --device cpu

# speculative cascades:
# python3 run_experiments.py --models google/gemma-2-1b-it --assistants google/gemma-2-270m-it --experiment cascades --device cpu

# speculative ensemble:
# python3 run_experiments.py --models google/gemma-2-1b-it --experiment ensemble --device cuda

#sampling with config file
# python3 evaluation/inference_sampling.py -c configs/sampling_gemma_270m_gsm8k.yaml

# speculative with config file
# python3 evaluation/inference_speculative.py -c configs/speculative_gemma_270m_1b_gamma5.yaml

