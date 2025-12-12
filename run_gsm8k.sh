python3 evaluation/inference_sampling.py -c configs/gsm8k/sampling_gemma_270m.yaml -d cuda
python3 evaluation/inference_sampling.py -c configs/gsm8k/sampling_gemma_1B.yaml -d cuda
python3 evaluation/inference_sampling.py -c configs/gsm8k/sampling_qwen_0.6B.yaml -d cuda
python3 evaluation/inference_sampling.py -c configs/gsm8k/sampling_qwen_1.7B.yaml -d cuda

python3 evaluation/inference_speculative.py -c configs/gsm8k/sepculative_gemma_270m_1b_gamma5.yaml -d cuda
python3 evaluation/inference_speculative.py -c configs/gsm8k/sepculative_qwen_0.6b_1.7b_gamma5.yaml -d cuda

python3 evaluation/inference_speculative_cascades.py -c configs/gsm8k/sepculative_cascades_gemma_270m_1b_gamma5.yaml -d cuda
python3 evaluation/inference_speculative_cascades.py -c configs/gsm8k/sepculative_cascades_qwen_0.6b_1.7b_gamma5.yaml -d cuda

python3 evaluation/inference_speculative_ensemble.py -c configs/gsm8k/sepculative_ensemble_gemma_270m_1b_gamma5.yaml -d cuda
python3 evaluation/inference_speculative_ensemble.py -c configs/gsm8k/sepculative_ensemble_qwen_0.6b_1.7b_gamma5.yaml -d cuda