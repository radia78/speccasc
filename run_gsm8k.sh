python3 evaluation/inference_sampling.py -c configs/gsm8k/sampling_gemma_270m.yaml -d cuda
python3 evaluation/inference_sampling.py -c configs/gsm8k/sampling_gemma_1B.yaml -d cuda
python3 evaluation/inference_sampling.py -c configs/gsm8k/sampling_qwen_0.6B.yaml -d cuda
python3 evaluation/inference_sampling.py -c configs/gsm8k/sampling_qwen_1.7B.yaml -d cuda

python3 evaluation/inference_speculative.py -c configs/gsm8k/speculative_gemma_270m_1b_gamma5.yaml -d cuda
python3 evaluation/inference_speculative.py -c configs/gsm8k/speculative_qwen_0.6b_1.7b_gamma5.yaml -d cuda

python3 evaluation/inference_speculative_cascades.py -c configs/gsm8k/speculative_cascades_gemma_270m_1b_gamma5.yaml -d cuda
python3 evaluation/inference_speculative_cascades.py -c configs/gsm8k/speculative_cascades_qwen_0.6b_1.7b_gamma5.yaml -d cuda

python3 evaluation/inference_speculative_ensemble.py -c configs/gsm8k/speculative_ensemble_gemma_270m_gamma5.yaml -d cuda
python3 evaluation/inference_speculative_ensemble.py -c configs/gsm8k/speculative_ensemble_qwen_0.6b_gamma5.yaml -d cuda