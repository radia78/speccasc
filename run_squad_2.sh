python3 evaluation/inference_sampling.py -c configs/squad_2/sampling_gemma_270m.yaml -d cuda
python3 evaluation/inference_sampling.py -c configs/squad_2/sampling_gemma_1B.yaml -d cuda
python3 evaluation/inference_sampling.py -c configs/squad_2/sampling_qwen_0.6B.yaml -d cuda
python3 evaluation/inference_sampling.py -c configs/squad_2/sampling_qwen_1.7B.yaml -d cuda

python3 evaluation/inference_speculative.py -c configs/squad_2/sepculative_gemma_270m_1b_gamma5.yaml -d cuda
python3 evaluation/inference_speculative.py -c configs/squad_2/sepculative_qwen_0.6b_1.7b_gamma5.yaml -d cuda

python3 evaluation/inference_speculative_cascades.py -c configs/squad_2/sepculative_cascades_gemma_270m_1b_gamma5.yaml -d cuda
python3 evaluation/inference_speculative_cascades.py -c configs/squad_2/sepculative_cascades_qwen_0.6b_1.7b_gamma5.yaml -d cuda

python3 evaluation/inference_speculative_ensemble.py -c configs/squad_2/sepculative_ensemble_gemma_270m_1b_gamma5.yaml -d cuda
python3 evaluation/inference_speculative_ensemble.py -c configs/squad_2/sepculative_ensemble_qwen_0.6b_1.7b_gamma5.yaml -d cuda