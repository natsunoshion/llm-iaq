if [ ! -d "./record" ]; then
    mkdir -p ./record
fi


CUDA_VISIBLE_DEVICES=7 python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/models--unsloth--Llama-2-7B/snapshots/xx/ \
    --w_bit 4 --q_group_size 128 \
    --run_awq \
    --dump_awq /home/fengsicheng/.iaq_cache/llama-2-7b-w4-g128.pt > ./record/llama-2-7b-w4-g128-iaq.txt

CUDA_VISIBLE_DEVICES=7 python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/models--unsloth--Llama-2-7B/snapshots/xx/ \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq /home/fengsicheng/.iaq_cache/llama-2-7b-w4-g128.pt\
    --q_backend fake >> ./record/llama-2-7b-w4-g128-iaq.txt


# generate real quantized weights
CUDA_VISIBLE_DEVICES=7 python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/models--unsloth--Llama-2-7B/snapshots/xx/ \
    --w_bit 4 --q_group_size 128 \
    --load_awq /home/fengsicheng/.iaq_cache/llama-2-7b-w4-g128.pt\
    --q_backend real --dump_quant /home/fengsicheng/.quant_cache/llama-2-7b-w4-g128-iaq.pt >> ./record/llama-2-7b-w4-g128-iaq.txt


# load and evaluate the real quantized model (smaller gpu memory usage)
CUDA_VISIBLE_DEVICES=7 python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/models--unsloth--Llama-2-7B/snapshots/xx/ \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant /home/fengsicheng/.quant_cache/llama-2-7b-w4-g128-iaq-v2.pt >> ./record/llama-2-7b-w4-g128-iaq.txt