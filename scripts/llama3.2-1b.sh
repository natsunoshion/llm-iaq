if [ ! -d "./record" ]; then
    mkdir -p ./record
fi


CUDA_VISIBLE_DEVICES=7 python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/1d05b8ce9cd75f6baca1ccebf9653626ac261438/ \
    --w_bit 4 --q_group_size 128 \
    --run_awq \
    --dump_awq /home/fengsicheng/.iaq_cache/llama3.2-1b-w4-g128.pt > ./record/llama3.2-1b-w4-g128-iaq.txt

CUDA_VISIBLE_DEVICES=7 python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/1d05b8ce9cd75f6baca1ccebf9653626ac261438/ \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq /home/fengsicheng/.iaq_cache/llama3.2-1b-w4-g128.pt\
    --q_backend fake >> ./record/llama3.2-1b-w4-g128-iaq.txt


# generate real quantized weights
CUDA_VISIBLE_DEVICES=7 python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/1d05b8ce9cd75f6baca1ccebf9653626ac261438/ \
    --w_bit 4 --q_group_size 128 \
    --load_awq /home/fengsicheng/.iaq_cache/llama3.2-1b-w4-g128.pt\
    --q_backend real --dump_quant /home/fengsicheng/.quant_cache/llama3.2-1b-w4-g128-iaq.pt >> ./record/llama3.2-1b-w4-g128-iaq.txt


# load and evaluate the real quantized model (smaller gpu memory usage)
CUDA_VISIBLE_DEVICES=7 python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/1d05b8ce9cd75f6baca1ccebf9653626ac261438/ \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant /home/fengsicheng/.quant_cache/llama3.2-1b-w4-g128-iaq-v2.pt >> ./record/llama3.2-1b-w4-g128-iaq.txt