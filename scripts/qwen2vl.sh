MODEL=Qwen2-VL-7B-Instruct

python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq /home/fengsicheng/.iaq_cache/$MODEL-w4-g128.pt

python -m iaq.entry --model_path /home/fengsicheng/.cache/huggingface/hub/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --load_awq /home/fengsicheng/.iaq_cache/$MODEL-w4-g128.pt \
    --q_backend real --dump_quant /home/fengsicheng/.quant_cache/$MODEL-w4-g128-awq.pt
