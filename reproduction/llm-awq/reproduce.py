import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from awq.quantize.quantizer import real_quantize_model_weight
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tinychat.demo import gen_params, stream_output
from tinychat.stream_generators import StreamGenerator
from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp
from tinychat.utils.prompt_templates import get_prompter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


model_path = "/data/llm/checkpoints/vicuna-hf/vicuna-7b"
load_quant_path = "/data/llm/checkpoints/vicuna-hf/vicuna-7b-awq-w4g128.pt"


config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16)
q_config = {"zero_point": True, "q_group_size": 128}
real_quantize_model_weight(
    model, w_bit=4, q_config=q_config, init_only=True)

model = load_checkpoint_and_dispatch(
    model, load_quant_path,
    device_map="auto",
    no_split_module_classes=["LlamaDecoderLayer"]
)

make_quant_attn(model, "cuda:4")
make_quant_norm(model)
make_fused_mlp(model)


model_prompter = get_prompter("llama", model_path)
stream_generator = StreamGenerator
count = 0
while True:
    # Get input from the user
    input_prompt = input("USER: ")
    if input_prompt == "":
        print("EXIT...")
        break
    model_prompter.insert_prompt(input_prompt)
    output_stream = stream_generator(model, tokenizer, model_prompter.model_input, gen_params, device="cuda:4")
    outputs = stream_output(output_stream)    
    model_prompter.update_template(outputs)
    count += 1