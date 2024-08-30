
from ...ours.custom_llama import LlamaForCausalLM
from transformers import AutoTokenizer,AutoConfig
from modify_llama import convert_kvcache_llama_heavy_recent
import torch
import time

model_path = "/dataset/crosspipe/OriginModel/Llama-2-7b-chat-hf/"
prompt_text = 'In a small, bustling cafe nestled in the heart of a vibrant city, a serendipitous event unfolded, leaving a lasting impression on all who witnessed it. As the patrons sat sipping their coffees and engaging in animated conversations, a talented street musician entered the cafe, carrying a weathered guitar and radiating an aura of creativity.'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
######## Generate with Full Cache
model = LlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16)
# print(model.state_dict())
# checkpoint = copy.deepcopy(model.state_dict())
model.eval().to(torch.device("cuda:2"))

# input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(model.device)
input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(torch.device("cuda:2"))

generate_ids = model.generate(input_ids, max_new_tokens=64)
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("################## Generated Context with Full Cache ###################")
print(result)

# print("loading checkpoint")
# time.sleep(5)
# config = AutoConfig.from_pretrained(model_path)
# config.heavy_ratio = 0.1
# config.recent_ratio = 0.1
# model_new = convert_kvcache_llama_heavy_recent(model, config)
# model_new.load_state_dict(checkpoint)
# print(model_new.state_dict())
# model_new.eval().to(torch.device("cuda:2"))
# print("loading new model")
# generate_ids_hh = model.generate(input_ids, max_new_tokens=64)
# result_hh = tokenizer.batch_decode(generate_ids_hh, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print("################## Generated Context with Heavy Hitter Oracle ###################")
# print(result_hh)