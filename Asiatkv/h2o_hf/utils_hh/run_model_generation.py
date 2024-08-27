import copy
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from modify_llama import convert_kvcache_llama_heavy_recent

model_path = "/dataset/crosspipe/OriginModel/Llama-2-7b-chat-hf/"

prompt_text = 'In a small, bustling cafe nestled in the heart of a vibrant city, a serendipitous event unfolded, leaving a lasting impression on all who witnessed it. As the patrons sat sipping their coffees and engaging in animated conversations, a talented street musician entered the cafe, carrying a weathered guitar and radiating an aura of creativity.'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

######## Generate with Full Cache
model = AutoModelForCausalLM.from_pretrained(model_path)
model.half().eval().cuda()

# input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(model.device)
input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

generate_ids = model.generate(input_ids, max_new_tokens=64)
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("################## Generated Context with Full Cache ###################")


print(result)
print("loading checkpoint")
# checkpoint = copy.deepcopy(model.state_dict())
# config = AutoConfig.from_pretrained(model_path)
# config.heavy_ratio = 0.1
# config.recent_ratio = 0.1
# model = convert_kvcache_llama_heavy_recent(model, config)
# checkpoint = copy.deepcopy(model.state_dict())
# model.load_state_dict(checkpoint)
# generate_ids_hh = model.generate(input_ids, max_new_tokens=64)
# result_hh = tokenizer.batch_decode(generate_ids_hh, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print("################## Generated Context with Heavy Hitter Oracle ###################")
# print(result_hh)