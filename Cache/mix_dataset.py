import random
import argparse
import numpy as np
import torch.cuda
import fire
import json
import os
import datetime
from tqdm import tqdm
from benchmark.longbench import LongBench
from model import Llama2
# , Falcon, Mpt
from cache import CacheEngine
from generation_engine import GenerationEngine, GenerationParameters
from benchmark.benchmark_base import DATASET_LIST, SCHEMA_FILE_DIRECTORY
# from benchmark.squad_v2 import SquadV2
# from benchmark.multi_news import MultiNews
# from benchmark.ms_marco_v1_1 import MSMarcoV1
from prompt import Prompt, read_file
BENCHMARK_PATH = "./benchmark"

# class MixedDataset:
#     def __init__(self, llm_config_path, dataset, enable_cache, use_cpu_for_inference=False):
#         with open("./config/dataset_maxlen.json", 'r') as f:
#             self.dataset_maxlen = json.load(f)
#         with open(llm_config_path, 'r') as f:
#             self.llm_config = json.load(f)
#         self.enable_cache = enable_cache
#         self.use_cpu_for_inference = use_cpu_for_inference
#         self.model_name = self.llm_config["name"]
#         if "Llama" in self.model_name:
#             self.model_name = "Llama"
#             self.lm_for_caching = Llama2(name=self.llm_config['name'], device_map="cuda:1")
#         else:
#             raise ValueError("Invalid model name")
        
#         if self.use_cpu_for_inference:
#             if "Llama" in self.model_name:
#                 self.lm = Llama2(name=self.llm_config['name'], device_map=None)
#         else:
#             self.lm = self.lm_for_caching

#         self.cache_engine = CacheEngine(self.llm_config.get("max_ctx_length", 4096), self.lm_for_caching,
#                                         target_device=self.lm.device, sharing_ratio=self.sharing_ratio,evicting_ratio=self.evicting_ratio)
#         self.gen_engine = GenerationEngine(self.lm)
#         self.preproc = [
#             # CompactSpaces(),
#             self.lm.get_formatter()
#         ]

#         self.parameter = GenerationParameters(
#             temperature=1.0,
#             repetition_penalty=1.0,
#             top_p=0.95,
#             top_k=-1,
#             max_new_tokens=self.dataset_maxlen[dataset],
#             stop_token_ids=self.lm.stop_token_ids,
#             stop_str=self.lm.stop_str
#         )

#         if dataset is None or dataset not in DATASET_LIST:
#             raise ValueError("Dataset name cannot be None, valid dataset names are: " + ", ".join(DATASET_LIST))

#         match dataset:
#             # here we get dataset name and initialize the dataset
#             case "narrativeqa":
#                 self.dataset = LongBench("narrativeqa")
            
#             case "mixed_dataset_all":
#                 self.dataset = LongBench("mixed_dataset_all")

#             case "mixed_dataset_5":
#                 self.dataset = LongBench("mixed_dataset_5")
        
#         print("initiating dataset")
        
#         self.dataset.init()

#         # create result of hitrate directory
#         self.hitrate_directory = os.path.join(BENCHMARK_PATH, "hitrate",
#                                              f"{self.model_name}-{self.dataset.dataset_name}")
#         if not os.path.exists(self.hitrate_directory):
#             os.makedirs(self.hitrate_directory)

#         self.hitrate_file_suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
#     def store_results(self, results, split):
#         if self.enable_cache:
#             prefix = "with_cache"
#         else:
#             prefix = "no_cache"

#         with open(os.path.join(self.hitrate_directory, f"{prefix}_split_{split[0]}_{split[1]}_time_{self.hitrate_file_suffix}.json"), "a") as f:
#             json.dump(results, f)
#             f.write("\n")
            
#     def run(self, split, verbose=False):
#         entry_count = self.dataset.get_entry_count()
#         split_count = entry_count // split[1]
#         # split[0] = 0, split[1] = 1
#         start = split_count * split[0]
#         end = split_count * (split[0] + 1)
#         print(f"Running benchmark on {self.dataset.dataset_name}, start: {start}, end: {end}")

#         for i in tqdm(range(start, end)):
#             # def get_query(self, range: Tuple[int, int]) -> List[Entry]:
#             #     return self.entries[range[0]:range[1]]
#             entries = self.dataset.get_query((i, i + 1))
from benchmark.longbench import LongBench, Entry
class MixedDataset:
    def __init__(self, datasets, sample_size):
        self.entries = []
        for dataset in datasets:
            lb = LongBench(dataset)
            lb.init()
            self.entries.extend(lb.entries)
        random.shuffle(self.entries)
        self.entries = self.entries[:sample_size]
        self.dataset_name = "mixed_dataset"

if __name__ == '__main__':
    sq = LongBench('narrativeqa')
    sq.init()
    print(sq.get_entry_count())
    print(sq.get_query((99, 101)))