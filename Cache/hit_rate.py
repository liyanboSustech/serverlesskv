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

