from datasets import load_dataset
from datasets import load_from_disk
import os
dataset = load_from_disk(os.path.join("F:/datasets/LongBench/2wikimqa"))