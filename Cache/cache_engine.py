from typing import List, Dict,Union,Tuple, Optional
import torch
import gc
import itertools
from .schema import TokenSequence, UnionModule, Schema, Path, Module
from .model import LanguageModel


KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


class TokenSequenceCache:
    # tokensequence是一个库，注意用来获取text：str的token_ids和position_ids
    token_sequence: TokenSequence
    host_cache: KVCache
    device_cache: Optional[KVCache] = None

    usage_counter: int = 0

    def __init__(self, seq: TokenSequence, cache: KVCache):
        self.token_sequence = seq
        self.host_cache = cache
        self.usage_counter = 0

    def inc_usage_counter(self):
        self.usage_counter += 1

    def upload(self, device: str):
        if self.device_cache is None:
            # 这个device_cache存储在哪里?
            # 这个device_cache存储在device上
            self.device_cache = [(kv[0].to(device, non_blocking=True),
                                  kv[1].to(device, non_blocking=True)) for kv in self.host_cache]

    def free(self):
        # need to invoke gc.collect() manually later
        if self.device_cache is not None:
            self.device_cache = None

    @property
    def cache(self) -> KVCache:
        # prioritize device cache
        if self.device_cache is None:
            return self.host_cache
        return self.device_cache

    def __len__(self):
        return len(self.token_sequence)


class PromptCache:
    staged: List[TokenSequenceCache]
    length: int

    max_ctx_length: int
    num_head: int
    head_dim: int
    device_cache: KVCache

    def __init__(self, max_ctx_length: int, num_layers: int, num_head: int, head_dim: int, target_device: torch.device):

        self.max_ctx_length = max_ctx_length
        self.num_head = num_head
        self.head_dim = head_dim
        # 得到的device_cache维度是(num_layers, num_head, max_ctx_length, head_dim)
        # 这里的num_layer和batchsize有什么关系？
        # num_layer是transformer的层数，batchsize是多少个token sequence
        # 获得的device_cache是一个四维的tensor，第一维是num_layers，第二维是num_head，第三维是max_ctx_length，第四维是head_dim
        # 不需要batchsize嘛？
        self.device_cache = [
            (torch.empty(num_head, max_ctx_length, head_dim, device=target_device, dtype=torch.half),  # key
             torch.empty(num_head, max_ctx_length, head_dim, device=target_device, dtype=torch.half)) for _ in
            range(num_layers)]

        # print(num_head, max_ctx_length, head_dim)

        # stores staged modules
        self.staged = []
        self.length = 0

    @torch.inference_mode()
    def update(self, modules: List[TokenSequenceCache]):

        # TODO: adopt in-place sorting to reduce redundant host-device memory copies

        # cache rearrangement -> becomes new layout
        # module_ordered的作用是将modules按照usage_counter从大到小排序
        modules_ordered = sorted(modules, key=lambda e: e.usage_counter, reverse=True)

        retained = []
        # 从staged中找到和modules_ordered中相同的部分的目的是为了将这部分的cache保留下来
        for (m, m_prev) in zip(modules_ordered, self.staged):
            if m.token_sequence == m_prev.token_sequence:
                retained.append(m)
            else:
                break

        offset = sum(map(len, retained))
        # 取出modules_ordered中没有的部分
        updates = modules_ordered[len(retained):]

        # update the cache
        for m in updates:
            # st 和 ed代表的是token sequence的起始和结束位置
            st = offset
            # ed为什么要多加st？
            # 
            ed = st + len(m)

            for i in range(len(self.device_cache)):
                k_cache_tgt, v_cache_tgt = self.device_cache[i]
                k_cache_src, v_cache_src = m.cache[i]

                # print('k_src', k_cache_src.shape)
                # print('v_src', v_cache_src.shape)
                # print('k_tgt', k_cache_tgt.shape)
                # print('v_tgt', v_cache_tgt.shape)

                k_cache_tgt[:, st:ed, :].copy_(k_cache_src, non_blocking=True)
                v_cache_tgt[:, st:ed, :].copy_(v_cache_src, non_blocking=True)

            offset += len(m)

        # re-organize the cache
        self.staged = modules
        self.length = offset

    def __len__(self):
        return self.length

    @property
    def cache(self) -> KVCache:
        return [(self.device_cache[i][0][:, :self.length, :],
                 self.device_cache[i][1][:, :self.length, :])
                for i in range(len(self.device_cache))]
                
class SchemaCache:

    

class CacheEngine:
    lm: LanguageModel
    schemas: Dict[str, SchemaCache]

    prompt_cache: PromptCache
    
    def __init__(self, max_ctx_length: int, lm: LanguageModel, target_device=None):

        self.lm = lm
        self.schemas = dict()
        self.target_device = lm.device if target_device is None else target_device

        num_layers, num_head, head_dim = lm.get_cache_shape()

        self.prompt_cache = PromptCache(
            max_ctx_length=max_ctx_length,
            num_layers=num_layers,
            num_head=num_head,
            head_dim=head_dim,
            target_device=self.target_device
        )
        
    def add_schema(self, schema: Union[str, Schema],
                   batch_size: int = 1,
                   max_tokens: Optional[int] = None,
                   no_cache: bool = False):

        if type(schema) == str:
            schema = Schema(schema, self.lm, max_tokens=max_tokens)

        if schema.name in self.schemas:
            raise ValueError(f'There is already a schema named {schema.name} in the cache')

        self.schemas[schema.name] = SchemaCache(schema, self.lm, batch_size, target_device=self.target_device,
                                                no_cache=no_cache)

    def get_schema(self, name: str) -> Optional[Schema]:
        if name not in self.schemas:
            return None
        return self.schemas[name].schema
    
    def remove_schema(self, name: str):
        if name not in self.schemas:
            raise ValueError(f'There is no such schema named {name} in the cache')

        del self.schemas[name]
        gc.collect()
        torch.cuda.empty_cache()

    def remove_all_schemas(self):

        # remove all schemas
        self.schemas = dict()

        gc.collect()
        torch.cuda.empty_cache()