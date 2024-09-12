import argparse
import torch
from .model import LanguageModel
from .schema import Schema, Module, ModuleRef, Prompt
from typing import List, Tuple, Dict, Union, Optional
import itertools
import gc
from cache_engine import KVCache, SchemaCache, PromptCache

class CacheEngine:
    lm: LanguageModel
    schemas: Dict[str, SchemaCache]

    prompt_cache: PromptCache
    
    def __init__(self, max_ctx_length: int, lm: LanguageModel, target_device=None,sharing_ratio=0.5, evicting_ratio=0.1):

        self.lm = lm
        self.schemas = dict()
        self.target_device = lm.device if target_device is None else target_device
        self.sharing_ratio = sharing_ratio
        self.evicting_ratio = evicting_ratio
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
    
    def process(self, prompt: Prompt, no_cache: bool = False, return_full_position_ids: bool = False) -> Tuple[
        List[int], List[int], float, Optional[KVCache]]:

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        # assert that root tag matches engine signature
        if prompt.schema not in self.schemas:
            raise ValueError(f'There is no such layout named {prompt.schema} in the cache')

        cached = self.schemas[prompt.schema]
        schema = cached.schema

        orig_ids_list = []
        orig_pos_ids_list = []

        used_sequences = []
        argument_ids_list = []
        argument_pos_ids_list = []

        # first add root level modules
        stack: List[(ModuleRef, Module)] = [(prompt, schema)]

        while len(stack) > 0:
            ref, module = stack.pop()

            # step 1. first add leaf nodes
            for m in module.token_sequences():
                # kv_cache_list.append(cached.get_cache_l1(m))
                used_sequences.append(m)

                if no_cache or return_full_position_ids:
                    orig_ids_list.append(m.token_ids())
                    orig_pos_ids_list.append(m.position_ids())

            # step 2. process parameter-argument pairs
            parameters = module.parameters()
            for arg in ref.args:

                parameter = None
                for p in parameters:
                    if p.name == arg.name:
                        parameter = p
                        break

                if parameter is None:
                    raise ValueError(f'There is no such parameter named {arg.name} in the module {module.name}')

                argument_ids = self.lm.encode(arg.value)

                if len(argument_ids) > parameter.length:
                    raise ValueError(
                        f'The argument {arg.name} is too long. It should be at most {parameter.length} characters long')

                argument_pos_ids = parameter.position_ids()[:len(argument_ids)]

                argument_ids_list.append(argument_ids)
                argument_pos_ids_list.append(argument_pos_ids)

            # step 3. update stack
            for m in ref.modules:
                submodule = module.select(m.name)
                if submodule is None:
                    raise ValueError(f'There is no such module named @{m.name} in the module @{module.name}')

                stack.append((m, submodule))
        
        if len(prompt.text) > 0:
            text_token_ids = self.lm.encode(prompt.text)
            text_position_ids = list(range(len(schema), len(schema) + len(text_token_ids)))

            argument_ids_list.append(text_token_ids)
            argument_pos_ids_list.append(text_position_ids)

        input_ids = list(itertools.chain(*argument_ids_list))
        position_ids = list(itertools.chain(*argument_pos_ids_list))

        if no_cache:
            orig_input_ids = list(itertools.chain(*orig_ids_list))
            orig_position_ids = list(itertools.chain(*orig_pos_ids_list))

            sorted_pairs = sorted(zip(orig_position_ids + position_ids, orig_input_ids + input_ids))

            # Unpack the sorted pairs into two lists
            orig_position_ids, orig_input_ids = zip(*sorted_pairs)

            end.record()
            torch.cuda.synchronize()
            cache_time = start.elapsed_time(end)

            # print(f'Cache overhead: {cache_time:.2f} ms')

            vv = list(range(len(orig_position_ids)))

            return orig_input_ids, vv, cache_time, None
        else:

            used_seq_caches = []

            for s in used_sequences:
                seq_cache = cached.get_cache_l1(s)

                seq_cache.inc_usage_counter()
                used_seq_caches.append(seq_cache)

            # update prompt cache. this incurs some memcpy overhead.
            self.prompt_cache.update(used_seq_caches)
            cache = self.prompt_cache.cache
            end.record()
            torch.cuda.synchronize()
            cache_time = start.elapsed_time(end)

            # apply read hook
            for i in range(len(cache)):
                cache[i] = (self.lm.read_k_hook(cache[i][0]), self.lm.read_v_hook(cache[i][1]))

            # print(f'Cache overhead: {cache_time:.2f} ms')

            if return_full_position_ids:
                orig_position_ids = list(itertools.chain(*orig_pos_ids_list))
                position_ids = orig_position_ids + position_ids

            # print(orig_position_ids)
            return input_ids, position_ids, cache_time, cache
    
    def evict(self, evicting_ratio ):
        cache = self.prompt_cache.cache
        
        num_cache = len(cache)
        
        num_cache_evict = int(num_cache * evicting_ratio)
            
        
            
        
            
        
