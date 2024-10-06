import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import transformers
import bitsandbytes as bnb
import torch.distributed as dist
from transformers import Trainer
from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from transformers import set_seed
from transformers import LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorForSeq2SeqForNeg:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = 'pt'

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        def pad_extra_attributes(attributes, name):
            max_length = max(len(attribute) for attribute in attributes)
            
            if self.pad_to_multiple_of is not None:
                max_length = (
                    (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
                )
                
                padding_side = self.tokenizer.padding_side
                        
                for feature in features:
                    remainder = [self.label_pad_token_id] * (max_length - len(feature[name]))

                    if isinstance(feature[name], list):
                        feature[name] = (
                            feature[name] + remainder if padding_side == 'right' else remainder + feature[name]
                        )
                    elif padding_side == 'right':
                        feature[name] = np.concatenate([feature[name], remainder]).astype(np.int64)
                    else:
                        feature[name] = np.concatenate([remainder, feature[name]]).astype(np.int64)
        
        pair = [feature['pair'] for feature in features] if 'pair' in features[0].keys() else None

        if pair:
            pad_extra_attributes(pair, 'pair')

        labels = [feature['label'] for feature in features]

        features = self.tokenizer.pad(
            {'input_ids': [feature['input_ids'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        features['pair'] = self.tokenizer.pad(
            {'input_ids': pair},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )['input_ids']

        features['label'] = torch.tensor(labels).to(features['input_ids'].device)
        return features

class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, x, y):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        var_x = torch.var(x)
        var_y = torch.var(y)

        pearson = torch.mean((x - mean_x) * (y - mean_y)) / (torch.sqrt(var_x) * torch.sqrt(var_y))
        return (-pearson + 1)

class SentembTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.is_sts:
            anchor, pair, label = inputs['input_ids'], inputs['pair'], inputs['label']
            pair[pair < 0] = 0

            # padding tensor length
            mw = max(anchor.size(1), pair.size(1))

            pad_size = mw - anchor.size(1)
            if pad_size > 0:
                anchor = torch.cat([torch.zeros(anchor.size(0), pad_size).cuda().long(), anchor], dim=1)

            pad_size = mw - pair.size(1)
            if pad_size > 0:
                pair = torch.cat([torch.zeros(pair.size(0), pad_size).cuda().long(), pair], dim=1)

            inputs['input_ids'] = torch.cat([anchor, pair], dim=0)
            inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
            del inputs['pair'], inputs['label']

        if hasattr(self, 'llama_avg') and self.llama_avg:
            hidden_states = model(output_hidden_states=True, return_dict=True, **inputs).hidden_states

            last_layer = hidden_states[-1]
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_layer.shape)
            pooler_output = (last_layer * attention_mask).mean(1)
        else:
            pooler_output = model(output_hidden_states=True, return_dict=True, **inputs).hidden_states[-1][:, -1, :]

        batch_size = pooler_output.size(0) // 2
        pooler_output = torch.stack([pooler_output[:batch_size],
                                     pooler_output[batch_size:]], dim=1)
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

        if dist.is_initialized():
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            label_list = [torch.zeros_like(label) for _ in range(dist.get_world_size())]

            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            dist.all_gather(tensor_list=label_list, tensor=label.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            label_list[dist.get_rank()] = label

            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
            label = torch.cat(label_list, 0)

        # anchor, pair
        cos_sim = F.cosine_similarity(z1, z2, dim=1)
        loss_function = PearsonCorrelationLoss()
        loss = loss_function(cos_sim, label)

        return (loss, pooler_output) if return_outputs else loss

def generate_sentemb_prompt(data_point, tokenizer, cutoff_len, template, prefix='input'):
    sp = f's{prefix}'

    if sp not in data_point:
        inputs = tokenizer(
            data_point[prefix],
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        inputs = tokenizer.decode(inputs['input_ids'])

        # handle special bug in OPT tokenizer
        if len(tokenizer.encode(inputs, add_special_tokens=False)) > cutoff_len:
            inputs = tokenizer.decode(tokenizer.encode(inputs, add_special_tokens=False)[:cutoff_len])

        data_point[sp] = inputs
    else:
        inputs = data_point[sp]

    del data_point[prefix]
    template = template.replace('_', ' ').replace('*sep+*', '').replace('*cls*', '')
    return template.replace('*sent 0*', inputs).strip()

def get_train_data(data, tokenizer, mask_embedding_sentence_template: str, cutoff_len: int = 32):
    def tokenize(anchor_prompt, pair_prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            anchor_prompt,
            padding=False,
            return_tensors=None,
        )
        
        pair_result = tokenizer(
            pair_prompt,
            padding=False,
            return_tensors=None,
        )

        result['pair'] = pair_result['input_ids']
        return result

    def generate_and_tokenize_prompt(data_point):
        anchor_template = mask_embedding_sentence_template
        pair_template = mask_embedding_sentence_template
        anchor_prefix, pair_prefix = 'sentence1', 'sentence2'

        anchor_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                                anchor_template, prefix=anchor_prefix)

        pair_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                              pair_template, prefix=pair_prefix)

        tokenized_full_prompt = tokenize(anchor_prompt, pair_prompt)
        tokenized_full_prompt['label'] = data_point['gold_label']
        return tokenized_full_prompt
    
    train_data = data['train'].shuffle().map(generate_and_tokenize_prompt, num_proc=25)
    return train_data

def train(
        # model/data params
        base_model: str = '',  # required
        lora_path: str = None,  # newly added
        data_path: str = '../data/stsb-sickr-train.jsonl',
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 256,
        micro_batch_size: int = 64,
        num_epochs: int = 1,
        learning_rate: float = 5e-4,
        cutoff_len: int = 32,
        # lora hyperparams
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        group_by_length: bool = False,  # faster, but produces an odd training loss curve,
        mask_embedding_sentence_template: str = None,
        run_name: str = None,
        load_kbit: int = 4,
        save_steps: int = 100,
        llama_avg: bool = False,
        seed: int = 42,
):
    set_seed(seed)
    assert load_kbit in {4, 8, 16}

    if 'sts' in data_path.lower():
        is_sts = True
    else:
        is_sts = False

    group_by_length = False
    run_name = data_path.split('.')[0]
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = 'auto'
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size != 1
    
    if ddp:
        device_map = {"": int(os.environ.get('LOCAL_RANK') or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    config = None
    if load_kbit == 4:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_kbit == 8 ,
            torch_dtype=torch.float16 if load_kbit == 16 else torch.float32,
            device_map=device_map,
        )
    
    if 'llama' in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

        if tokenizer.bos_token_id == 0:
            tokenizer.bos_token_id = 1
            tokenizer.eos_token = '</s>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)             

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    if load_kbit == 4:
        model = prepare_model_for_kbit_training(model)

        if lora_path:
            print(f'Load lora weight from {lora_path}')
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                torch_dtype=torch.float32,
                device_map=device_map,
                is_trainable=True,
            )
        else:
            def find_all_linear_names(model):
                cls = bnb.nn.Linear4bit
                lora_module_names = set()
                for name, module in model.named_modules():
                    if isinstance(module, cls):
                        names = name.split('.')
                        lora_module_names.add(names[0] if len(names) == 1 else names[-1])
                
                # needed for 16-bit
                if 'lm_head' in lora_module_names:
                    lora_module_names.remove('lm_head')
                    
                return list(lora_module_names)
            
            target_modules = find_all_linear_names(model)
            print('all linear layers: ', target_modules)

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)            

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.float32)
            if 'norm' in name:
                module = module.to(torch.float32)
            if ('lm_head' in name or 'embed_tokens' in name) and hasattr(module, 'weight'):
                module = module.to(torch.float32)
    else:
        if load_kbit == 8:
            model = prepare_model_for_int8_training(model)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
    
    # Be more transparent about the % of trainable params.
    model.print_trainable_parameters()

    if 'csv' in data_path:
        data = load_dataset('csv', data_files=data_path)
    else:
        data = load_dataset('json', data_files=data_path)

    DC_FUN = DataCollatorForSeq2SeqForNeg if is_sts else transformers.DataCollatorForSeq2Seq

    train_data = get_train_data(data, tokenizer=tokenizer, 
                                mask_embedding_sentence_template=mask_embedding_sentence_template, cutoff_len=cutoff_len)
    
    train_data = train_data.remove_columns(['ssentence1', 'ssentence2', 'attention_mask', 'gold_label'])
    print(f'world_size = {world_size}, gradient_accumulation_steps = {gradient_accumulation_steps}')

    trainer = SentembTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy='no',
            save_strategy='steps',
            eval_steps=None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=100,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            run_name=run_name,
            report_to=None,
            remove_unused_columns=False,
        ),
        data_collator=DC_FUN(
            tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True
        ),
    )
    
    trainer.is_sts = is_sts
    trainer.tokenizer = tokenizer
    trainer.llama_avg = llama_avg
    model.config.use_cache = False

    if torch.__version__ >= '2' and sys.platform != 'win32':
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    from settings import *
    assert backbone in ['opt-6.7b', 'llama-7b', 'llama2-7b', 'mistral-7b']

    data_file = '../data/merged-SICK-STS-B-train.jsonl'
    lora_path = f'./{backbone}-lora-{prompt_type}'
    output_dir = f'tune-{backbone}-lora-{prompt_type}'
    
    if backbone == 'opt-6.7b':
        if prompt_type == 'sth':
            num_epochs = 10
            batch_size = 216
            micro_batch_size = 54
        elif prompt_type == 'sum':
            num_epochs = 8
            batch_size = 200
            micro_batch_size = 50
        elif prompt_type == 'eol':
            num_epochs = 10
            batch_size = 232
            micro_batch_size = 58
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
                
        params = {
            'base_model': '../../models/opt-6.7b',
            'lora_path': lora_path,
            'data_path': data_file,
            'batch_size': batch_size,
            'micro_batch_size': micro_batch_size,
            'num_epochs': num_epochs,
            'learning_rate': 5e-4,
            'cutoff_len': 64,
            'lora_r': 64,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'output_dir': output_dir,
            'mask_embedding_sentence_template': manual_template,
            'save_steps': 25,
            'load_kbit': 4
        }
    elif backbone == 'llama-7b':
        if prompt_type == 'sth':
            num_epochs = 10
            batch_size = 232
            micro_batch_size = 58
        elif prompt_type == 'sum':
            num_epochs = 10
            batch_size = 200
            micro_batch_size = 50
        elif prompt_type == 'eol':
            num_epochs = 8
            batch_size = 192
            micro_batch_size = 48
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
        
        params = {
            'base_model': '../../models/llama-7b',
            'lora_path': lora_path,
            'data_path': data_file,
            'batch_size': batch_size,
            'micro_batch_size': micro_batch_size,
            'num_epochs': num_epochs,
            'learning_rate': 5e-4,
            'cutoff_len': 64,
            'lora_r': 64,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'output_dir': output_dir,
            'mask_embedding_sentence_template': manual_template,
            'save_steps': 25,
            'load_kbit': 4
        }
    elif backbone == 'llama2-7b':
        if prompt_type == 'sth':
            num_epochs = 10
            batch_size = 208
            micro_batch_size = 52
        elif prompt_type == 'sum':
            num_epochs = 8
            batch_size = 216
            micro_batch_size = 54
        elif prompt_type == 'eol':
            num_epochs = 10
            batch_size = 216
            micro_batch_size = 54
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
        
        params = {
            'base_model': '/home/datamining/ckh/llama/llama-2-7b-hf',
            'lora_path': lora_path,
            'data_path': data_file,
            'batch_size': batch_size,
            'micro_batch_size': micro_batch_size,
            'num_epochs': num_epochs,
            'learning_rate': 5e-4,
            'cutoff_len': 64,
            'lora_r': 64,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'output_dir': output_dir,
            'mask_embedding_sentence_template': manual_template,
            'save_steps': 25, 
            'load_kbit': 4
        }
    elif backbone == 'mistral-7b':
        if prompt_type == 'sth':
            num_epochs = 8
            batch_size = 240
            micro_batch_size = 60
        elif prompt_type == 'sum':
            num_epochs = 8
            batch_size = 208
            micro_batch_size = 52
        elif prompt_type == 'eol':
            num_epochs = 8
            batch_size = 216
            micro_batch_size = 54
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')

        params = {
            'base_model': '../../models/mistral-7b-v0.1',
            'lora_path': lora_path,
            'data_path': data_file,
            'batch_size': batch_size,
            'micro_batch_size': micro_batch_size,
            'num_epochs': num_epochs,
            'learning_rate': 5e-4,
            'cutoff_len': 64,
            'lora_r': 64,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'output_dir': output_dir,
            'mask_embedding_sentence_template': manual_template,
            'save_steps': 25,
            'load_kbit': 4
        }

    train(**params)
