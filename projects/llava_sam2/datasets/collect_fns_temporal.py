from typing import Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX

def video_temporal_collate_fn(instances: Sequence[Dict],
                            pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                            return_hf_format: bool = False,
                            use_varlen_attn: bool = False):
    """Temporal grounding专用的collate函数"""
    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids, labels = [], []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    has_pe = any(inst.get('image_grid_thw', None) is not None for inst in instances)
    has_query_tokens = any(inst.get('query_tokens') is not None for inst in instances)
    has_temporal_info = any(inst.get('temporal_info') is not None for inst in instances)

    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image:
        pixel_values = []
        image_grid_thw = []
    
    if has_query_tokens:
        query_tokens = []
    
    if has_temporal_info:
        temporal_info = []

    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))

        if has_image:
            pixel_values.append(example['pixel_values'])
            if has_pe:
                image_grid_thw.append(example['image_grid_thw'])
        
        if has_query_tokens:
            if 'query_tokens' in example and example['query_tokens'] is not None:
                query_tokens.append(example['query_tokens'])
            else:
                # 如果没有预设query_tokens，创建默认的
                # 这里暂时用零初始化，实际应该在模型中初始化
                default_query = torch.zeros(64, 1024)  # (num_query_tokens, query_hidden_dim)
                query_tokens.append(default_query)
        
        if has_temporal_info:
            if 'temporal_info' in example and example['temporal_info'] is not None:
                temporal_info.append(example['temporal_info'])

    ori_length = [len(ids) for ids in input_ids]
    
    # 处理input_ids和labels的padding
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    # 处理attention_mask和position_ids
    if use_varlen_attn:
        assert input_ids.size(1) % seq_parallel_world_size == 0
        attention_mask = None
        position_ids = torch.stack(position_ids, dim=0)
    else:
        attention_mask = torch.zeros_like(input_ids).bool()
        for i, length in enumerate(ori_length):
            attention_mask[i, :length] = True

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    # 序列并行处理
    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        if attention_mask is not None:
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)

    # 构建基础数据字典
    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }

    # 添加视频相关数据
    if has_image:
        if all(x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = torch.stack(pixel_values, dim=0)
        data_dict['pixel_values'] = pixel_values
        if has_pe:
            data_dict['image_grid_thw'] = image_grid_thw

    # 添加query tokens
    if has_query_tokens:
        if all(x.shape == query_tokens[0].shape for x in query_tokens):
            query_tokens = torch.stack(query_tokens, dim=0)
        data_dict['query_tokens'] = query_tokens

    # 添加temporal info
    if has_temporal_info:
        data_dict['temporal_info'] = temporal_info

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}