from typing import Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
import pdb

def video_shared_collate_fn(instances: Sequence[Dict],
                           pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                           return_hf_format: bool = False,
                           use_varlen_attn: bool = False):
    """
    Unified collate function for both spatial and temporal grounding tasks
    """
    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids, labels = [], []
    
    # Common features
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    has_pe = any(inst.get('image_grid_thw', None) is not None for inst in instances)
    has_data_type = any(inst.get('type') is not None for inst in instances)
    data_type = instances[0].get('type') if has_data_type else None
    
    # Spatial grounding features
    has_bboxes = any(inst.get('bboxes') is not None for inst in instances)
    
    # Temporal grounding features
    has_temporal_info = any(inst.get('temporal_info') is not None for inst in instances)

    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    # Initialize containers for common features
    if has_image:
        pixel_values = []
        image_grid_thw = []
        frames_per_batch = []
    
    # Initialize containers for spatial features
    if has_bboxes:
        object_bboxes = []
    
    # Initialize containers for temporal features
    if has_temporal_info:
        temporal_info = []

    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))

        # Process common image features
        if has_image:
            pixel_values.append(example['pixel_values'])
            if has_pe:
                image_grid_thw.append(example['image_grid_thw'])
            if 'g_pixel_values' in example.keys():
                if isinstance(example['g_pixel_values'], list):
                    frames_per_batch.append(len(example['g_pixel_values']))
                else:
                    frames_per_batch.append(1)
        
        # Process spatial grounding features
        if has_bboxes:
            if 'bboxes' in example.keys() and example['bboxes'] is not None:
                object_bboxes.append(example['bboxes'])

        # Process temporal grounding features
        if has_temporal_info:
            if 'temporal_info' in example and example['temporal_info'] is not None:
                temporal_info.append(example['temporal_info'])

    # Process input_ids and labels padding
    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    # Process attention_mask and position_ids
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

    # Handle sequence parallel
    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        if attention_mask is not None:
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)

    # Build base data dictionary
    if use_varlen_attn: # False
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

    # Add common image data
    if has_image:
        if all(x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = torch.stack(pixel_values, dim=0)
        data_dict['frames_per_batch'] = frames_per_batch
        data_dict['pixel_values'] = pixel_values
        if has_pe:
            data_dict['image_grid_thw'] = image_grid_thw

    # Add spatial grounding data
    if has_bboxes:
        data_dict['bboxes'] = object_bboxes

    # Add temporal grounding data
    if has_temporal_info:
        data_dict['temporal_info'] = temporal_info

    # Add data type for task identification
    data_dict['type'] = data_type
    
    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}

"""
common: input_ids, labels, attention_mask, position_ids, pixel_values, type, frames_per_batch
spatial_grounding: (nothing)
temporal_grounding: temporal_info
"""