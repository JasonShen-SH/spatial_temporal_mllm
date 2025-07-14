import copy
from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
import pdb

def video_lisa_encode_fn(
        example,
        tokenizer,
        max_length,
        input_ids_with_output=True,
        **kwargs
):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for turn_idx, single_turn_conversation in enumerate(example['conversation']):
        input = single_turn_conversation['input']
        input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get('output_with_loss', True)
            output = single_turn_conversation['output']
            
            # new 
            if '{coords}' in output and turn_idx < len(example['bboxes_expression']):
                coords = example['bboxes_expression'][turn_idx]
                coords_concat = []
                for frame_idx in range(len(coords)):
                    frame_coords = coords[frame_idx].replace('<box>', '').replace('</box>', '')
                    coords_concat.append(frame_coords)
                coords_concat = '<next>'.join(coords_concat)
                output = output.replace('{coords}', coords_concat)
            
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss: # True
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
                
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    
    return {'input_ids': input_ids, 'labels': labels}


# def video_lisa_encode_multi_conv_fn(
#         example,
#         tokenizer,
#         max_length,
#         input_ids_with_output=True
# ):
#     """We only support the following three scenarios:

#     1. Incremental pretraining dataset.
#         example['conversation'] = [
#                 {
#                     'input': '',
#                     'output': '### Human: Can you write xxx'
#                 }
#             ]

#     2. Single-turn conversation dataset.
#         example['conversation'] = [
#                 {
#                     'input': 'Give three tips for staying healthy.',
#                     'output': '1.Eat a balanced diet xxx'
#                 }
#             ]

#     3. Multi-turn conversation dataset.
#         example['conversation'] = [
#                 {
#                     'input': 'Give three tips for staying healthy.',
#                     'output': '1.Eat a balanced diet xxx'
#                 },
#                 {
#                     'input': 'Please expand on the second point.',
#                     'output': 'Here is an expanded explanation of the xxx'
#                 }
#             ]
#     """
#     bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
#     assert not input_ids_with_output
#     input_id_list = []
#     for conv in example['conversation']:
#         input_ids = []
#         next_needs_bos_token = True
#         for single_turn_conversation in conv:
#             input = single_turn_conversation['input']
#             input_encode = tokenizer.encode(input, add_special_tokens=False)
#             if next_needs_bos_token:
#                 input_ids += bos_token_id
#             input_ids += input_encode

#         if len(input_ids) > max_length:
#             input_ids = input_ids[:max_length]

#         input_id_list.append(input_ids)
#     return {'input_ids': input_id_list}


# def video_lisa_encode_fn_spatial_switch(
#         example,
#         tokenizer,
#         max_length,
#         input_ids_with_output=True,
#         **kwargs
# ):
#     bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
#     is_multi_turn_conversation = len(example['conversation']) > 1
#     if is_multi_turn_conversation:
#         assert input_ids_with_output

#     input_ids, labels = [], []
#     next_needs_bos_token = True
    
#     switch_frame_idxs = example['real_switch_frames_idx']

#     for turn_idx, single_turn_conversation in enumerate(example['conversation']):
#         input = single_turn_conversation['input']
#         input_encode = tokenizer.encode(input, add_special_tokens=False)
#         # BOS
#         if next_needs_bos_token:
#             input_ids += bos_token_id
#             labels += [IGNORE_INDEX] * len(bos_token_id)
#         input_ids += input_encode
#         labels += [IGNORE_INDEX] * len(input_encode)
#         if input_ids_with_output: # True
#             # Add output
#             output_with_loss = single_turn_conversation.get('output_with_loss', True)
#             output = single_turn_conversation['output']
            
#             # new 
#             if '{coords}' in output and turn_idx < len(example['bboxes_expression']):
#                 coords = example['bboxes_expression'][turn_idx]
#                 coords_concat = []
#                 for frame_idx in range(len(coords)):
#                     frame_coords = coords[frame_idx].replace('<box>', '').replace('</box>', '')
#                     if frame_idx in switch_frame_idxs:
#                         frame_coords = '<switch>' + frame_coords
#                     coords_concat.append(frame_coords)
#                 coords_concat = '<next>'.join(coords_concat)
#                 output = output.replace('{coords}', coords_concat)
                
#             output_encode = tokenizer.encode(output, add_special_tokens=False)
#             input_ids += output_encode
#             if output_with_loss:
#                 labels += copy.deepcopy(output_encode)
#             else:
#                 labels += [IGNORE_INDEX] * len(output_encode)
#             # Add EOS_TOKEN (with loss)
#             if single_turn_conversation.get('need_eos_token', True):
#                 next_needs_bos_token = True
#                 input_ids += eos_token_id
#                 if output_with_loss:
#                     labels += copy.deepcopy(eos_token_id)
#                 else:
#                     labels += [IGNORE_INDEX] * len(eos_token_id)
#             else:
#                 next_needs_bos_token = False
#             # Add SEP (without loss)
#             sep = single_turn_conversation.get('sep', '')
#             if sep != '':
#                 sep_encode = tokenizer.encode(sep, add_special_tokens=False)
#                 input_ids += sep_encode
#                 labels += [IGNORE_INDEX] * len(sep_encode)
            
#             # 总结:
#             # INPUT, BOS, SEP, 不计算loss
#             # OUTPUT, EOS, 计算loss
            
#     if len(input_ids) > max_length:
#         input_ids = input_ids[:max_length]
#         labels = labels[:max_length]
    
#     return {'input_ids': input_ids, 'labels': labels}



# def video_lisa_encode_fn_box_point(
#         example,
#         tokenizer,
#         max_length,
#         input_ids_with_output=True,
#         **kwargs
# ):
#     """We only support the following three scenarios:

#     1. Incremental pretraining dataset.
#         example['conversation'] = [
#                 {
#                     'input': '',
#                     'output': '### Human: Can you write xxx'
#                 }
#             ]

#     2. Single-turn conversation dataset.
#         example['conversation'] = [
#                 {
#                     'input': 'Give three tips for staying healthy.',
#                     'output': '1.Eat a balanced diet xxx'
#                 }
#             ]

#     3. Multi-turn conversation dataset.
#         example['conversation'] = [
#                 {
#                     'input': 'Give three tips for staying healthy.',
#                     'output': '1.Eat a balanced diet xxx'
#                 },
#                 {
#                     'input': 'Please expand on the second point.',
#                     'output': 'Here is an expanded explanation of the xxx'
#                 }
#             ]
#     """
#     bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
#     is_multi_turn_conversation = len(example['conversation']) > 1
#     if is_multi_turn_conversation:
#         assert input_ids_with_output

#     input_ids, labels = [], []
#     next_needs_bos_token = True
#     for turn_idx, single_turn_conversation in enumerate(example['conversation']):
#         input = single_turn_conversation['input']
#         input_encode = tokenizer.encode(input, add_special_tokens=False)
#         if next_needs_bos_token:
#             input_ids += bos_token_id
#             labels += [IGNORE_INDEX] * len(bos_token_id)
#         input_ids += input_encode
#         labels += [IGNORE_INDEX] * len(input_encode)
#         if input_ids_with_output:
#             # Add output
#             output_with_loss = single_turn_conversation.get('output_with_loss', True)
#             output = single_turn_conversation['output']
            
#             if '{coords}' in output and turn_idx < len(example['bboxes_expression']):
#                 coords = example['bboxes_expression'][turn_idx]
#                 points = example['bboxes_points'][turn_idx]
#                 coords_concat = []
#                 points_concat = []
#                 for frame_idx in range(len(coords)):
#                     frame_coords = coords[frame_idx].replace('<box>', '').replace('</box>', '')
#                     frame_points = points[frame_idx].replace('<points>', '').replace('</points>', '')
#                     coords_concat.append(frame_coords)
#                     points_concat.append(frame_points)
#                 coords_concat = '<next>'.join(coords_concat)
#                 points_concat = '<next>'.join(points_concat)
#                 output = output.replace('{coords}', coords_concat)
#                 output = output.replace('{point_coords}', points_concat)
            
#             output_encode = tokenizer.encode(output, add_special_tokens=False)
#             input_ids += output_encode
#             if output_with_loss: # True
#                 labels += copy.deepcopy(output_encode)
#             else:
#                 labels += [IGNORE_INDEX] * len(output_encode)
#             # Add EOS_TOKEN (with loss)
#             if single_turn_conversation.get('need_eos_token', True):
#                 next_needs_bos_token = True
#                 input_ids += eos_token_id
#                 if output_with_loss:
#                     labels += copy.deepcopy(eos_token_id)
#                 else:
#                     labels += [IGNORE_INDEX] * len(eos_token_id)
#             else:
#                 next_needs_bos_token = False
                
#             # Add SEP (without loss)
#             sep = single_turn_conversation.get('sep', '')
#             if sep != '':
#                 sep_encode = tokenizer.encode(sep, add_special_tokens=False)
#                 input_ids += sep_encode
#                 labels += [IGNORE_INDEX] * len(sep_encode)

#     if len(input_ids) > max_length:
#         input_ids = input_ids[:max_length]
#         labels = labels[:max_length]

#     return {'input_ids': input_ids, 'labels': labels}