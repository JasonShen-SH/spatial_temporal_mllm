import copy
import random
import glob
import json
import logging
import os
from typing import Literal

import torch

from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from xtuner.dataset.utils import encode_fn
from xtuner.dataset.map_fns import llava_map_fn

from projects.glamm.datasets.utils.utils import expand2square

from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST, BOX_QUESTIONS, BOX_ANSWER_LIST
from projects.glamm.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from third_parts.mmdet.datasets.refcoco import RefCocoDataset

from .utils import dynamic_preprocess
import pdb

# <box_sep>
# Refcoco系列不存在no_box, 故没有添加

class ReferSegmDataset_box(RefCocoDataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 data_root,
                 ann_file=None,
                 split_file=None,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 data_prefix=dict(img_path='train2014/'),
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=None,
            ann_file=ann_file,
            split_file=split_file,
            **kwargs,
        )
        self.begin_str = f'{DEFAULT_IMAGE_TOKEN}\n'
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = data_root
        self.template = prompt_template
        self.max_length = max_length
        if self.arch_type == 'intern_vl':
            # self._system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
            self._system = ''
            self.template['INSTRUCTION'] = '<|user|>\n{input}<|end|><|assistant|>\n'
        elif self.arch_type == 'qwen':
            self._system = ''
        elif self.arch_type == 'llava':
            self._system = ''

        self.num_classes_per_sample = num_classes_per_sample
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)
        self.arch_type = arch_type
        self.single_image_mode = single_image_mode
        self._max_refetch = 1000

        print("Image RES dataset, include {} items.".format(len(self)))

    @property
    def modality_length(self):
        import pickle
        length_list = []
        for idx in range(len(self)):
            length_list.append(100)
        return length_list

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # masks, phrases = [], []
        # bboxes, phrases = [], []
        phrases = []
        instances, text = ann_info['instances'], ann_info['text']

        # 重采样
        index = np.random.choice(range(len(instances)), self.num_classes_per_sample, replace=True)
        bboxes = [[] for _ in range(len(index))]
        
        for real_idx, idx in enumerate(index):
            inst = instances[idx]
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrases.append(phrase)
            
            # binary_mask = np.zeros((height, width), dtype=np.uint8)
            for seg_idx, seg in enumerate(inst["mask"]):
                rles = mask_utils.frPyObjects([seg], height, width)
                m = mask_utils.decode(rles)
                m = m.astype(np.uint8)
                # 计算box
                y_indices, x_indices = np.where(m.squeeze() > 0)
                if len(x_indices) == 0:
                    print("very strange mask, its image: ", ann_info['img_path'], " text: ", ann_info['text'])
                    x_min, x_max = int(width * 0.495), int(width * 0.505)
                    y_min, y_max = int(height * 0.495), int(height * 0.505) 
                else:
                    x_min, x_max = max(0, np.min(x_indices)), min(width, np.max(x_indices))
                    y_min, y_max = max(0, np.min(y_indices)), min(height, np.max(y_indices))
                    
                if seg_idx == 0:
                    bboxes[real_idx] = [x_min / width, y_min / height, x_max / width, y_max / height]
                elif seg_idx == 1:
                    tmp = bboxes[real_idx]
                    bboxes[real_idx] = []
                    bboxes[real_idx].append([tmp[0], tmp[1], tmp[2], tmp[3]])
                    bboxes[real_idx].append([x_min / width, y_min / height, x_max / width, y_max / height])
                    del tmp
                else:
                    bboxes[real_idx].append([x_min / width, y_min / height, x_max / width, y_max / height])
                
                # binary_mask += m.squeeze()
                
            # masks.append(binary_mask)
            
            # y_indices, x_indices = np.where(binary_mask > 0)
            # assert len(y_indices) > 0 and len(x_indices) > 0
            # x_min, x_max = np.min(x_indices), np.max(x_indices)
            # y_min, y_max = np.min(y_indices), np.max(y_indices)
            # normalized_bbox = [
            #     x_min / width,    # x1
            #     y_min / height,   # y1
            #     x_max / width,    # x2
            #     y_max / height    # y2
            # ]
            # bboxes.append(normalized_bbox)
        
        # pdb.set_trace()

        conversation = []
        for i, phrase in enumerate(phrases):
            # question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            question = random.choice(BOX_QUESTIONS).format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            # conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
            answer = random.choice(BOX_ANSWER_LIST).format(class_name=phrase,coords="{coords}")
            conversation.append({'from': 'gpt', 'value': answer})
            
        # masks = torch.stack([torch.from_numpy(mask) for mask in masks], dim=0)
        # bboxes = torch.tensor(bboxes, dtype=torch.float32)
        # pdb.set_trace() 

        ann_info.update({
            # 'masks': masks,
            'bboxes': bboxes,
            'conversations': conversation,
            'image': image_path
        })
        return ann_info

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        # if 'masks' in data_dict:
        #     out_data_dict['masks'] = data_dict['masks']
        
        if 'bboxes' in data_dict:
            out_data_dict['bboxes'] = data_dict['bboxes']

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None
            if hasattr(self, 'extra_image_processor'):
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                out_data_dict['g_pixel_values'] = g_pixel_values

            if self.single_image_mode:
                images = [image]
            else:
                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)
            if self.preprocessor is not None:
                if self.arch_type == 'qwen':
                    _data_dict = self.preprocessor(images, do_resize=True)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                    num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                elif self.arch_type == 'llava':
                    _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    num_image_tokens = _data_dict['pixel_values'].shape[0] * self.patch_token
                else:
                    raise NotImplementedError
                out_data_dict.update(_data_dict)
            else:
                pixel_values = [self.transformer(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                out_data_dict['pixel_values'] = pixel_values

                num_image_tokens = pixel_values.shape[0] * self.patch_token
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            token_dict = self.get_inputid_labels(data_dict['conversations'], data_dict['bboxes'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(data_dict['conversations'], data_dict['bboxes'], None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(1, 3, self.image_size, self.image_size)
            
        return out_data_dict

    def get_inputid_labels(self, conversations, bboxes, image_token_str) -> dict:
        input = ''
        out_conversation = []
        # pdb.set_trace()
        while conversations and conversations[0]['from'] == 'gpt': # False
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        
        bbox_idx = 0
        for idx, msg in enumerate(conversations):
            if msg['from'] == 'human':
                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                bbox = bboxes[bbox_idx]
                if type(bbox[0]) is list:
                    num_boxes = len(bbox)
                    coords_str = ""
                    for i in range(num_boxes):
                        coords_str += f"{bbox[i][0]:.3f}, {bbox[i][1]:.3f}, {bbox[i][2]:.3f}, {bbox[i][3]:.3f}"
                        if i < num_boxes - 1:
                            coords_str += "<box_sep>"
                else:
                    coords_str = f"{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}"
                msg_value = msg['value'].replace('{coords}', coords_str)
                out_conversation.append({
                    'input': input,
                    'output': msg_value.rstrip('.') + '<single_img>.'
                })
                input = ''
                bbox_idx += 1
            else:
                raise NotImplementedError

        # pdb.set_trace()
        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # pdb.set_trace()
        return {'input_ids': input_ids, 'labels': labels}

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            data['type'] = 'image'
            # 为节省CUDA，不赋值，因为值已经进入labels
            return data


# if __name__ == '__main__':
#     from transformers import CLIPImageProcessor, AutoTokenizer
#     from third_parts.segment_anything.utils.transforms import ResizeLongestSide

#     pretrained_model = 'MBZUAI/GLaMM-GranD-Pretrained'
#     llm_name_or_path = 'lmsys/vicuna-7b-v1.5'

#     tokenizer = dict(
#         type=AutoTokenizer.from_pretrained,
#         pretrained_model_name_or_path=llm_name_or_path)
#     image_processor = dict(
#         type=CLIPImageProcessor.from_pretrained,
#         pretrained_model_name_or_path='openai/clip-vit-large-patch14-336')
#     extra_image_processor = dict(
#         type=ResizeLongestSide,
#         target_length=1024,
#     )
#     from xtuner.utils.templates import PROMPT_TEMPLATE

#     prompt_template = PROMPT_TEMPLATE.vicuna
#     from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, template_map_fn
#     from projects.glamm.datasets.collate_fns.glamm_collate_fn import glamm_collate_fn

#     dataset = ReferSegmDataset(
#         tokenizer=tokenizer,
#         special_tokens=['[SEG]'],
#         extra_image_processor=extra_image_processor,
#         prompt_template=prompt_template,
#         data_root='data/coco/',
#         data_prefix=dict(img_path='train2014/'),
#         ann_file='refcoco+/instances.json',
#         split_file='refcoco+/refs(unc).p',
#     )
#     for i in range(1000):
#         dataset[i]