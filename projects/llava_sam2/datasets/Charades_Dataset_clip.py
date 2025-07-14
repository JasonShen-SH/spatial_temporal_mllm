import os
import json
import random
import logging
import numpy as np
from typing import Literal
import re

import torch
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset

from .encode_fn import video_lisa_encode_fn
import copy
import pdb

from PIL import ImageFont, ImageDraw
import open_clip

TEMPORAL_GROUNDING_QUESTIONS = [
    "When does '{action}' occur in this video?",
    "During what frames does '{action}' happen?", 
    "Determine the temporal boundaries of '{action}'.",
    "When can we see '{action}' in the video?",
    "What is the start and end frames of '{action}'?",
    "Find the frames when '{action}' occurs.",
    "When does '{action}' take place?",
    "Determine when '{action}' happens in the video.",
]

TEMPORAL_GROUNDING_ANSWERS = [
    "{start_tokens}-{end_tokens}"
]
FRAME_DIGIT_TOKENS = [
    '<frame_0>', '<frame_1>', '<frame_2>', '<frame_3>', '<frame_4>',
    '<frame_5>', '<frame_6>', '<frame_7>', '<frame_8>', '<frame_9>'
]

class VideoCharadesTemporalDataset_clip(Dataset):
    def __init__(self,
                image_folder,
                expression_file,
                tokenizer,
                template_map_fn=None,
                max_length=2048,
                lazy=True,
                repeats=1,
                special_tokens=None,
                sampled_frames=64,
                system_message='',
                fps=10,
                use_original_qa=False,
                num_query_tokens=128,
                query_hidden_dim=1024,
                **kwargs):
        
        super().__init__()
        
        self.image_folder = image_folder
        
        self.tokenizer = BUILDER.build(tokenizer)
        
        self.template_map_fn = template_map_fn
        self.lazy = lazy
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)
            
        self.max_length = max_length
        self.repeats = repeats
        self.sampled_frames = sampled_frames
        self._system = system_message
        self.fps = fps
        self.use_original_qa = use_original_qa
        self.num_query_tokens = num_query_tokens
        self.query_hidden_dim = query_hidden_dim
        
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

        self.tokenizer.add_tokens(special_tokens, special_tokens=True)
    
        self.time_token_map = {
            '<TIME_ZERO>': '0',
            '<TIME_ONE>': '1', 
            '<TIME_TWO>': '2',
            '<TIME_THREE>': '3',
            '<TIME_FOUR>': '4',
            '<TIME_FIVE>': '5',
            '<TIME_SIX>': '6',
            '<TIME_SEVEN>': '7',
            '<TIME_EIGHT>': '8',
            '<TIME_NINE>': '9',
            '<TIME_DOT>': '.'
        }
        
        self.json_file_preprocess(expression_file)
        
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        self.image_size = 448
        self.transformer = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.clip_features_path = 'clip_features/charades'
        
    def number_to_frame_tokens(self, number):
        number_str = f"{number:02d}"
        tokens = []
        for digit in number_str:
            tokens.append(f'<frame_{digit}>')
        return ''.join(tokens)

    def parse_time_tokens(self, time_string):
        for token, digit in self.time_token_map.items():
            time_string = time_string.replace(token, digit)
        
        try:
            return float(time_string)
        except ValueError:
            print(f"Warning: Could not parse time string: {time_string}")
            return 0.0

    def extract_time_range(self, answer_text):
        time_pattern = r'(<TIME_[A-Z]+>)+'
        time_matches = re.findall(time_pattern, answer_text)
        
        if len(time_matches) < 2:
            print(f"Warning: Could not find two time ranges in: {answer_text}")
            return 0.0, 0.0
        
        parts = answer_text.split(' - ')
        if len(parts) < 2:
            print(f"Warning: Could not split time range in: {answer_text}")
            return 0.0, 0.0
        
        start_part = parts[0]
        end_part = parts[1].split(' ')[0]
        
        start_time = self.parse_time_tokens(start_part)
        end_time = self.parse_time_tokens(end_part)
        
        return start_time, end_time

    def seconds_to_frame_index(self, seconds):
        return int(seconds * self.fps)

    def get_video_frames(self, video_path):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frame_folder = os.path.join(self.image_folder, video_id)
        
        if not os.path.exists(frame_folder):
            print(f"Warning: Frame folder not found: {frame_folder}")
            return []
        
        frame_files = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]
        frame_files.sort()
        
        return frame_files

    def json_file_preprocess(self, expression_file):
        with open(expression_file, 'r') as f:
            expression_data = json.load(f)
        
        self.text_data = []
        self.videos = set()
        
        self.bad_indices = set()
        
        for idx, item in enumerate(expression_data):
            video_path = item['video']
            if 'AKKWU' in video_path or 'LEOL6' in video_path or 'IOL8Q' in video_path:
                continue
            qa_pairs = item['QA']
            
            frame_files = self.get_video_frames(video_path)
            if not frame_files:
                continue
            
            total_frames = len(frame_files)
            
            for qa in qa_pairs:
                question = qa['q']
                answer = qa['a']
                
                start_time, end_time = self.extract_time_range(answer)
                start_frame = self.seconds_to_frame_index(start_time)
                end_frame = self.seconds_to_frame_index(end_time)
                
                if start_frame >= total_frames or end_frame >= total_frames + 20:
                    self.bad_indices.add(idx)
                
                start_frame = max(0, min(start_frame, total_frames - 1))
                end_frame = max(start_frame, min(end_frame, total_frames - 1))
                
                action_match = re.search(r"'([^']+)'", question)
                action = action_match.group(1) if action_match else "unknown action"
                
                meta = {
                    'video': video_path,
                    'video_id': os.path.splitext(os.path.basename(video_path))[0],
                    'frames': frame_files,
                    'action': action,
                    'question': question,
                    'answer': answer,
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'total_frames': total_frames
                }
                self.text_data.append(meta)
                self.videos.add(meta['video_id'])
        
        self.videos = list(self.videos)
        print(f"Loaded {len(self.text_data)} QA pairs from {len(self.videos)} videos")

    def prepare_text(self, qa_info, selected_frame_indices):
        task_prefix = '<TEMPORAL>'
        
        num_img_tokens = len(selected_frame_indices)
        query_token_str = f'{self.IMG_CONTEXT_TOKEN * num_img_tokens}'
        # 注意, temporal没有<IMG_START>和<IMG_END>, 因为每帧只有唯一的token
        
        if self.use_original_qa:
            question = qa_info['question']
            answer = qa_info['answer']
        else:
            action = qa_info['action']
            start_frame_abs = qa_info['start_frame']
            end_frame_abs = qa_info['end_frame']
            
            start_relative = selected_frame_indices.index(start_frame_abs)
            end_relative = selected_frame_indices.index(end_frame_abs)
            
            start_tokens = self.number_to_frame_tokens(start_relative)
            end_tokens = self.number_to_frame_tokens(end_relative)
            
            question_template = random.choice(TEMPORAL_GROUNDING_QUESTIONS)
            answer_template = random.choice(TEMPORAL_GROUNDING_ANSWERS)
            
            question = question_template.format(action=action)
            answer = answer_template.format(start_tokens=start_tokens, end_tokens=end_tokens)
        
        full_input = task_prefix + query_token_str + question
        
        conversation = [{
            'input': full_input,
            'output': answer,
            'system': self._system
        }]
        
        return {'conversation': conversation}

    def sample_frames(self, total_frames, start_idx, end_idx):
        # 第一步：随机决定范围内和范围外的帧数
        target_frames = random.randint(20, 40)
        
        selected_indices = []
        # 第二步：范围内采样（必须包含start_idx和end_idx）
        target_range = list(range(start_idx, end_idx + 1))
        selected_in_target = [start_idx, end_idx]
        remaining_target = [idx for idx in target_range if idx not in selected_in_target]
        if len(remaining_target) >= (target_frames - 2):
            additional_target = random.sample(remaining_target, target_frames - 2)
        else:
            target_frames = len(remaining_target) + 2
            additional_target = random.sample(remaining_target, len(remaining_target))
        selected_in_target.extend(additional_target)
        selected_indices.extend(selected_in_target)
        
        # 第三步：范围外采样
        other_frames = self.sampled_frames - target_frames
        other_range = [i for i in range(total_frames) if i < start_idx or i > end_idx]
        if len(other_range) >= other_frames:
            selected_others = random.sample(other_range, other_frames)
        else:
            print(f"Warning: Not enough frames in 'other' range, got only {len(other_range)} frames, resampling...")
            selected_others = random.choices(other_range, k=other_frames)
        selected_indices.extend(selected_others)
        
        selected_indices = [int(idx) for idx in selected_indices]
        selected_indices.sort()
        
        return selected_indices

    def __len__(self):
        return len(self.text_data) * self.repeats
    
    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = 33000
            length_list.append(cur_len)
        return length_list

    def real_len(self):
        return len(self.text_data)

    def __getitem__(self, index):
        index = index % self.real_len()
        if index in self.bad_indices:
            return self.__getitem__(random.randint(0, self.real_len()))
    
        qa_info = copy.deepcopy(self.text_data[index])
        total_frames = qa_info['total_frames']
        start_idx = qa_info['start_frame']
        end_idx = qa_info['end_frame']
        
        if start_idx >= end_idx:
            print(f"Warning: 开始时间大于结束时间: {qa_info['video_id']}")
            return self.__getitem__(random.randint(0, self.real_len()))
        
        video_id = qa_info['video_id']
        
        ####################################################
        # requirement 1
        if total_frames < self.sampled_frames:
            print(f"Warning: Video too short ({total_frames} frames), resampling...")
            return self.__getitem__(random.randint(0, self.real_len()))
        # # requirement 2
        target_interval = end_idx - start_idx + 1
        if target_interval < 20:
            print(f"Warning: Target interval too small ({target_interval} frames), resampling...")
            return self.__getitem__(random.randint(0, self.real_len()))
        # requirement 3
        frames_before = start_idx
        frames_after = total_frames - (end_idx + 1)
        available_other_frames = frames_before + frames_after
        if available_other_frames < 80:
            print(f"Warning: Not enough frames outside target interval (only {available_other_frames} frames), resampling...")
            return self.__getitem__(random.randint(0, self.real_len()))
        ####################################################
        
        if start_idx >= total_frames or end_idx >= total_frames:
            print(f"Warning: Frame indices out of range: start={start_idx}, end={end_idx}, total={total_frames}")
            start_idx = min(start_idx, total_frames - 1)
            end_idx = min(end_idx, total_frames - 1)
        
        selected_frame_indices = self.sample_frames(total_frames, start_idx, end_idx)
        
        ##################################################################
        # 打乱顺序
        # frame_order_pairs = list(enumerate(selected_frame_indices))
        # random.shuffle(frame_order_pairs)
        # shuffled_indices, original_frame_indices = zip(*frame_order_pairs)
        # selected_frame_indices = list(original_frame_indices)
        # shuffled_positions = list(shuffled_indices)
        ##################################################################
        
        pixel_values = []
        images = []
        frame_files = qa_info['frames']
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        
        for idx, frame_idx in enumerate(selected_frame_indices, 0):
            # if frame_idx >= len(frame_files):
            #     frame_idx = len(frame_files) - 1  
            # frame_file = frame_files[frame_idx]
            # image_path = os.path.join(self.image_folder, qa_info['video_id'], frame_file)
            # images.append(image_path)
            
            # if not os.path.exists(image_path):
            #     print(f"Warning: Image not found: {image_path}")
            #     frame_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
            # else:
            #     frame_image = Image.open(image_path).convert('RGB')
                
            #     draw = ImageDraw.Draw(frame_image)
            #     width, height = frame_image.size
            #     text = str(idx)
            #     text_bbox = draw.textbbox((0, 0), text, font=font)
            #     text_width = text_bbox[2] - text_bbox[0]
            #     text_height = text_bbox[3] - text_bbox[1]
            #     x = width - text_width - 20
            #     y = height - text_height - 20
            #     draw.text((x, y), text, fill='red', font=font)
            
            # # self._ensure_clip_on_cuda()
            
            # with torch.no_grad(), torch.autocast("cuda"):
            #     preprocessed_image = self.clip_preprocess(frame_image).cuda()
            #     clip_features = self.clip_model.encode_image(preprocessed_image.unsqueeze(0))
            #     clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            #     clip_features = clip_features.to(torch.bfloat16)
            #     pixel_values.append(clip_features.squeeze(0).cpu())
            
            clip_features = torch.load(os.path.join(self.clip_features_path, f'{video_id}.pt')).to(torch.bfloat16)
            pixel_values.append(clip_features[frame_idx])
            
        pixel_values = torch.stack(pixel_values, dim=0) # 这里的pixel_values已经是特征
        
        text_dict = self.prepare_text(qa_info, selected_frame_indices)
        
        data_dict = {
            'images': images,
            'pixel_values': pixel_values,
            'conversation': text_dict['conversation'],
            'type': 'video',
            'selected_frame_indices': selected_frame_indices,
            'qa_info': qa_info,
            # 'frame_positions': torch.tensor(selected_frame_indices, dtype=torch.long),
            'frame_positions': None,
            'target_start_frame': start_idx,
            'target_end_frame': end_idx,
            'num_query_tokens': self.num_query_tokens,
            'query_hidden_dim': self.query_hidden_dim,
            # 'frame_positions': shuffled_positions
        }
        
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)
        
        return data_dict