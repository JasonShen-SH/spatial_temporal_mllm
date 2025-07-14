from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .llava_sam2_box_2 import VideoLLaVASAMModel
from .temporal_grounding_model import (
    PooledTemporalPositionalEncoding, 
    QueryToFrameProjection
)
from .lisa import LisaModel
from xtuner.model.utils import get_peft_model_state_dict
import pdb

class VideoLLaVA_Shared(VideoLLaVASAMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        llm_dim = self.mllm.model.config.llm_config.hidden_size
        # Temporal grounding components
        self.query_hidden_dim = 1024
        self.query_to_frame = QueryToFrameProjection(
            query_dim=self.query_hidden_dim,
            llm_dim=llm_dim,
        )
        print("VideoLLaVA_Shared initialized with special_tokens: ", self.special_tokens)
    
    def _detect_task_type(self, data):
        pixel_values = data.get('pixel_values')
        if pixel_values is None:
            return 'unknown'
        
        # Temporal task
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 3:
            return 'temporal'
        
        # Spatial task: 
        # Case 1: Video format
        elif isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
            return 'spatial'
        
        # Case 2: Image format
        elif isinstance(pixel_values, list) and len(pixel_values) > 0:
            if all(isinstance(item, torch.Tensor) and item.dim() == 4 for item in pixel_values):
                return 'spatial'
        return 'unknown'
    
    def forward(self, data, data_samples=None, mode='loss'):
        task_type = self._detect_task_type(data)

        pixel_values = data['pixel_values']
        g_pixel_values = data.pop('g_pixel_values', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        bboxes = data.pop('bboxes', None)
        del g_pixel_values, bboxes, frames_per_batch
        
        if task_type == 'temporal' and pixel_values is not None:
            batch_size, num_frames = pixel_values.shape[:2]
            encoded_features = pixel_values
            # frame_positions = torch.arange(num_frames, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)
            # encoded_features = self.temporal_encoding(encoded_features, frame_positions)
            frame_features = self.query_to_frame(encoded_features)
            data['pooled_features'] = frame_features
        del pixel_values
        
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)
        
        torch.cuda.empty_cache()
        
        loss_dict = {
            'llm_loss': output.loss,
        }
        
        return loss_dict
    
    def state_dict(self, *args, **kwargs):
        # Overwrite
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        from collections import OrderedDict
        to_return = OrderedDict()
        
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
            raise NotImplementedError
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v for k, v in state_dict.items() if 'visual_encoder.' in k
            })
            raise NotImplementedError
            
        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
            elif self.arch_type == 'qwen':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.model, state_dict=state_dict)
                )
            elif self.arch_type == 'llava':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
        elif not self.mllm.freeze_llm:
            to_return.update({
                k: v for k, v in state_dict.items() if 'llm.' in k
            })
            raise NotImplementedError
            
        to_return.update({
            k: v for k, v in state_dict.items() if 'mlp1.' in k
        })
        to_return.update({
            k: v for k, v in state_dict.items() if 'model.multi_modal_projector.' in k
        })
        to_return.update({
            k: v for k, v in state_dict.items() 
            if 'lm_head.weight' in k or ('output' in k and 'sam2_model' not in k)
        })
        to_return.update({
            k: v for k, v in state_dict.items() 
            if 'embed_tokens.weight' in k or 'tok_embeddings' in k
        })
        
        temporal_components = [
            # 'temporal_encoding.',
            'query_to_frame.',
        ]
        
        for component in temporal_components:
            to_return.update({
                k: v for k, v in state_dict.items() if component in k
            })
                
        return to_return