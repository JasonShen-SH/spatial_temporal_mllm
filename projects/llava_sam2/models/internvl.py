import torch
from xtuner.model import InternVL_V1_5
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
import torch.nn as nn
import pdb
from mmengine import print_log
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BitsAndBytesConfig)
from xtuner.model.utils import (find_all_linear_names, get_peft_model_state_dict,
                    guess_load_checkpoint, make_inputs_require_grad)
import os
import math
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from projects.llava_sam2.models.decoder import create_simple_internvl_decoder

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

# This function is used to split large model
def split_model(model_name):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.2))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.8)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    return device_map

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
        
class InternVL_Slowfast(InternVL_V1_5):

    def __init__(self,
                 model_path,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 quantization_vit=False,
                 quantization_llm=False,
                 pretrained_pth=None,
                 special_tokens=None,
                 model_split=False,
                 ):
        print_log('Start to load InternVL_V1_5 model.', logger='current')
        super(InternVL_V1_5, self).__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ['mlp1']
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append('vision_model')

            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append('language_model')

            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
            quantization_clazz = quantization_config.pop('type')
            quantization = quantization_clazz(**quantization_config)

        if model_split:
            # print("\n\nDone Model Split !!!!!!!!!!!\n\n")
            device_map = split_model("InternVL2-26B")
            # print(device_map)
            self.device = 'cuda'
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device_map).eval()

        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization,
                config=config,
                trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.tokenizer = tokenizer

        if special_tokens is not None:
            self._add_special_tokens(special_tokens)

        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = img_context_token_id
        
        # query_token_id = tokenizer.convert_tokens_to_ids('<query>')
        # self.model.query_token_id = query_token_id
        
        # mask_token_id = tokenizer.convert_tokens_to_ids('<mask>')
        # self.model.mask_token_id = mask_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)

        if hasattr(self.model.language_model, 'enable_input_require_grads'):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings(
            ).register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self._count = 0
        print_log(self, logger='current')
        print_log('InternVL_V1_5 construction is complete', logger='current')

        self.transfer_to_hf = False
        
        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, 4096))
        # trunc_normal_(self.mask_tokens, std=.02)
        torch.nn.init.normal_(self.mask_tokens, std=.02)
        
        # self.mae_decoder = create_simple_internvl_decoder(lightweight=False)
    
    def _add_special_tokens(self, special_tokens):
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)

        if num_new_tokens > 0:
            self.model.language_model.resize_token_embeddings(len(self.tokenizer))

    def _post_init(self, fast_pool_size=4, fast_pool=True):
        if fast_pool:
            self.fast_pool = nn.AdaptiveAvgPool2d((fast_pool_size, fast_pool_size))
        return

    def forward(self, data, data_samples=None, mode='loss', fast_token_idx=None):
        if 'fast_pixel_values' in data.keys():
            assert False
            assert fast_token_idx is not None
            fast_pixel_values = data['fast_pixel_values']
            if type(fast_pixel_values) is list or fast_pixel_values.ndim == 5:
                if type(fast_pixel_values) is list:
                    fast_pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in fast_pixel_values
                    ]
                # b*n, c, h, w
                fast_concat_images = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in fast_pixel_values], dim=0)
            else:
                raise NotImplementedError()
        else:
            fast_pixel_values = None
            fast_concat_images = None

        pixel_values = data['pixel_values']
        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        labels = data['labels']
        use_cache = False
        
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 3:
            # temporal
            concat_images = None
            image_flags = None
        else:
            # spatial
            if type(pixel_values) is list or pixel_values.ndim == 5:
                if type(pixel_values) is list:
                    pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                    ]
                # b*n, c, h, w
                concat_images = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)
            else:
                raise NotImplementedError()

            # sum is 0 are text
            image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()

        if 'vp_overall_mask' not in data.keys():
            vp_overall_mask = None
        else:
            vp_overall_mask = data['vp_overall_mask']

        if 'prompt_masks' in data.keys():
            prompt_masks = data['prompt_masks']
        else:
            prompt_masks = None
        
        del pixel_values
        
        ##############################################
        # shared用这个
        pooled_features = data.get('pooled_features', None)
        outputs = self._llm_forward_shared(
            pixel_values=concat_images,      # spatial会用这个
            pooled_features=pooled_features, # temporal会用这个，spatial时为None
            image_flags=image_flags,         # spatial会用这个
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        ##############################################
        # 专为temporal打造, spatial时去掉
        # outputs = self._llm_forward_temporal_pooled(
        #     input_ids=input_ids,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     # image_flags=image_flags,
        #     # pixel_values=concat_images,
        #     # query_embeds=data['query_embeds'],
        #     pooled_features=data['pooled_features'],
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_hidden_states=True,
        # )
        ##############################################
        
        ##############################################
        # 专为同时压缩图像和视频打造
        
        # instance_types = data.get('instance_types', ['image'] * len(input_ids))
        # has_image_type = data.get('has_image_type', True)
        # has_video_type = data.get('has_video_type', False)
        # print("instance_types", instance_types)
        # print("input ids", input_ids.shape)
        
        # if not has_image_type and has_video_type:
        #     outputs = self._llm_forward_compress_pseudo(
        #         input_ids=input_ids,
        #         position_ids=position_ids,
        #         attention_mask=attention_mask,
        #         image_flags=image_flags,
        #         pixel_values=concat_images,
        #         labels=labels,
        #         use_cache=use_cache,
        #         output_hidden_states=True,
        #         fast_pixel_values=fast_concat_images,
        #         fast_token_idx=fast_token_idx,
        #         vp_overall_mask=vp_overall_mask,
        #         prompt_masks=prompt_masks,
        #         dense_frames_per_batch=data.get('dense_frames_per_batch'),
        #         sparse_frames_per_batch=data.get('sparse_frames_per_batch'),
        #     )
            
        # elif has_image_type and not has_video_type:
        #     assert False
        #     outputs = self._llm_forward(
        #         input_ids=input_ids,
        #         position_ids=position_ids,
        #         attention_mask=attention_mask,
        #         image_flags=image_flags,
        #         pixel_values=concat_images,
        #         labels=labels,
        #         use_cache=use_cache,
        #         output_hidden_states=True,
        #         fast_pixel_values=fast_concat_images,
        #         fast_token_idx=fast_token_idx,
        #         vp_overall_mask=vp_overall_mask,
        #         prompt_masks=prompt_masks,
        #     )

        # else:
        #     assert False
        #     image_indices = [i for i, t in enumerate(instance_types) if t == 'image']
        #     video_indices = [i for i, t in enumerate(instance_types) if t == 'video']
        #     assert image_indices and video_indices
        #     outputs = self._llm_forward_compress_pseudo_img_video(
        #         input_ids=input_ids,
        #         position_ids=position_ids,
        #         attention_mask=attention_mask,
        #         image_flags=image_flags,
        #         pixel_values=concat_images,
        #         labels=labels,
        #         use_cache=use_cache,
        #         output_hidden_states=True,
        #         # fast_pixel_values=fast_concat_images,
        #         # fast_token_idx=fast_token_idx,
        #         # vp_overall_mask=vp_overall_mask,
        #         # prompt_masks=prompt_masks,
        #         dense_frames_per_batch=data.get('dense_frames_per_batch'),
        #         sparse_frames_per_batch=data.get('sparse_frames_per_batch'),
        #         img_video_combine=True,
        #         image_indices=image_indices,
        #         video_indices=video_indices,
        #     )
        ##############################################
        
            # image_data = {}
            # image_data['input_ids'] = input_ids[image_indices]
            # image_data['position_ids'] = position_ids[image_indices]
            # image_data['attention_mask'] = attention_mask[image_indices]
            # image_data['image_flags'] = image_flags[:pixel_values[0].shape[0]] if image_indices[0] == 0 else image_flags[pixel_values[0].shape[0]:]
            # image_data['pixel_values'] = pixel_values[image_indices[0]].squeeze().to(torch.bfloat16)
            # image_data['labels'] = labels[image_indices]
            # image_data['use_cache'] = use_cache
            # image_outputs = self._llm_forward(
            #     input_ids=image_data['input_ids'],
            #     position_ids=image_data['position_ids'],
            #     attention_mask=image_data['attention_mask'],
            #     image_flags=image_data['image_flags'],
            #     pixel_values=image_data['pixel_values'],
            #     labels=image_data['labels'],
            #     use_cache=use_cache,
            #     output_hidden_states=True,
            #     fast_pixel_values=image_data.get('fast_pixel_values'), # 空
            #     fast_token_idx=fast_token_idx, # 空
            #     vp_overall_mask=image_data.get('vp_overall_mask'), # 空
            #     prompt_masks=image_data.get('prompt_masks'), # 空
            # )
            # for i, idx in enumerate(image_indices):
            #     all_outputs[idx] = image_outputs
            # del image_data, image_outputs
            
            # video_data = {}
            # video_data['input_ids'] = input_ids[video_indices]
            # video_data['position_ids'] = position_ids[video_indices]
            # video_data['attention_mask'] = attention_mask[video_indices]
            # video_data['image_flags'] = image_flags[:pixel_values[0].shape[0]] if video_indices[0] == 0 else image_flags[pixel_values[0].shape[0]:]
            # video_data['pixel_values'] = pixel_values[video_indices[0]].squeeze().to(torch.bfloat16)
            # video_data['labels'] = labels[video_indices]
            # video_data['use_cache'] = use_cache
            # video_data['dense_frames_per_batch'] = data['dense_frames_per_batch'][video_indices[0]]
            # video_data['sparse_frames_per_batch'] = data['sparse_frames_per_batch'][video_indices[0]]
            # video_outputs = self._llm_forward_compress_pseudo(
            #     input_ids=video_data['input_ids'],
            #     position_ids=video_data['position_ids'],
            #     attention_mask=video_data['attention_mask'],
            #     image_flags=video_data['image_flags'],
            #     pixel_values=video_data['pixel_values'],
            #     labels=video_data['labels'],
            #     use_cache=use_cache,
            #     output_hidden_states=True,
            #     fast_pixel_values=video_data.get('fast_pixel_values'), # 空
            #     fast_token_idx=fast_token_idx, # 空
            #     vp_overall_mask=video_data.get('vp_overall_mask'), # 空
            #     prompt_masks=video_data.get('prompt_masks'), # 空
            #     dense_frames_per_batch=[video_data['dense_frames_per_batch']],
            #     sparse_frames_per_batch=[video_data['sparse_frames_per_batch']],
            # )
            # for i, idx in enumerate(video_indices):
            #     all_outputs[idx] = video_outputs
            # del video_data, video_outputs
        
            # outputs = self._merge_outputs(all_outputs)
            # del all_outputs
            # torch.cuda.empty_cache()
        
        # del instance_types, has_image_type, has_video_type
        
        # outputs = self._llm_forward(
        #     input_ids=input_ids,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     image_flags=image_flags,
        #     pixel_values=concat_images,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_hidden_states=True,
        #     fast_pixel_values=fast_concat_images,
        #     fast_token_idx=fast_token_idx,
        #     vp_overall_mask=vp_overall_mask,
        #     prompt_masks=prompt_masks,
        # )
        
        # outputs = self._llm_forward_compress(
        #     input_ids=input_ids,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     image_flags=image_flags,
        #     pixel_values=concat_images,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_hidden_states=True,
        #     fast_pixel_values=fast_concat_images,
        #     fast_token_idx=fast_token_idx,
        #     vp_overall_mask=vp_overall_mask,
        #     prompt_masks=prompt_masks,
        #     dense_frames_per_batch=data.get('dense_frames_per_batch'),
        #     sparse_frames_per_batch=data.get('sparse_frames_per_batch'),
        #     # use_dense_sparse=data.get('use_dense_sparse'),
        # )
        
        # outputs = self._llm_forward_compress_pseudo(
        #     input_ids=input_ids,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     image_flags=image_flags,
        #     pixel_values=concat_images,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_hidden_states=True,
        #     fast_pixel_values=fast_concat_images,
        #     fast_token_idx=fast_token_idx,
        #     vp_overall_mask=vp_overall_mask,
        #     prompt_masks=prompt_masks,
        #     dense_frames_per_batch=data.get('dense_frames_per_batch'),
        #     sparse_frames_per_batch=data.get('sparse_frames_per_batch'),
        #     # middle_frames_per_batch=data.get('middle_frames_per_batch'),
        # )

        # outputs = self._llm_forward_mask_token(
        #     input_ids=input_ids,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     image_flags=image_flags,
        #     pixel_values=concat_images,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_hidden_states=True,
        #     fast_pixel_values=fast_concat_images,
        #     fast_token_idx=fast_token_idx,
        #     vp_overall_mask=vp_overall_mask,
        #     prompt_masks=prompt_masks,
        #     data_type=data_type,
        # )
                
        return outputs
    
    def _llm_forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fast_pixel_values=None,
        fast_token_idx=None,
        vp_overall_mask=None,
        prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids).clone()

        if fast_pixel_values is not None:
            n_fast_images = fast_pixel_values.shape[0]
            whole_pixel_values = torch.cat([fast_pixel_values, pixel_values], dim=0)
            vit_embeds = self.model.extract_feature(whole_pixel_values)
            vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
            fast_vit_embeds = vit_embeds[:n_fast_images]  # (n_fast_images, hw, c)
            _size = int(fast_vit_embeds.shape[1] ** 0.5)
            fast_vit_embeds = fast_vit_embeds.reshape(fast_vit_embeds.shape[0], _size, _size, fast_vit_embeds.shape[-1])
            # pooling
            fast_vit_embeds = fast_vit_embeds.permute(0, 3, 1, 2)  # (n_fast_images, c, h, w)
            fast_vit_embeds = self.fast_pool(fast_vit_embeds).flatten(2)  # (n_fast_images, c, hw)
            fast_vit_embeds = fast_vit_embeds.permute(0, 2, 1)
            vit_embeds = vit_embeds[n_fast_images:]
        else:
            vit_embeds = self.model.extract_feature(pixel_values)
            vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
            fast_vit_embeds = None

        # pdb.set_trace()
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        if vp_overall_mask is not None and prompt_masks is not None:
            assert False
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask] # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1
            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)

        if vp_embeds is None:
            try:
                input_embeds[selected] = vit_embeds.reshape(-1, C)
            except Exception as e:
                assert False
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vit_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vit_embeds) + 1
                    vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vit_embeds[:n_token]
        else:
            assert False
            try:
                input_embeds[selected] = vp_embeds.reshape(-1, C)
            except Exception as e:
                vp_embeds = vp_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vp_embeds[:n_token]

        if fast_vit_embeds is not None:
            assert False
            selected = (input_ids == fast_token_idx)
            selected_tot = selected.sum().item()
            if selected_tot > fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]:
                assert selected_tot % (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]) == 0
                repeat_times = selected_tot / (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1])
                fast_vit_embeds = fast_vit_embeds.repeat(int(repeat_times), 1, 1)
            try:
                input_embeds[selected] = fast_vit_embeds.reshape(-1, C)
            except Exception as e:
                fast_vit_embeds = fast_vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[fast_selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'fast_vit_embeds.shape={fast_vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = fast_vit_embeds[:n_token]

        del vit_embeds
        
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        del input_embeds

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def grid_pool_features(self, features, num_tokens=4):
        """
        Input: features (B, N, C) where N is number of tokens
        Output: features (B, num_tokens, C)
        """
        B, N, C = features.shape
        H = W = int(math.sqrt(N))
        x = features.view(B, H, W, C)
        grid_size = int(math.sqrt(num_tokens))
        pooled = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), (grid_size, grid_size))
        return pooled.permute(0, 2, 3, 1).reshape(B, num_tokens, C)
        
    def _llm_forward_compress(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fast_pixel_values=None,
        fast_token_idx=None,
        vp_overall_mask=None,
        prompt_masks=None,
        dense_frames_per_batch=None,
        sparse_frames_per_batch=None,
        # use_dense_sparse=False,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # image_flags = image_flags.squeeze(-1)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.model.extract_feature(pixel_values)
        # pixel_values: (64,3,448,448)
        # vit_embeds: (64,256,2048)， 每帧, 256个tokens, 每个token embedding是2048维
        vit_embeds = vit_embeds.to(input_embeds.dtype)
        
        # print(vit_embeds.shape)
        # if 32 < vit_embeds.shape[0] < 50:
        #     print(vit_embeds.shape)
            
        if dense_frames_per_batch is not None:
            assert sparse_frames_per_batch is not None
            processed_vit_embeds = {} # dict (key-value)
            
            for batch_idx in range(len(dense_frames_per_batch)):
                dense_count = dense_frames_per_batch[batch_idx]
                sparse_count = sparse_frames_per_batch[batch_idx]
                
                if len(dense_count) == 0 and len(sparse_count) == 0: # 图像
                    continue
            
                assert len(dense_count) == 4, f"Expected 4 dense frames, got {len(dense_count)}"
                
                if batch_idx == 0:
                    offset = 0
                else:
                    if len(dense_frames_per_batch[0]) == 0 and len(sparse_frames_per_batch[0]) == 0:
                        # 第一项是图像 (异常)
                        video_frames_count = len(dense_count) + len(sparse_count)
                        offset = vit_embeds.shape[0] - video_frames_count
                    else:
                        # 第一项是视频 (正常)
                        assert dense_frames_per_batch[0] is not None and sparse_frames_per_batch[0] is not None
                        bs = len(dense_frames_per_batch[0]) + len(sparse_frames_per_batch[0])
                        offset = bs
                
                for frame_idx in dense_count:
                    real_frame_idx = offset + frame_idx
                    assert real_frame_idx < len(vit_embeds)
                    processed_vit_embeds[real_frame_idx] = vit_embeds[real_frame_idx]

                for frame_idx in sparse_count:
                    real_frame_idx = offset + frame_idx
                    assert real_frame_idx < len(vit_embeds)
                    sparse_frame_features = vit_embeds[real_frame_idx]
                    pooled_features = self.grid_pool_features(sparse_frame_features.unsqueeze(0), num_tokens=4).squeeze(0)
                    processed_vit_embeds[real_frame_idx] = pooled_features
            
            if processed_vit_embeds:
                sorted_keys = sorted(processed_vit_embeds.keys())
                video_vit_embeds = torch.cat([processed_vit_embeds[idx] for idx in sorted_keys], dim=0)
        else:
            video_vit_embeds = None
                
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)

        assert selected.sum() != 0

        if video_vit_embeds is None:
            assert False
            # 纯图像batch，使用原始vit_embeds
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        elif selected.sum() == len(video_vit_embeds):
            # 纯视频batch，使用video_vit_embeds
            input_embeds[selected] = video_vit_embeds.reshape(-1, C).to(input_embeds.device)
        else:
            assert False
            if len(dense_frames_per_batch[0]) == 0 and len(sparse_frames_per_batch[0]) == 0:
                # 视频在后面，图像在前面
                n_frames_video = len(dense_frames_per_batch[1]) + len(sparse_frames_per_batch[1])
                image_vit_embeds = vit_embeds[:-n_frames_video].to(input_embeds.device)
                n_video = len(video_vit_embeds)
                input_embeds[selected][:-n_video] = image_vit_embeds.reshape(-1, C).to(input_embeds.device)
                input_embeds[selected][-n_video:] = video_vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                # 视频在前面，图像在后面
                assert len(dense_frames_per_batch[1]) == 0 and len(sparse_frames_per_batch[1]) == 0
                n_frames_video = len(dense_frames_per_batch[0]) + len(sparse_frames_per_batch[0])
                image_vit_embeds = vit_embeds[n_frames_video:].to(input_embeds.device)
                n_video = len(video_vit_embeds)
                input_embeds[selected][:n_video] = video_vit_embeds.reshape(-1, C).to(input_embeds.device)
                input_embeds[selected][n_video:] = image_vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def _llm_forward_compress_pseudo(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fast_pixel_values=None,
        fast_token_idx=None,
        vp_overall_mask=None,
        prompt_masks=None,
        dense_frames_per_batch=None,
        sparse_frames_per_batch=None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)
            
        if dense_frames_per_batch is not None:
            assert sparse_frames_per_batch is not None
            processed_vit_embeds = {}
            
            for batch_idx in range(len(dense_frames_per_batch)):
                dense_count = dense_frames_per_batch[batch_idx]
                sparse_count = sparse_frames_per_batch[batch_idx]

                if len(dense_count) == 0 and len(sparse_count) == 0:
                    assert False
                    continue
            
                assert len(dense_count) == 4, f"Expected 4 dense frames, got {len(dense_count)}"
                
                if batch_idx == 0:
                    offset = 0
                else:
                    if len(dense_frames_per_batch[0]) == 0 and len(sparse_frames_per_batch[0]) == 0:
                        assert False
                        video_frames_count = len(dense_count) + len(sparse_count)
                        offset = vit_embeds.shape[0] - video_frames_count
                    else:
                        assert dense_frames_per_batch[0] is not None and sparse_frames_per_batch[0] is not None
                        bs = len(dense_frames_per_batch[0]) + len(sparse_frames_per_batch[0])
                        offset = bs
                
                dense_keep_indices = torch.randperm(256)[:225]
                for frame_idx in dense_count:
                    real_frame_idx = offset + frame_idx
                    assert real_frame_idx < len(vit_embeds)
                    dense_frame_features = vit_embeds[real_frame_idx]
                    processed_vit_embeds[real_frame_idx] = dense_frame_features[dense_keep_indices]

                sparse_keep_indices = torch.randperm(256)[:50]
                for frame_idx in sparse_count:
                    real_frame_idx = offset + frame_idx
                    assert real_frame_idx < len(vit_embeds)
                    sparse_frame_features = vit_embeds[real_frame_idx]
                    processed_vit_embeds[real_frame_idx] = sparse_frame_features[sparse_keep_indices]
            
            if processed_vit_embeds:
                sorted_keys = sorted(processed_vit_embeds.keys())
                video_vit_embeds = torch.cat([processed_vit_embeds[idx] for idx in sorted_keys], dim=0)
        else:
            assert False
            video_vit_embeds = None
        del vit_embeds, processed_vit_embeds
        
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)

        assert selected.sum() != 0

        if video_vit_embeds is None:
            assert False
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        elif selected.sum() == len(video_vit_embeds):
            input_embeds[selected] = video_vit_embeds.reshape(-1, C).to(input_embeds.device)
        else:
            print(f"video_vit_embeds.shape={video_vit_embeds.shape}, selected.sum()={selected.sum()}")
            assert False
            if len(dense_frames_per_batch[0]) == 0 and len(sparse_frames_per_batch[0]) == 0 and len(middle_frames_per_batch[0]) == 0:
                n_frames_video = len(dense_frames_per_batch[1]) + len(sparse_frames_per_batch[1]) + len(middle_frames_per_batch[1])
                image_vit_embeds = vit_embeds[:-n_frames_video].to(input_embeds.device)
                n_video = len(video_vit_embeds)
                input_embeds[selected][:-n_video] = image_vit_embeds.reshape(-1, C).to(input_embeds.device)
                input_embeds[selected][-n_video:] = video_vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                assert len(dense_frames_per_batch[1]) == 0 and len(sparse_frames_per_batch[1]) == 0 and len(middle_frames_per_batch[1]) == 0
                n_frames_video = len(dense_frames_per_batch[0]) + len(sparse_frames_per_batch[0]) + len(middle_frames_per_batch[0])
                image_vit_embeds = vit_embeds[n_frames_video:].to(input_embeds.device)
                n_video = len(video_vit_embeds)
                input_embeds[selected][:n_video] = video_vit_embeds.reshape(-1, C).to(input_embeds.device)
                input_embeds[selected][n_video:] = image_vit_embeds.reshape(-1, C).to(input_embeds.device)
        del video_vit_embeds
        
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values, # 无
            use_cache=use_cache, # 无
            output_attentions=output_attentions, # 无
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict, # 无
        )
        logits = outputs.logits
        del input_embeds

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    
    def _llm_forward_mask_token(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fast_pixel_values=None,
        fast_token_idx=None,
        vp_overall_mask=None,
        prompt_masks=None,
        data_type=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        if fast_pixel_values is not None:
            assert False
            n_fast_images = fast_pixel_values.shape[0]
            whole_pixel_values = torch.cat([fast_pixel_values, pixel_values], dim=0)
            vit_embeds = self.model.extract_feature(whole_pixel_values)
            vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
            fast_vit_embeds = vit_embeds[:n_fast_images]  # (n_fast_images, hw, c)
            _size = int(fast_vit_embeds.shape[1] ** 0.5)
            fast_vit_embeds = fast_vit_embeds.reshape(fast_vit_embeds.shape[0], _size, _size, fast_vit_embeds.shape[-1])
            # pooling
            fast_vit_embeds = fast_vit_embeds.permute(0, 3, 1, 2)  # (n_fast_images, c, h, w)
            fast_vit_embeds = self.fast_pool(fast_vit_embeds).flatten(2)  # (n_fast_images, c, hw)
            fast_vit_embeds = fast_vit_embeds.permute(0, 2, 1)
            vit_embeds = vit_embeds[n_fast_images:]
        else:
            vit_embeds = self.model.extract_feature(pixel_values)
            
            if data_type == 'image':
                total_images = pixel_values.shape[0]  # 例如20
                patches_per_image = 256
                feature_dim = vit_embeds.shape[-1]
                mask_ratio = 0.75
                num_masked = int(patches_per_image * mask_ratio)  # 192
                
                vit_embeds = vit_embeds.view(total_images, patches_per_image, feature_dim)
                
                keep_indices_list = []
                if self.training:
                    masked_vit_embeds = []
                    for img_idx in range(total_images):
                        # 每张图都随机sample不同的keep_indices
                        keep_indices = torch.randperm(patches_per_image)[num_masked:]
                        image_embeds = vit_embeds[img_idx]
                        
                        image_embeds = image_embeds[keep_indices, :]
                        masked_vit_embeds.append(image_embeds)
                        keep_indices_list.append(keep_indices)
                    
                    vit_embeds = torch.stack(masked_vit_embeds, dim=0)
                
                vit_embeds = vit_embeds.view(-1, patches_per_image - num_masked, feature_dim)
                vit_embeds = vit_embeds.to(input_embeds.dtype) # (20,64,4096)
                mask_embeds = self.mask_tokens.repeat(2, num_masked, 1).view(-1, feature_dim)
            
            else:  # 正常的视频
                assert data_type == 'video'
                batch_size = 2
                frames_per_video = 10
                patches_per_frame = 256
                feature_dim = vit_embeds.shape[-1]
                mask_ratio = 0.75
                num_masked = int(patches_per_frame * mask_ratio) # 192
                
                vit_embeds = vit_embeds.view(batch_size, frames_per_video, patches_per_frame, feature_dim)
                
                keep_indices_list = []
                if self.training:
                    masked_vit_embeds = []
                    for batch_idx in range(batch_size):
                        keep_indices = torch.randperm(patches_per_frame)[num_masked:]
                        video_embeds = vit_embeds[batch_idx]
                        
                        video_embeds = video_embeds[:, keep_indices, :]
                        masked_vit_embeds.append(video_embeds)
                        keep_indices_list.append(keep_indices)
                    
                    vit_embeds = torch.stack(masked_vit_embeds, dim=0)

                vit_embeds = vit_embeds.view(-1, patches_per_frame - num_masked, feature_dim)
                vit_embeds = vit_embeds.to(input_embeds.dtype)
                mask_embeds = self.mask_tokens.repeat(2, num_masked, 1).view(-1, feature_dim)
            fast_vit_embeds = None

        vit_embeds = vit_embeds[image_flags == 1]
        # vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        if vp_overall_mask is not None and prompt_masks is not None:
            assert False
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask] # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1
            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)
        selected_mask = (input_ids == self.model.mask_token_id)

        if vp_embeds is None:
            try:
                input_embeds[selected] = vit_embeds.reshape(-1, C)
                # new
                input_embeds[selected_mask] = mask_embeds.reshape(-1, C)
                
            except Exception as e:
                assert False
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vit_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vit_embeds) + 1
                    vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vit_embeds[:n_token] 
        else:
            assert False
            try:
                input_embeds[selected] = vp_embeds.reshape(-1, C)
            except Exception as e:
                vp_embeds = vp_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vp_embeds[:n_token]

        if fast_vit_embeds is not None:
            assert False
            selected = (input_ids == fast_token_idx)
            selected_tot = selected.sum().item()
            if selected_tot > fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]:
                assert selected_tot % (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]) == 0
                repeat_times = selected_tot / (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1])
                fast_vit_embeds = fast_vit_embeds.repeat(int(repeat_times), 1, 1)
            try:
                input_embeds[selected] = fast_vit_embeds.reshape(-1, C)
            except Exception as e:
                fast_vit_embeds = fast_vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[fast_selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'fast_vit_embeds.shape={fast_vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = fast_vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        
        total_mae_loss = 0.0
        
        if data_type == 'image':
            num_images = 0
            total_images = pixel_values.shape[0]
            patches_per_image = 256
            num_masked = int(patches_per_image * 0.75)  # 192
            
            for img_idx in range(total_images):
                keep_indices = keep_indices_list[img_idx]
                token_mask = torch.ones(1, patches_per_image, device=vit_embeds.device)
                token_mask[:, keep_indices] = 0
                
                gt_img = pixel_values[img_idx, :, :, :].unsqueeze(0)  # (1,3,448,448)
                output_x, mae_loss = self.mae_decoder(
                    vit_embeds[img_idx, :, :].unsqueeze(0), 
                    self.mask_tokens.repeat(1, num_masked, 1),
                    target_imgs=gt_img,
                    mask=token_mask
                )
                total_mae_loss += mae_loss
                num_images += 1
            
            avg_mae_loss = total_mae_loss / num_images if num_images > 0 else 0.0
    
        else:  # 正常的视频
            assert data_type == 'video'
            num_frames = 0
            # vit_embeds: (2*10, 64, 4096)
            # mask_embeds: (2*192, 4096)
            for batch_idx in range(B):
                keep_indices = keep_indices_list[batch_idx]
                token_mask = torch.ones(1, 256, device=vit_embeds.device)
                token_mask[:, keep_indices] = 0

                for frame_idx in range(frames_per_video):
                    real_frame_idx = batch_idx * frames_per_video + frame_idx
                    gt_img = pixel_values[real_frame_idx, :, :, :].unsqueeze(0)  # (1,3,448,448)
                    output_x, mae_loss = self.mae_decoder(
                        vit_embeds[real_frame_idx, :, :].unsqueeze(0), 
                        self.mask_tokens.repeat(1, 192, 1),
                        target_imgs=gt_img,
                        mask=token_mask
                    )
                    total_mae_loss += mae_loss
                    num_frames += 1
            
            avg_mae_loss = total_mae_loss / num_frames if num_frames > 0 else 0.0
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)
            
            loss = lm_loss + avg_mae_loss
            # print(f"lm_loss: {lm_loss}, avg_mae_loss: {avg_mae_loss}")

        if not return_dict:
            assert False
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fast_token_idx=None,
        fast_pixel_values=None,
        prompt_masks=None,
        vp_overall_mask=None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.model.device
        assert self.model.img_context_token_id is not None

        if fast_pixel_values is not None:
            assert fast_token_idx is not None
            if type(fast_pixel_values) is list or fast_pixel_values.ndim == 5:
                if type(fast_pixel_values) is list:
                    fast_pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in fast_pixel_values
                    ]
                # b*n, c, h, w
                fast_pixel_values = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in fast_pixel_values], dim=0)

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    # b*n, c, h, w
                    pixel_values = torch.cat(
                        [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)

                if fast_pixel_values is not None:
                    n_fast_images = fast_pixel_values.shape[0]
                    whole_pixel_values = torch.cat([fast_pixel_values, pixel_values], dim=0)
                    vit_embeds = self.model.extract_feature(whole_pixel_values.to(device))
                    # vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
                    fast_vit_embeds = vit_embeds[:n_fast_images]  # (n_fast_images, hw, c)
                    _size = int(fast_vit_embeds.shape[1] ** 0.5)
                    fast_vit_embeds = fast_vit_embeds.reshape(fast_vit_embeds.shape[0], _size, _size,
                                                              fast_vit_embeds.shape[-1])
                    # pooling
                    fast_vit_embeds = fast_vit_embeds.permute(0, 3, 1, 2)  # (n_fast_images, c, h, w)
                    fast_vit_embeds = self.fast_pool(fast_vit_embeds).flatten(2)  # (n_fast_images, c, hw)
                    fast_vit_embeds = fast_vit_embeds.permute(0, 2, 1)
                    vit_embeds = vit_embeds[n_fast_images:]
                else:
                    fast_vit_embeds = None
                    vit_embeds = self.model.extract_feature(pixel_values.to(device))
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]
            
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            if vp_overall_mask is not None and prompt_masks is not None:
                vp_embeds = []
                vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
                prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

                vp_overall_mask = vp_overall_mask[image_flags == 1]
                overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

                i_vp_img = 0
                for i_img in range(len(vit_embeds)):
                    vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                    if vp_overall_mask[i_img]:
                        tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                        objects_prompt_masks = prompt_masks[i_vp_img]
                        n_obj = len(objects_prompt_masks)
                        tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                        objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                        vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                        i_vp_img += 1
                vp_embeds = torch.cat(vp_embeds, dim=0)
            else:
                vp_embeds = None

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            if vp_embeds is None:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                    print("Shape mismatch, selected is {}, vp embeds is {} !!!"\
                          .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
                    min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
                    input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
                else:
                    input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

            if fast_vit_embeds is not None:
                selected = (input_ids == fast_token_idx)
                # FIXME, add repeat.
                assert selected.sum() != 0
                input_embeds[selected] = fast_vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def state_dict(self, *args, **kwargs):
        if self.transfer_to_hf:
            state_dict = super(InternVL_V1_5, self).state_dict(*args, **kwargs)
            return state_dict
        else:
            return super().state_dict(*args, **kwargs)

    # def _merge_outputs(self, all_outputs):
    #     valid_outputs = [out for out in all_outputs if out is not None]

    #     assert len(valid_outputs) > 1

    #     total_loss = 0
    #     for output in valid_outputs:
    #         if hasattr(output, 'loss') and output.loss is not None:
    #             total_loss += output.loss
    #     avg_loss = total_loss / len(valid_outputs) if valid_outputs else None

    #     logits_list = []
    #     for output in valid_outputs:
    #         if hasattr(output, 'logits') and output.logits is not None:
    #             logits_list.append(output.logits)
    #     merged_logits = torch.cat(logits_list, dim=0) if logits_list else None

    #     hidden_states_list = []
    #     if hasattr(valid_outputs[0], 'hidden_states') and valid_outputs[0].hidden_states is not None:
    #         num_layers = len(valid_outputs[0].hidden_states)
    #         merged_hidden_states = []
    #         for layer_idx in range(num_layers):
    #             layer_states = []
    #             for output in valid_outputs:
    #                 if hasattr(output, 'hidden_states') and output.hidden_states is not None:
    #                     layer_states.append(output.hidden_states[layer_idx])
    #             if layer_states:
    #                 merged_hidden_states.append(torch.cat(layer_states, dim=0))
    #         hidden_states_list = tuple(merged_hidden_states)

    #     return CausalLMOutputWithPast(
    #         loss=avg_loss,
    #         logits=merged_logits,
    #         hidden_states=hidden_states_list,
    #     )

    
    def _llm_forward_compress_pseudo_img_video(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dense_frames_per_batch=None,
        sparse_frames_per_batch=None,
        img_video_combine=False,
        image_indices=None,
        video_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        assert img_video_combine 
        assert image_indices is not None and video_indices is not None
        
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)
            
        processed_vit_embeds = {}
        for frame_idx in range(len(vit_embeds)):
            processed_vit_embeds[frame_idx] = vit_embeds[frame_idx]
        
        # Video部分
        assert len(video_indices) == 1
        video_batch_idx = video_indices[0]
            
        if video_batch_idx == 0:
            video_frame_offset = 0  # Video在前
        else:
            video_frame_offset = (pixel_values.shape[0] - 7)  # Video在后
        
        dense_count = dense_frames_per_batch[video_batch_idx]
        sparse_count = sparse_frames_per_batch[video_batch_idx]

        dense_keep_indices = torch.randperm(256)[:225]
        for frame_idx in dense_count:
            real_frame_idx = video_frame_offset + frame_idx
            assert real_frame_idx < len(vit_embeds)
            dense_frame_features = vit_embeds[real_frame_idx]
            processed_vit_embeds[real_frame_idx] = dense_frame_features[dense_keep_indices]
        del dense_keep_indices, dense_frame_features

        sparse_keep_indices = torch.randperm(256)[:50]
        for frame_idx in sparse_count:
            real_frame_idx = video_frame_offset + frame_idx
            assert real_frame_idx < len(vit_embeds)
            sparse_frame_features = vit_embeds[real_frame_idx]
            processed_vit_embeds[real_frame_idx] = sparse_frame_features[sparse_keep_indices]
        del sparse_keep_indices, sparse_frame_features
        
        # Image部分
        assert len(image_indices) == 1
        image_batch_idx = image_indices[0]
        
        del vit_embeds, dense_count, sparse_count, video_batch_idx, image_batch_idx, video_frame_offset
        
        sorted_keys = sorted(processed_vit_embeds.keys())
        combined_vit_embeds = torch.cat([processed_vit_embeds[idx] for idx in sorted_keys], dim=0)
        del processed_vit_embeds, sorted_keys

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)

        assert selected.sum() != 0
        input_embeds[selected] = combined_vit_embeds.reshape(-1, C).to(input_embeds.device)
        del combined_vit_embeds, selected

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        del input_embeds
        
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            del shift_logits, shift_labels, loss_fct

        if not return_dict:
            assert False
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
    def _llm_forward_temporal(
        self,
        query_embeds: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fast_pixel_values=None,
        fast_token_idx=None,
        vp_overall_mask=None,
        prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        
        # pdb.set_trace()
        B, N, C = input_embeds.shape

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.query_token_id)
        input_embeds = input_embeds.reshape(B * N, C)
        input_embeds[selected] = query_embeds.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        # pdb.set_trace()
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # vit_embeds=vit_embeds,
        )
        
        
    def _llm_forward_temporal_pooled(
        self,
        pixel_values: torch.FloatTensor = None,
        pooled_features: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fast_pixel_values=None,
        fast_token_idx=None,
        vp_overall_mask=None,
        prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        # vit_embeds = self.extract_feature(pixel_values)
        
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        input_ids = input_ids.reshape(B * N)
        # pdb.set_trace()
        selected = (input_ids == self.model.img_context_token_id)
        input_embeds = input_embeds.reshape(B * N, C)
        input_embeds[selected] = pooled_features.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def _llm_forward_shared(
        self,
        pixel_values: torch.FloatTensor = None,
        pooled_features: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        统一的LLM前向传播，支持spatial和temporal任务
        - spatial: 传入pixel_values，函数内部提取视觉特征
        - temporal: 传入pooled_features，直接使用预处理特征
        """
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        
        if pooled_features is not None:
            # Temporal
            vit_embeds = pooled_features
        elif pixel_values is not None:
            # Spatial
            if image_flags is not None:
                image_flags = image_flags.squeeze(-1)
                vit_embeds = self.model.extract_feature(pixel_values)
                vit_embeds = vit_embeds.to(input_embeds.dtype)
                vit_embeds = vit_embeds[image_flags == 1]
            else:
                vit_embeds = self.model.extract_feature(pixel_values)
                vit_embeds = vit_embeds.to(input_embeds.dtype)
        else:
            raise ValueError("Either pixel_values or pooled_features must be provided")
        
        input_ids_flat = input_ids.reshape(B * N)
        input_embeds_flat = input_embeds.reshape(B * N, C)
        
        selected = (input_ids_flat == self.model.img_context_token_id)
        
        try:
            input_embeds_flat[selected] = vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'Warning: {e}, input_embeds[selected].shape={input_embeds_flat[selected].shape}, '
                f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            if n_token > len(vit_embeds):
                print(f"Error: {n_token} image tokens in text but only {len(vit_embeds)} vit embeds")
                expand_ratio = n_token // len(vit_embeds) + 1
                vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)
            input_embeds_flat[selected] = vit_embeds[:n_token]
        
        del vit_embeds
        input_embeds = input_embeds_flat.reshape(B, N, C)
        
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        logits = outputs.logits
        del input_embeds
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )