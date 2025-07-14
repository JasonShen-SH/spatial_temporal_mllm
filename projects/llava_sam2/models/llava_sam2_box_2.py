from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from third_parts.mmdet.models.losses import CrossEntropyLoss

from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict

from .lisa import LisaModel

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from transformers import GenerationConfig
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

import numpy as np

from .internvl import InternVL_Slowfast
from .utils import dynamic_preprocess

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from pycocotools import mask as _mask

from types import MethodType

from xtuner.model.utils import guess_load_checkpoint

from mmcv.ops import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness
import pdb

class VideoLLaVASAMModel(LisaModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 torch_dtype=torch.bfloat16,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 use_fast_supervision=False,
                 # for inference
                 phi3=True,
                 template=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 # for inference large model
                 split_model=False,
                 # ext
                 preprocessor=None,
                 # bs
                 bs:int=0,
                 ):
        super(LisaModel, self).__init__()
        self.split_model = split_model
        if split_model:
            mllm.model_split = split_model
        if special_tokens is None:
            special_tokens = ['[SEG]']
        self.special_tokens = special_tokens
        if 'special_tokens' not in mllm.keys():
            mllm.special_tokens = special_tokens
        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type
        self.fast_pool = fast_pool
        self.fast_pool_size = fast_pool_size
        if hasattr(self.mllm, '_post_init'):
            self.mllm._post_init(
                fast_pool_size=self.fast_pool_size,
                fast_pool=self.fast_pool
            )
        else:
            print("No _post_init() in mllm !!!")

        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens()

        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            elif self.arch_type == 'qwen': 
                self.mllm.model.model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.get_output_embeddings().weight.requires_grad_(True)
            elif self.arch_type == 'llava':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)

        if self.arch_type == 'intern_vl':
            in_dim = self.mllm.model.config.llm_config.hidden_size
        elif self.arch_type == 'qwen':
            in_dim = self.mllm.model.config.hidden_size
        elif self.arch_type == 'llava':
            # for llava, the hidden size is in language model
            in_dim = self.mllm.model.language_model.config.hidden_size
        
        if use_fast_supervision:
            self.loss_exists = BUILDER.build(dict(
                type=CrossEntropyLoss,
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0)
            )

        self.torch_dtype = torch_dtype

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        if fast_pool: # False
            self.fast_token_idx = self.tokenizer("<FAST_IMG_CONTEXT>", add_special_tokens=False).input_ids[0]
        else:
            self.fast_token_idx = None
        self.use_fast_supervision = use_fast_supervision

        self.phi3 = phi3
        self.template = template

        if preprocessor is None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = BUILDER.build(preprocessor)

        self.bs = bs
        
        embed_dim = self.mllm.model.language_model.base_model.model.get_input_embeddings().weight.shape[1]
        self.embed_dim = embed_dim
        
    def _merge_lora(self):
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        return state_dict

    def activation_checkpointing_disable(self):
        if self.arch_type == 'qwen':
            self.mllm.model.model.gradient_checkpointing_disable()
        else:
            self.mllm.model.language_model.gradient_checkpointing_disable()


    def _add_special_tokens(self):
        special_tokens = self.special_tokens
        _num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)

    def state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        from collections import OrderedDict
        to_return = OrderedDict()
        
        # Step 1. visual_encoder
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
            raise NotImplementedError
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
            raise NotImplementedError
        
        # Step 2. LLM
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
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
            raise NotImplementedError
        
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.multi_modal_projector.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k or 'output' in k and 'sam2_model' not in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'embed_tokens.weight' in k or 'tok_embeddings' in k})
        
        # Step 4. Temporal Components
        temporal_components = [
            'query_to_frame.',
        ]
        for component in temporal_components:
            to_return.update(
                {k: v for k, v in state_dict.items() if component in k}
            )
            
        return to_return

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        assert self.bs > 0
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        fast_exists = data.pop('fast_exists', None)
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)
              
        if gt_masks is None:
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            raise NotImplementedError

        if self.use_fast_supervision and fast_exists is not None:
            fast_flag = input_ids == self.fast_token_idx
            fast_tokens = output.hidden_states[-1][fast_flag]
            exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
            gt_exists = torch.cat(fast_exists)
            loss_exists = self.loss_exists(exists_logit, gt_exists)
        else:
            loss_exists = None

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        if loss_exists is not None:
            loss_dict['loss_exists'] = loss_exists
            
        return loss_dict

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def genetate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        if len(pred_embeddings_list) == len(frames_per_batch): # 它们的length就是batch_size=2
            success = True
        else:
            success = False
            print("len(pred_embeddings_list):{} is not equal to len(frames_per_batch):{} !!!".format(len(pred_embeddings_list), len(frames_per_batch)))
         # pred_embeddings_list: (len:batch_size, per:(NUMBER_OF_SEG_TOKENS_PER_SAMPLE,256)) e.g.([(8*256), (8*256)])
         # frames_per_batch: (len:batch_size, per:(NUMBER_OF_FRAMES_PER_SAMPLE) e.g.([5, 5])
        
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch): # batch_size(2)
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums  #乘以frame_nums  
        return pred_embeddings_list_video, success

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []
        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def process_video_gt_bboxes(self, gt_bboxes, frames_per_batch):
        gt_bboxes_video = []
        assert len(gt_bboxes) == len(frames_per_batch)
        for gt_bboxes_batch, frames_num in zip(gt_bboxes, frames_per_batch):
            N, H = gt_bboxes_batch.shape
            assert N % frames_num == 0
            gt_bboxes_batch = gt_bboxes_batch.reshape(
                N // frames_num, frames_num, H)
            for i in range(frames_num):
                gt_bboxes_video.append(gt_bboxes_batch[:, i])
        return gt_bboxes_video
    

    def preparing_for_generation(self, metainfo, **kwargs):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['phi3_chat']
        if self.template is None:
            self.template = template
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True

        self.mllm.to(self.torch_dtype)
        self.text_hidden_fcs.to(self.torch_dtype)
        
        # for sam image processor
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        if 'max_dynamic_patch' in metainfo.keys():
            self.max_dynamic_patch = metainfo['max_dynamic_patch']
        else:
            self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size

        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        if self.preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            # self.preprocessor = BUILDER.build(self.preprocessor)

        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'

        # change phi3 prepare for generation fuction
        if self.phi3:
            self.mllm.model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.mllm.model.language_model)
        return

    def predict_video(self, pixel_values, text_prompts, **kwargs):
        ori_h, ori_w = kwargs['ori_height'], kwargs['ori_width']

        _input_ids = kwargs['input_ids']

        g_pixel_values = kwargs.pop('g_pixel_values', None)
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])

        fast_pixel_values = kwargs.pop('fast_pixel_values', None)
        if fast_pixel_values is None:
            fast_token_idx = None
        else:
            fast_token_idx = self.fast_token_idx

        predictions = []
        pred_masks = []
        is_exists_list = []
        for input_ids in _input_ids:
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            pixel_values = pixel_values.to(dtype=self.torch_dtype)
            if fast_pixel_values is not None:
                fast_pixel_values = fast_pixel_values.to(dtype=self.torch_dtype)
            mm_inputs = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'fast_pixel_values': fast_pixel_values,
                'fast_token_idx': fast_token_idx,
            }
            if kwargs.get('image_grid_thw', None) is not None:
                mm_inputs['image_grid_thw'] = kwargs['image_grid_thw']

            generate_output = self.mllm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            predict = self.tokenizer.decode(generate_output.sequences[0], skip_special_tokens=False).strip()

            predictions.append(predict)

            hidden_states = generate_output.hidden_states
            last_hidden_states = [item[-1][0] for item in hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            seg_hidden_states = get_seg_hidden_states(
                last_hidden_states, generate_output.sequences[0][:-1],
                seg_id=self.seg_token_idx
            )

            if len(seg_hidden_states) == 0:
                print("Warning, no [SEG] tokens !!!")
                pred_masks.append(torch.zeros((g_pixel_values.shape[0], ori_h, ori_w), dtype=torch.int))
                continue
            elif len(seg_hidden_states) > 1:
                print("Warning, {} [SEG] tokens !!!".format(len(seg_hidden_states)))
                seg_hidden_states = seg_hidden_states[:1]
            seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

            seg_hidden_states = seg_hidden_states.to(dtype=torch.float32)

            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            # TODO: change 5
            if len(pixel_values) < 5:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * pixel_values.shape[0])
            else:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * 5)
            pred_mask = F.interpolate(
                pred_mask,
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=False,
            )
            pred_mask = pred_mask[:, 0]
            pred_mask = pred_mask.sigmoid() > 0.5
            pred_mask = pred_mask.int()
            # supervision
            if self.use_fast_supervision and (input_ids == self.fast_token_idx).sum() > 0:
                fast_flag = input_ids.squeeze(0) == self.fast_token_idx
                len_out = generate_output.sequences[0][:-1].shape[0]
                fast_tokens = last_hidden_states[:-len_out][fast_flag].to(dtype=torch.float32)
                exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
                is_exists = exists_logit.squeeze(-1).sigmoid() > 0.5
                is_exists_list.append(is_exists)
                not_exists = torch.logical_not(is_exists)
                if torch.any(not_exists):
                    pred_mask[not_exists] = pred_mask[not_exists] * 0
            pred_masks.append(pred_mask)
            
        assert len(pred_masks) == len(text_prompts)
        ret_dict = {
            'prediction': predictions,
            'prediction_masks': [mask_to_rle(_item.cpu().numpy()) for _item in pred_masks],
        }
        if 'id' in kwargs.keys():
            ret_dict['id'] = kwargs['id']

        if len(is_exists_list) > 0:
            ret_dict['is_exists'] = is_exists_list
        return ret_dict


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs


class VideoLLaVASAMModel_zero3(VideoLLaVASAMModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 torch_dtype=torch.bfloat16,
                 special_tokens=['[SEG]', ],
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 arch_type='intern_vl',
                 # zero3
                 bs=1,
                 ):
        super(VideoLLaVASAMModel_zero3, self).__init__(
            mllm=mllm,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            special_tokens=special_tokens,
            loss_sample_points=loss_sample_points,
            num_points=num_points,
            # for slow fast arch
            fast_pool=fast_pool,
            fast_pool_size=fast_pool_size,
            arch_type=arch_type,
        )
        self.bs = bs
        print("special_tokens: ", self.special_tokens)

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_bboxes = torch.zeros((5, 4), dtype=torch.float, device=device) 
        gt_bboxes = [gt_bboxes] * self.bs  
        return g_pixel_values, frames_per_batch, gt_bboxes

    def forward(self, data, data_samples=None, mode='loss'):
        # input_ids: 输入tokens
        # labels: 只关注output tokens, 将input_tokens, BOS, SEP, EOS全部置为-100, 因为cross_entropy_loss针对-100是自动不计算loss的
        # attention_mask: 由于同个batch内, input_ids有长有短, 短的需要补齐到长的长度 (默认batch_size=2)
        # position_ids: tokens的位置编码 (0,1,2,3,..1575)
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_bboxes_areas = data.pop('bboxes_areas', None)
        gt_bboxes_expression = data.pop('bboxes_expression', None)
        gt_bboxes_points = data.pop('bboxes_points', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)
        
        del g_pixel_values, gt_bboxes_areas, gt_bboxes_expression, gt_bboxes_points, frames_per_batch
        
        torch.cuda.empty_cache()

        loss_dict = {
            'llm_loss': output.loss,
        }
        return loss_dict


###############################################################
# 只对temporal负责

from .temporal_grounding_model import (
    QueryToFrameProjection
)

class VideoLLaVASAMModel_temporal_direct(VideoLLaVASAMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.query_hidden_dim = 1024
        llm_dim = self.mllm.model.config.llm_config.hidden_size
        self.query_to_frame = QueryToFrameProjection(
            query_dim=self.query_hidden_dim,
            llm_dim=llm_dim,
        )
        
    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data.get('pixel_values', None)
        frame_positions = data.get('frame_positions', None)
        
        #############################################
        # 使用CLIP特征专属
        try:
            batch_size, num_frames = pixel_values.shape[:2]
        except:
            print("legth of pixel_values: ", len(pixel_values))
            print("element 1: ", pixel_values[0].shape)
            print("element 2: ", pixel_values[1].shape)
        if pixel_values is not None:
            encoded_features = pixel_values
            # frame_positions = torch.arange(num_frames, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)
            # encoded_features = self.temporal_encoding(encoded_features, frame_positions)
            frame_features = self.query_to_frame(encoded_features)
            data['pooled_features'] = frame_features
        #############################################
        
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_bboxes = data.pop('bboxes', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        
        del g_pixel_values, gt_bboxes, frames_per_batch, pixel_values
        
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)
        
        loss_dict = {
            'llm_loss': output.loss,
        }
        
        return loss_dict