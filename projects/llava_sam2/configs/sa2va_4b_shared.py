from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from peft import LoraConfig

# import sys
# sys.path.append('/home/rqshen/spatial_temporal_mllm')

from projects.llava_sam2.datasets import VideoReVOSDataset_box, VideoMeVISDataset_box, VideoRefYoutubeVOSDataset_box
from projects.llava_sam2.datasets import VideoCharadesTemporalDataset_clip, VideoDidemoTemporalDataset_clip
from projects.llava_sam2.datasets import ReferSegmDataset_box
from projects.llava_sam2.datasets import video_shared_collate_fn
from projects.llava_sam2.datasets import video_temporal_collate_fn, video_lisa_collate_fn
from projects.llava_sam2.models.internvl import InternVL_Slowfast
from projects.llava_sam2.models.preprocess.image_resize import DirectResize
from projects.llava_sam2.models.llava_shared import VideoLLaVA_Shared

from projects.llava_sam2.datasets.task_grouped_sampler import TaskGroupedSampler

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
path='OpenGVLab/InternVL2_5-4B'

template = "phi3_chat"
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

batch_size = 2
accumulative_counts = 4
dataloader_num_workers = 0
max_epochs = 10
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.05

save_steps = 1000
save_total_limit = 2

special_tokens = [
    # task
    '<SPATIAL>', '<TEMPORAL>',
    # Spatial grounding tokens
    '<box_sep>', '<no_box>', '<next>', '<single_img>',
    # Temporal grounding tokens
    '<frame_0>', '<frame_1>', '<frame_2>', '<frame_3>', '<frame_4>', 
    '<frame_5>', '<frame_6>', '<frame_7>', '<frame_8>', '<frame_9>'
]

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVA_Shared,
    special_tokens=special_tokens,
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'),
        special_tokens=special_tokens,
    ),
    tokenizer=tokenizer,
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

# Spatial grounding datasets (Video)

# ReVOS是手动下载的
data_root_revos = '/home/volume_shared/share_datasets/video_datas/ReVOS/'
video_revos_image_folder = data_root_revos + 'JPEGImages'
video_revos_expression_file = data_root_revos + 'meta_expressions_train_.json'
video_revos_box_file = 'special_jsons/spatial/revos/revos_bbox.json'

data_root_mevis = '/home/volume_shared/share_datasets/video_datas/mevis/train/'
video_mevis_image_folder = data_root_mevis + 'JPEGImages'
video_mevis_expression_file = data_root_mevis + 'meta_expressions.json'
video_mevis_box_file = 'special_jsons/spatial/mevis/mevis_bbox.json' 

data_root_refytvos = '/home/volume_shared/share_datasets/video_datas/rvos/'
video_refytvos_image_folder = data_root_refytvos + 'train/JPEGImages/'
video_refytvos_expression_file = 'special_jsons/spatial/refytvos/ref_ytvos_expressions.json'
video_refytvos_box_file = 'special_jsons/spatial/refytvos/ref_ytvos_bbox.json'

video_revos_dataset = dict(
    type=VideoReVOSDataset_box,
    image_folder=video_revos_image_folder,
    expression_file=video_revos_expression_file,
    box_file=video_revos_box_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=8,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=6,
)

video_mevis_dataset = dict(
    type=VideoMeVISDataset_box,
    image_folder=video_mevis_image_folder,
    expression_file=video_mevis_expression_file,
    box_file=video_mevis_box_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=6,
)

video_refytvos_dataset = dict(
    type=VideoRefYoutubeVOSDataset_box,
    image_folder=video_refytvos_image_folder,
    expression_file=video_refytvos_expression_file,
    box_file=video_refytvos_box_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=6,
)

# Spatial grounding datasets (Image)
refcoco_segm_dataset = dict(
    type=ReferSegmDataset_box,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='/home/volume_shared/share_datasets/refer_seg/refcoco',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

refcoco_plus_segm_dataset = dict(
    type=ReferSegmDataset_box,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='/home/volume_shared/share_datasets/refer_seg/refcoco+',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

refcocog_segm_dataset = dict(
    type=ReferSegmDataset_box,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='/home/volume_shared/share_datasets/refer_seg/refcocog',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(umd).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

# Temporal grounding datasets
data_root_charades = '/home/volume_shared/share_datasets/temporal/Charades/' # 5338/9848
video_charades_dataset = dict(
    type=VideoCharadesTemporalDataset_clip,
    image_folder=data_root_charades + 'frames_10', # fps=10,可更改
    expression_file='special_jsons/temporal/Charades/time_token.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    sampled_frames=100,
)

data_root_didemo = '/home/volume_shared/share_datasets/temporal/didemo/' # 8356/8395
video_didemo_dataset = dict(
    type=VideoDidemoTemporalDataset_clip,
    image_folder=data_root_didemo + 'video/frames', # fps=10,可更改
    expression_file='special_jsons/temporal/DideMo/time_token.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    sampled_frames=100,
)

# 下面，hirest和queryd共享videos路径

# data_root_hirest = '/home/volume_shared/share_datasets/hirest_queryd/other_videos/'
# video_hirest_dataset = dict(
#     type=VideoHirestTemporalDataset_clip,
#     image_folder=data_root_hirest + 'frames', # fps=3,可更改
#     expression_file='special_jsons/temporal/HiREST_grounding/time_token.json',
#     tokenizer=tokenizer,
#     template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=4,
#     special_tokens=special_tokens,
#     sampled_frames=64,
# )

# data_root_queryd = '/home/volume_shared/share_datasets/hirest_queryd/other_videos/'
# video_queryd_dataset = dict(
#     type=VideoQueryDTemporalDataset_clip,
#     image_folder=data_root_queryd + 'frames', # fps=3,可更改
#     expression_file='special_jsons/temporal/QueryD/time_token.json',
#     tokenizer=tokenizer,
#     template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=4,
#     special_tokens=special_tokens,
#     sampled_frames=64,
# )

train_dataset = dict(
    type=ConcatDataset, 
    datasets=[
        # Spatial grounding datasets
        video_revos_dataset,
        video_mevis_dataset, video_mevis_dataset,
        video_refytvos_dataset,
        refcoco_segm_dataset,
        refcoco_plus_segm_dataset,
        refcocog_segm_dataset,
        # Temporal grounding datasets
        video_charades_dataset,
        video_didemo_dataset,
        # video_hirest_dataset,
        # video_queryd_dataset,
    ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=TaskGroupedSampler,
        batch_size=batch_size * accumulative_counts,
        shuffle=True
    ),
    collate_fn=dict(type=video_shared_collate_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)