# 环境准备
```shell
基本要求
- Python 3.10
- PyTorch 2.6.0
- CUDA 12.4

推荐使用conda创建独立环境, 名为spatial
conda create -n spatial python=3.10 -y 
conda activate spatial
pip install -r requirements.txt
```

# 数据准备

下载命令如下

```shell
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/Dense-World/Sa2VA-Training

# 应该会自动开始下载大文件, 如果没有, 手动执行
cd Sa2VA-Training
git lfs pull

# 存储路径为: /home/volume_shared/share_datasets，请更改

# 也可以先下载到当前路径，然后软链接到/home/volume_shared/share_datasets (run.sh方式)
```

下载完成后, 只需要解压其中的
```shell
ref_seg_coco.zip, 
ref_seg_coco_g.zip，
ref_seg_coco_+.zip，
video_datas_mevis.zip，
video_datas_rvos.zip 
其余数据集无需解压
```

注意，由于ReVOS在上述链接中不完整，需要手动下载, 并与上述文件放置在同一位置

(上述内容，您那边可能已经下载好了，若没有请联系我，我随时重新上传)


最终期待的数据结构为:
```shell
/home/volume_shared/share_datasets/
├── refer_seg/
│   ├── refcoco/
│   ├── refcocog/
│   └── refcoco+/
├── video_datas/
│   ├── mevis/
│   ├── rvos/
│   └── ReVOS/
└── temporal/
    ├── Charades/
    └── didemo/
```

若有变动，请在配置文件projects/llava_sam2/configs/sa2va_4b_shared.py中同步修改存储路径 (8B模型: projects/llava_sam2/configs/sa2va_8b_shared.py)

所有重要的json files都已经放在special_jsons/下，结构如下：
```shell
special_jsons/
├── spatial/
│   ├── mevis/
│   ├── refytvos/
│   └── revos/
└── temporal/
    ├── Charades/
    └── DideMo/
```

此外，temporal dataset的预处理clip features需要放在clip_features/下，结构如下：
```shell
clip_features/
├── charades/
└── didemo/
**您需要将下载下来的charades和didemo的clip features, 手动移动至这个路径**

目前clip features的提取fps都是10，如果需要调整，请看adjust_fps.md
```

#  训练
```shell
# 4B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_4b_shared.py 8 --work-dir spatial_temporal_4b

# 8B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b_shared.py 8 --work-dir spatial_temporal_8b

默认每1000个steps保存一次，会自动将最早的ckpt覆盖，无需手动操作
```