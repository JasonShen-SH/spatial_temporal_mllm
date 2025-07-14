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

注意，由于ReVOS在上述链接中不完整，需要手动下载
[ReVOS](https://mailsjlueducn-my.sharepoint.com/personal/yancl9918_mails_jlu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyancl9918%5Fmails%5Fjlu%5Fedu%5Fcn%2FDocuments%2Fdataset%2Frevos%5Feccv%5Fdataset%2FReVOS&ga=1)，
并与上述文件放置在同一位置


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

若有变动，请在配置文件 projects/llava_sam2/configs/sa2va_4b_shared.py中同步修改存储路径

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
# 8卡训练

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_4b_shared.py 8 --work-dir spatial_temporal_4b

CUDA_VISIBLE_DEVICES=1 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_4b_shared.py 1 --work-dir spatial_temporal_4b
CUDA_VISIBLE_DEVICES=7 bash tools/debug.sh train projects/llava_sam2/configs/sa2va_4b_shared.py --work-dir debug5
```
```shell
默认参数(可修改)
1. 保存路径 spatial_temporal_4b
2. 每1000个steps保存一次，会自动将最早的ckpt覆盖，无需手动操作
```