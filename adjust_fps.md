```shell
由于temporal datasets官方提供的都是raw videos,
因此需要手动抽帧，然后提取clip features；
我统一设置fps=10

抽帧: try_other_fps/split_temporal_dataset_into_frames.py
提取clip features: try_other_fps/extract_clip_features.py
```