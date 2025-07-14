import random
from torch.utils.data import Sampler

class TaskGroupedSampler(Sampler):
    """确保batch内任务类型一致的采样器"""
    
    def __init__(self, dataset, batch_size=1, shuffle=True, seed=None, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        
        # 显式定义任务类型
        self.spatial_dataset_types = {
            'VideoReVOSDataset_box',
            'VideoMeVISDataset_box', 
            'VideoRefYoutubeVOSDataset_box',
            'ReferSegmDataset_box'
        }
        
        self.temporal_dataset_types = {
            'VideoCharadesTemporalDataset_clip',
            'VideoDidemoTemporalDataset_clip'
        }
        
        # 分析数据集，按任务类型分组索引
        self.spatial_indices = []
        self.temporal_indices = []
        self._group_indices()
        
    def _group_indices(self):
        """按任务类型分组索引"""
        if hasattr(self.dataset, 'datasets'):  # ConcatDataset
            current_idx = 0
            for dataset in self.dataset.datasets:
                dataset_type = type(dataset).__name__
                dataset_len = len(dataset)
                indices = list(range(current_idx, current_idx + dataset_len))
                
                if dataset_type in self.temporal_dataset_types:
                    self.temporal_indices.extend(indices)
                    print(f"Temporal dataset: {dataset_type}, indices: {len(indices)}")
                elif dataset_type in self.spatial_dataset_types:
                    self.spatial_indices.extend(indices)
                    print(f"Spatial dataset: {dataset_type}, indices: {len(indices)}")
                else:
                    raise ValueError(f"Unknown dataset type: {dataset_type}")
                    
                current_idx += dataset_len
        else:
            indices = list(range(len(self.dataset)))
            dataset_type = type(self.dataset).__name__
            if dataset_type in self.temporal_dataset_types:
                self.temporal_indices = indices
            elif dataset_type in self.spatial_dataset_types:
                self.spatial_indices = indices
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        print(f"Total spatial indices: {len(self.spatial_indices)}")
        print(f"Total temporal indices: {len(self.temporal_indices)}")
    
    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)
            
        # 打乱各自的索引
        if self.shuffle:
            random.shuffle(self.spatial_indices)
            random.shuffle(self.temporal_indices)
        
        task_grouped_indices = []
        
        # 处理两种类型的indices
        for indices in [self.spatial_indices, self.temporal_indices]:
            num_samples = len(indices)
            if num_samples == 0:
                continue
                
            # 处理完整batch
            num_full_batches = num_samples // self.batch_size
            for i in range(num_full_batches):
                start_idx = i * self.batch_size
                batch = indices[start_idx:start_idx + self.batch_size]
                task_grouped_indices.extend(batch)
            
            # 处理最后一个不完整batch
            if not self.drop_last and num_samples % self.batch_size != 0:
                last_batch = indices[num_full_batches * self.batch_size:]
                # 填充到完整batch大小
                if len(last_batch) < self.batch_size:
                    last_batch.extend(random.choices(indices, k=self.batch_size - len(last_batch)))
                task_grouped_indices.extend(last_batch)
        
        # 在batch级别打乱
        if self.shuffle:
            num_batches = len(task_grouped_indices) // self.batch_size
            batch_indices = list(range(num_batches))
            random.shuffle(batch_indices)
            
            shuffled_indices = []
            for batch_idx in batch_indices:
                start = batch_idx * self.batch_size
                end = start + self.batch_size
                shuffled_indices.extend(task_grouped_indices[start:end])
            
            task_grouped_indices = shuffled_indices
        
        print(f"Generated {len(task_grouped_indices)} indices in task-grouped order")
        return iter(task_grouped_indices)
    
    def __len__(self):
        total_full_batches = 0
        for indices in [self.spatial_indices, self.temporal_indices]:
            num_samples = len(indices)
            num_batches = num_samples // self.batch_size
            if not self.drop_last and num_samples % self.batch_size != 0:
                num_batches += 1
            total_full_batches += num_batches
        return total_full_batches * self.batch_size
    
    def set_epoch(self, epoch):
        """为分布式训练设置epoch"""
        if self.seed is not None:
            random.seed(self.seed + epoch)