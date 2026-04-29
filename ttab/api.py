# -*- coding: utf-8 -*-
import itertools
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data.dataset import random_split
import random

State = List[torch.Tensor]
Gradient = List[torch.Tensor]
Parameters = List[torch.Tensor]
Loss = float
Quality = Mapping[str, float]


class Batch(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return len(self._x)

    def to(self, device) -> "Batch":
        return Batch(self._x.to(device), self._y.to(device))

    def __getitem__(self, index):
        return self._x[index], self._y[index]


class GroupBatch(object):
    def __init__(self, x, y, g):
        self._x = x
        self._y = y
        self._g = g

    def __len__(self) -> int:
        return len(self._x)

    def to(self, device) -> "Batch":
        return GroupBatch(self._x.to(device), self._y.to(device), self._g.to(device))

    def __getitem__(self, index):
        return self._x[index], self._y[index], self._g[index]


class Dataset:
    def random_split(self, fractions: List[float]) -> List["Dataset"]:
        pass

    def iterator(
        self, batch_size: int, shuffle: bool, repeat=True
    ) -> Iterable[Tuple[float, Batch]]:
        pass

    def __len__(self) -> int:
        pass


class PyTorchDataset(object):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        device: str,
        prepare_batch: Callable,
        num_classes: int,
    ):
        self._set = dataset
        self._device = device
        self._prepare_batch = prepare_batch
        self._num_classes = num_classes

    def __len__(self):
        return len(self._set)

    def replace_indices(
        self,
        indices_pattern: str = "original",
        new_indices: List[int] = None,
        random_seed: int = None,
    ) -> None:
        """Change the order of dataset indices in a particular pattern."""
        if indices_pattern == "original":
            pass
        elif indices_pattern == "random_shuffle":
            rng = np.random.default_rng(random_seed)
            rng.shuffle(self.dataset.indices)
        elif indices_pattern == "new":
            if new_indices is None:
                raise ValueError("new_indices should be specified.")
            self.dataset.update_indices(new_indices=new_indices)
        else:
            raise NotImplementedError

    def query_dataset_attr(self, attr_name: str) -> Any:
        return getattr(self._set, attr_name, None)

    @property
    def dataset(self):
        return self._set

    @property
    def num_classes(self):
        return self._num_classes

    def no_split(self) -> List[Dataset]:
        return [
            PyTorchDataset(
                dataset=self._set,
                device=self._device,
                prepare_batch=self._prepare_batch,
                num_classes=self._num_classes,
            )
        ]

    def random_split(self, fractions: List[float], seed: int = 0) -> List[Dataset]:
        lengths = [int(f * len(self._set)) for f in fractions]
        lengths[0] += len(self._set) - sum(lengths)
        return [
            PyTorchDataset(
                dataset=split,
                device=self._device,
                prepare_batch=self._prepare_batch,
                num_classes=self._num_classes,
            )
            for split in random_split(
                self._set, lengths, torch.Generator().manual_seed(seed)
            )
        ]

    def iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False,
        ref_num_data: Optional[int] = None,
        num_workers: int = 1,
        sampler: Optional[torch.utils.data.Sampler] = None,
        generator: Optional[torch.Generator] = None,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> Iterable[Tuple[int, float, Batch]]:
        # 新增，适配CTTA       
        if sampler is None and hasattr(self, 'lengths'):
            # # 使用域感知采样器
            # sampler = DomainAwareSampler(self.lengths, batch_size, shuffle)
            # shuffle = False  # sampler和shuffle不能同时使用
            # 使用域感知批量采样器
            batch_sampler = DomainAwareBatchSampler(
                lengths=self.lengths, 
                batch_size=batch_size, 
                shuffle=shuffle,
                drop_last=drop_last
            )
                
        _num_batch = 1 if not drop_last else 0
        if ref_num_data is None:
            num_batches = int(len(self) / batch_size + _num_batch)
        else:
            num_batches = int(ref_num_data / batch_size + _num_batch)
        if sampler is not None:
            shuffle = False

        # 注意：指定了sampler后就不需要指定batch_size, shuffle和drop_last了，否则sampler会被batch_sampler覆盖
        # loader = torch.utils.data.DataLoader(
        #     self._set,
        #     # batch_size=batch_size,
        #     shuffle=shuffle,
        #     pin_memory=pin_memory,
        #     # drop_last=drop_last,
        #     num_workers=num_workers,
        #     sampler=sampler,
        #     generator=generator,
        # )
        if sampler is None and hasattr(self, 'lengths'):
            loader = torch.utils.data.DataLoader(
                self._set,
                pin_memory=pin_memory,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                generator=generator,
            )
        else:
            loader = torch.utils.data.DataLoader(
                self._set,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                drop_last=drop_last,
                num_workers=num_workers,
                sampler=sampler,
                generator=generator,
            )

        step = 0
        for _ in itertools.count() if repeat else [0]:
            for i, batch in enumerate(loader):
                step += 1
                epoch_fractional = float(step) / num_batches
                yield step, epoch_fractional, self._prepare_batch(batch, self._device)

    def record_class_distribution(
        self,
        targets: Union[List, np.ndarray],
        indices: Union[List, np.ndarray],
        print_fn: Callable = print,
        is_train: bool = True,
        display: bool = True,
    ):
        targets_np = np.array(targets)
        unique_elements, counts_elements = np.unique(
            targets_np[indices] if indices is not None else targets_np,
            return_counts=True,
        )
        element_counts = list(zip(unique_elements, counts_elements))

        if display:
            print_fn(
                f"\tThe histogram of the targets in {'train' if is_train else 'test'}: {element_counts}"
            )
        return element_counts
    
    
# # 新增：适配CTTA
# class DomainAwareSampler(torch.utils.data.Sampler):
#     """保证不同域的数据不会混在同一个batch中的采样器"""
#     def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True):
#         """
#         Args:
#             lengths: 每个域包含的样本数量列表
#             batch_size: batch大小
#             shuffle: 是否打乱每个域内的样本顺序
#         """
#         self.lengths = lengths
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.num_samples = sum(lengths)
#         self.dataset_starts = [0] + list(np.cumsum(lengths))[:-1]
        
#     def __iter__(self):
#         indices = []
#         # 为每个域生成索引
#         for start, length in zip(self.dataset_starts, self.lengths):
#             domain_indices = list(range(start, start + length))
#             if self.shuffle:
#                 random.shuffle(domain_indices)
            
#             # 添加调试信息
#             print(f"Domain range: {start} to {start + length}, batch_size: {self.batch_size}")
            
#             # 按batch_size分组，保留不完整的最后一个batch
#             for i in range(0, len(domain_indices), self.batch_size):
#                 batch_indices = domain_indices[i:i + self.batch_size]
#                 if len(batch_indices) > 0:
#                     indices.extend(batch_indices)
                    
#         return iter(indices)
    
#     def __len__(self):
#         return self.num_samples


# 新的DomainAwareBatchSampler类，替代原来的DomainAwareSampler
class DomainAwareBatchSampler(torch.utils.data.Sampler):
    """域感知批量采样器，确保每个批次只包含来自同一个域的数据"""
    
    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Args:
            lengths: 每个域包含的样本数量列表
            batch_size: 批量大小
            shuffle: 是否在每个域内打乱数据
            drop_last: 是否丢弃不足一个批次的最后几个样本
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = sum(lengths)
        
        # 计算每个域的起始索引
        self.dataset_starts = [0]
        for i in range(len(lengths) - 1):
            self.dataset_starts.append(self.dataset_starts[-1] + lengths[i])
            
        # 计算批次总数
        if drop_last:
            self.num_batches = sum(length // batch_size for length in lengths)
        else:
            self.num_batches = sum((length + batch_size - 1) // batch_size for length in lengths)
    
    def __iter__(self):
        # 存储所有批次的索引列表
        batches = []
        
        # 为每个域生成批次
        for domain_idx, (start, length) in enumerate(zip(self.dataset_starts, self.lengths)):
            # 域内的所有索引
            domain_indices = list(range(start, start + length))
            
            # 如果需要打乱，则打乱域内索引
            if self.shuffle:
                random.shuffle(domain_indices)
                
            # 添加调试信息
            print(f"Domain range: {start} to {start + length}, batch_size: {self.batch_size}")
            
            # 将索引分成批次
            for i in range(0, len(domain_indices), self.batch_size):
                if i + self.batch_size <= len(domain_indices) or not self.drop_last:
                    # 如果批次完整或者不丢弃不完整的批次
                    batch = domain_indices[i:i + self.batch_size]
                    if len(batch) > 0:  # 确保批次不为空
                        batches.append(batch)
        
        # 如果shuffle为True，还可以选择打乱批次的顺序（但批次内部的索引顺序保持不变）
        # 注意：这是可选的，取决于您是否希望域之间的批次顺序随机
        if self.shuffle:
            random.shuffle(batches)
            
        return iter(batches)
    
    def __len__(self):
        return self.num_batches