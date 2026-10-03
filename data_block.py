#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
芯片执行任务的动态调度动画 - 数据块类

该模块定义了数据块类，用于表示在芯片和DDR之间移动的数据块。
"""

from enum import Enum, auto
from typing import Tuple, Optional


class DataBlockType(Enum):
    """数据块类型"""
    SEQ = auto()  # 序列数据块
    EXPERT = auto()  # 专家数据块


class DataBlockState(Enum):
    """数据块状态枚举（与类型分离）"""
    COMPUTING = auto()  # 正在计算的数据块
    MOVING = auto()     # 正在移动的数据块
    AFTER_COMPUTE = auto()  # 计算完成后的数据块


class DataBlock:
    """数据块类"""
    def __init__(self, name: str, block_type: DataBlockType = DataBlockType.SEQ,
                 location: Tuple[int, int] = None, position: Tuple[float, float] = None,
                 state: Optional[DataBlockState] = None):
        self.name = name
        self.copy_number = 0 # 数据块复制次数
        self.block_type = block_type 
        self.location = location # tuple的第一位表示ddr或者chip ID, 第二位表示具体的buffer index
        self.position = position # 画图时的位置
        self.state = state

        # MoE特化的信息
        self.seq_or_expert_id = 0  # 序列数据块或者专家数据块的ID
        self.mico_slice = 0

    
    @property
    def block_id(self):            # 数据块唯一标识
        return self.name + '-' + str(self.copy_number)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
        
    def __eq__(self, other):
        if not isinstance(other, DataBlock):
            return False
        return self.block_id == other.block_id
        
    def __hash__(self):
        return hash(self.block_id)
