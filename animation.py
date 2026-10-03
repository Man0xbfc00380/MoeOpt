#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
芯片执行任务的动态调度动画 - 动画模块

该模块实现了芯片与 DDR 的可视化，以及数据块在 DDR 与各芯片之间的装载、移动、计算、释放等动画。

注意：本文件仅负责可视化与动画播放，不改变调度策略与数据结构本身的行为。
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation  # noqa: F401  保留：外部可能使用到
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text

from data_block import DataBlock, DataBlockType, DataBlockState
from hardware_config import HardwareConfig
from scheduler import Action, ActionType, Scheduler
import random
from matplotlib.animation import FuncAnimation

# 设置Arial字体
font_arial = FontProperties(family='Arial')


class ChipletAnimator:
    """
    芯片动画器。

    职责：
    - 根据 `HardwareConfig` 与 `Scheduler` 状态，绘制芯片/DDR/缓冲区与数据块。
    - 逐帧执行调度器返回的动作(Action)，并以动画形式呈现：
      - DDR/CHIP 之间移动
      - 分配/释放
      - 计算高亮
    - 展示简单的“计算占用率”曲线。

    行为约束：
    - 不修改调度器与数据结构的业务逻辑，仅做可视化展示。
    """

    def __init__(self, scheduler: Scheduler, show_group_overlay: bool = True):
        """初始化动画器。

        Args:
            scheduler: 任务调度器实例，负责提供逐帧动作。
            show_group_overlay: 是否显示每个 expert 的 chip 分组填充/边框。
        """
        self.scheduler = scheduler
        self.show_group_overlay = show_group_overlay

        self.row_number_of_chip = HardwareConfig.row_number_of_chip
        self.column_number_of_chip = HardwareConfig.column_number_of_chip
        # 芯片和DDR的大小
        self.chip_width = 0.2
        self.chip_height = 0.28
        self.ddr_width = 0.1
        self.ddr_height = self.chip_height
        self.pos1 = (0.1, 0.3)  # 最左下角的开始画形状的起始位置
        self.ddr_chip_x_interval = 0.05  # 芯片和DDR之间的间隔
        self.ddr_ddr_y_interval = 0.05  # DDR和DDR之间的间隔

        # 计算画布大小
        self.canvas_width = (2*self.pos1[0] + 2*self.ddr_width + 2*self.ddr_chip_x_interval +
                             self.column_number_of_chip * (self.chip_width + self.ddr_chip_x_interval))
        self.canvas_height = (self.pos1[1] + self.row_number_of_chip *
                              (self.ddr_height + self.ddr_ddr_y_interval) + 0.04)
        self.font_scale = max(self.canvas_width, self.canvas_height)

        # 按1x1画布大小归一化组件尺寸
        self.chip_width = 0.2 / self.canvas_width
        self.chip_height = 0.28 / self.canvas_height
        self.ddr_width = 0.14 / self.canvas_width
        self.ddr_height = self.chip_height
        self.pos1 = (0.1 / self.canvas_width, 0.3 /
                     self.canvas_height)  # 最左下角的开始画形状的起始位置
        self.ddr_chip_x_interval = 0.05 / self.canvas_width  # 芯片和DDR之间的间隔
        self.ddr_ddr_y_interval = 0.05 / self.canvas_height  # DDR和DDR之间的间隔

        # 计算芯片位置
        self.chip_positions = {}
        for row in range(self.row_number_of_chip):
            for col in range(self.column_number_of_chip):
                chip_id = f"Chip({row}, {col})"

                # 计算每个芯片的x坐标
                chip_x = (self.pos1[0] + self.ddr_width + self.ddr_chip_x_interval +
                          col * (self.chip_width + self.ddr_chip_x_interval))

                # 计算每个芯片的y坐标，从上到下排列
                chip_y = (self.pos1[1] + (self.row_number_of_chip - 1 - row) *
                          (self.ddr_height + self.ddr_ddr_y_interval))

                self.chip_positions[chip_id] = (chip_x, chip_y)

        # 计算DDR位置
        ddr_left_x = self.pos1[0]
        ddr_right_x = self.chip_positions[f"Chip(0, {self.column_number_of_chip-1})"][0] + \
            self.chip_width + self.ddr_chip_x_interval
        ddr_top_y = self.chip_positions["Chip(0, 0)"][1]
        chip_bottom_y = self.chip_positions[f"Chip({self.row_number_of_chip-1}, 0)"][1]

        self.ddr_positions = {
            "DDR1": (ddr_left_x, ddr_top_y),
            "DDR2": (ddr_right_x, ddr_top_y),
            "DDR3": (ddr_right_x, chip_bottom_y),
            "DDR4": self.pos1,
        }

        # 独立的序列与专家缓冲区容量
        self.token_buffer_size = HardwareConfig.token_buffer_size
        self.expert_buffer_size = HardwareConfig.expert_buffer_size
        self.ddr_buffer_size = 4  # 显式的大小，不代表真实大小

        # 数据块的大小
        self.data_block_width = 0.07 / self.canvas_width
        # 数据块高度根据两类缓冲区的最大容量缩放，保证能完整显示
        self.data_block_height = 0.02 * \
            (10 / max(self.token_buffer_size, self.expert_buffer_size)) / self.canvas_height
        # Buffer的大小和位置偏移
        self.buffer_width = 0.07 / self.canvas_width

        self.chip_token_buffer_height = self.data_block_height * self.token_buffer_size
        self.chip_expert_buffer_height = self.data_block_height * self.expert_buffer_size
        self.ddr_buffer_height = self.data_block_height * self.ddr_buffer_size
        self.seq_buffer_offset_x = (
            self.chip_width - 2 * self.buffer_width) / 3
        self.expert_buffer_offset_x = 2 * self.seq_buffer_offset_x + self.buffer_width
        self.buffer_offset_y = self.seq_buffer_offset_x
        self.ddr_columns = {}
        self.columns_interval = {}

        # 数据块之间的间距
        self.data_block_spacing = 0.000

        # 数据块的颜色
        self.data_block_colors = {
            DataBlockType.SEQ: 'skyblue',
            # 为EXPERT随机选择一种颜色
            DataBlockType.EXPERT: ['plum', 'lightcoral', 'peachpuff', 'lightsalmon'],
            DataBlockState.COMPUTING: 'lightgreen',
        }

        # 动画对象
        self.fig = None  # Figure
        self.ax = None   # Axes
        self.chip_patches = {}
        self.ddr_patches = {}
        self.buffer_patches = {}
        self.data_block_patches = {}
        self.data_block_texts = {}
        self.arrow_patches = []
        # 专家组覆盖层（半透明填充 + 边框）
        self.group_overlay_patches = []  # type: ignore[assignment]
        self.group_overlay_texts = []    # type: ignore[assignment]

        # 动画状态
        self.frame_count = 0
        self.action_queue = []  # 待执行的动作队列
        self.current_actions = {}  # 当前正在执行的动作 {action_id: (action, progress)}
        self.next_action_id = 0  # 用于生成唯一的动作ID

        self.is_running = False  # 动画是否正在运行
        # 添加开始按钮
        self.start_button_ax = None  # Axes
        self.start_button = None     # plt.Button
        self.text_box = None  # Text

        # 添加折线图相关属性
        self.line_plot_ax = None  # Axes
        # 使用更直观的写法初始化折线图对象
        self.line = None
        self.x_data = []  # 存储帧数
        self.y_data = []  # 存储action数量
        self._current_computing = {
            chip_id: False for chip_id in self.chip_positions.keys()}

        self.finish = False

    def set_show_group_overlay(self, show: bool) -> None:
        """切换是否显示专家组覆盖层（在 init 之后调用将重建覆盖层）。"""
        self.show_group_overlay = show
        # 若图已建立，尝试重建覆盖层
        if hasattr(self, 'ax') and self.ax is not None:
            self._clear_group_overlays()
            if self.show_group_overlay:
                self._draw_group_overlays()

    def _clear_group_overlays(self) -> None:
        """移除已绘制的专家组覆盖层。"""
        for p in self.group_overlay_patches:
            try:
                p.remove()
            except Exception:
                pass
        for t in self.group_overlay_texts:
            try:
                t.remove()
            except Exception:
                pass
        self.group_overlay_patches.clear()
        self.group_overlay_texts.clear()

    def _compute_group_bbox(self, chips: List[str]) -> Optional[Tuple[float, float, float, float]]:
        """计算一组芯片的包围盒 (x, y, w, h)。若列表为空返回 None。"""
        valid = [c for c in chips if c in self.chip_positions]
        if not valid:
            return None
        xs = [self.chip_positions[c][0] for c in valid]
        ys = [self.chip_positions[c][1] for c in valid]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        # 芯片矩形是统一尺寸
        x = min_x - 0.01 / self.canvas_width
        y = min_y - 0.01 / self.canvas_height
        w = (max_x - min_x) + self.chip_width + 0.02 / self.canvas_width
        h = (max_y - min_y) + self.chip_height + 0.02 / self.canvas_height
        return (x, y, w, h)

    def _draw_group_overlays(self) -> None:
        """为每个专家组绘制半透明背景与边框以及标签。"""
        # 简单的颜色循环
        palette = [
            '#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B',
            '#EECA3B', '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC'
        ]
        for idx, (expert_id, chip_list) in enumerate(self.scheduler.expert_groups.items()):
            bbox = self._compute_group_bbox(chip_list)
            if not bbox:
                continue
            color = palette[idx % len(palette)]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.08,
                zorder=0.5
            )
            self.ax.add_patch(rect)
            self.group_overlay_patches.append(rect)

            # 标签放在包围盒左上角
            label_x = bbox[0] + 0.006 / self.canvas_width
            label_y = bbox[1] + bbox[3] - 0.02 / self.canvas_height
            txt = self.ax.text(
                label_x, label_y, f"Expert {expert_id}",
                ha='left', va='top', fontsize=11 / self.font_scale,
                fontproperties=font_arial, color=color, zorder=0.6
            )
            self.group_overlay_texts.append(txt)

    def _on_start_click(self, event):
        """开始按钮点击事件处理：将动画置为运行态并禁用按钮。"""
        self.is_running = True
        # 更新按钮样式
        self.start_button.color = 'lightgray'
        self.start_button.active = False

    def _get_data_block_color(self, data_block: DataBlock):
        """根据数据块类型与状态选择颜色（COMPUTING 状态优先）。"""
        color = self.data_block_colors[data_block.block_type]
        if data_block.state == DataBlockState.COMPUTING:
            color = self.data_block_colors[data_block.state]
        if isinstance(color, list):
            return color[data_block.seq_or_expert_id % len(color)]
        return color

    def init(self):
        """初始化动画图层与元素（芯片、DDR、缓冲区、已有数据块、连线、控制组件）。

        返回值用于 FuncAnimation 的 blit 模式。
        """
        # 创建开始按钮
        self.start_button_ax = plt.axes(
            [0.05, 0.125 / self.canvas_height, 0.08,  0.04 / self.canvas_height])
        self.start_button = plt.Button(
            self.start_button_ax, 'Start', color='lightgreen')
        self.start_button.on_clicked(self._on_start_click)

        text_box_ax = plt.axes(
            [0.14, 0.05 / self.canvas_height, 0.4, 0.2 / self.canvas_height])
        self.text_box = text_box_ax.text(0.01, 0.95, '',
                                         verticalalignment='top',
                                         fontproperties=font_arial,
                                         fontsize=8 / self.font_scale,
                                         bbox=dict(facecolor='none',
                                                   edgecolor='none'))
        text_box_ax.set_xticks([])
        text_box_ax.set_yticks([])

        # 创建折线图
        self.line_plot_ax = plt.axes(
            [0.56, 0.05 / self.canvas_height, 0.39, 0.2 / self.canvas_height])
        self.line_plot_ax.grid(False)
        self.line_plot_ax.set_ylim(0, 1)  # 固定y轴范围为0-1
        # 显示刻度线但不显示刻度文字
        self.line_plot_ax.set_xticks([])
        self.line_plot_ax.set_yticks(np.arange(0, 1.001, 0.2), labels=[])
        # 隐藏右边框和上边框
        self.line_plot_ax.spines['right'].set_visible(False)
        self.line_plot_ax.spines['top'].set_visible(False)
        self.line, = self.line_plot_ax.plot([], [], 'b-')

        plt.axes([0, 0, 1, 1])

        # 创建图形和坐标轴
        self.ax = plt.gca()
        # 记录当前 Figure，供后续子动画（如烟花）使用
        self.fig = self.ax.figure
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        # 组覆盖层（在芯片与缓冲区之前渲染，使其位于下层）
        if self.show_group_overlay:
            # 先在拥有坐标轴之后绘制，以便添加到 ax
            # 注意：此时 chip_positions 已经计算完毕
            self._draw_group_overlays()

        # 创建芯片和DDR的矩形
        for chip_id, pos in self.chip_positions.items():
            rect = patches.Rectangle(
                pos, self.chip_width, self.chip_height,
                linewidth=1, edgecolor='black', facecolor='none',
                zorder=1
            )
            self.ax.add_patch(rect)
            self.chip_patches[chip_id] = rect

            self.ax.text(
                pos[0] + self.chip_width/2, pos[1] +
                self.chip_height - 0.03 / self.canvas_height,
                chip_id, ha='center', va='bottom', fontproperties=font_arial, fontsize=12 / self.font_scale,
                zorder=5
            )

            # 添加Buffer
            seq_buffer = patches.Rectangle(
                (pos[0] + self.seq_buffer_offset_x,
                 pos[1] + self.buffer_offset_y),
                self.buffer_width, self.chip_token_buffer_height,
                linewidth=1, edgecolor='black', facecolor='skyblue', alpha=0.3,
                zorder=1
            )
            self.ax.add_patch(seq_buffer)
            self.buffer_patches[f"{chip_id}_seq"] = seq_buffer

            # 添加Buffer标签
            self.ax.text(
                pos[0] + self.seq_buffer_offset_x + self.buffer_width/2,
                pos[1] + self.buffer_offset_y +
                self.chip_token_buffer_height + 0.005 / self.canvas_height,
                "Seq Buffer", ha='center', va='bottom', fontsize=10 / self.font_scale,
                fontproperties=font_arial,
                zorder=5
            )

            expert_buffer = patches.Rectangle(
                (pos[0] + self.expert_buffer_offset_x,
                 pos[1] + self.buffer_offset_y),
                self.buffer_width, self.chip_expert_buffer_height,
                linewidth=1, edgecolor='black', facecolor='lightcoral', alpha=0.3,  # 降低透明度让文字更容易看到
                zorder=1
            )
            self.ax.add_patch(expert_buffer)
            self.buffer_patches[f"{chip_id}_expert"] = expert_buffer

            # 添加Buffer标签
            self.ax.text(
                pos[0] + self.expert_buffer_offset_x + self.buffer_width/2,
                pos[1] + self.buffer_offset_y + self.chip_expert_buffer_height +
                0.005 / self.canvas_height,  # 调整文字位置到buffer上方
                "Expert Buffer", ha='center', va='bottom', fontsize=10 / self.font_scale,
                fontproperties=font_arial,
                zorder=5
            )

        for ddr_id, pos in self.ddr_positions.items():
            rect = patches.Rectangle(
                pos, self.ddr_width, self.ddr_height,
                linewidth=1, edgecolor='black', facecolor='none',
                zorder=1
            )
            self.ax.add_patch(rect)
            self.ddr_patches[ddr_id] = rect

            # 添加DDR标签
            self.ax.text(
                pos[0] + self.ddr_width/2, pos[1] +
                self.ddr_height - 0.03 / self.canvas_height,
                ddr_id, ha='center', va='bottom', fontproperties=font_arial, fontsize=12 / self.font_scale,
                zorder=5
            )

        # 为每个数据块创建图形
        ddr_max_indices = {}
        for datablocks in self.scheduler.data_blocks.values():
            # 计算每个DDR中最大的buffer_index来确定需要多少列
            for datablock in datablocks:
                location, buffer_index = datablock.location
                if location.startswith("DDR"):
                    ddr_max_indices[location] = max(
                        ddr_max_indices.get(location, 0), buffer_index)

        # 计算每个 DDR 需要的列数；为所有已知 DDR 预设默认值，避免后续惰性访问出现 KeyError
        # 默认至少 1 列；若已观察到更大索引则按需扩展列数
        self.ddr_columns = {ddr: 1 for ddr in self.ddr_positions.keys()}
        for ddr, max_idx in ddr_max_indices.items():
            self.ddr_columns[ddr] = max(self.ddr_columns[ddr], (max_idx // self.ddr_buffer_size) + 1)

        # 列间距按列数计算（列数为 1 时即居中）
        self.columns_interval = {ddr: (self.ddr_width - self.data_block_width) /
                                 (self.ddr_columns[ddr] + 1) for ddr in self.ddr_columns}

        for datablocks in self.scheduler.data_blocks.values():
            for datablock in datablocks:
                location, buffer_index = datablock.location
                # 计算数据块位置
                if location.startswith("DDR"):
                    pos = self.ddr_positions[location]
                    # 计算当前数据块应该在第几列
                    column = buffer_index // self.ddr_buffer_size
                    # 计算在当前列中的位置
                    folded_index = buffer_index % self.ddr_buffer_size

                    # 从右到左排列列，第一列在最右边
                    total_columns = self.ddr_columns[location]
                    column_from_right = total_columns - column
                    # 计算x坐标，每列错开一定距离
                    column_offset = column_from_right * \
                        self.columns_interval[location]
                    x = pos[0] + column_offset

                    # 计算y坐标
                    y = pos[1] + self.buffer_offset_y + folded_index * \
                        (self.data_block_height + self.data_block_spacing)

                    # 设置z-order，使第一列在最上层
                    z_order = 4 - column * 0.01
                else:  # Chip
                    pos = self.chip_positions[location]
                    if datablock.block_type == DataBlockType.SEQ:
                        x = pos[0] + self.seq_buffer_offset_x
                    else:  # EXPERT
                        x = pos[0] + self.expert_buffer_offset_x
                    y = pos[1] + self.buffer_offset_y + buffer_index * \
                        (self.data_block_height + self.data_block_spacing)
                    z_order = 4

                # 创建数据块矩形
                color = self._get_data_block_color(datablock)
                rect = patches.Rectangle(
                    (x, y), self.data_block_width, self.data_block_height,
                    linewidth=1, edgecolor='black', facecolor=color,
                    linestyle='-',
                    zorder=z_order
                )
                self.ax.add_patch(rect)
                self.data_block_patches[datablock.block_id] = rect

                # 创建数据块文本
                text = self.ax.text(
                    x + self.data_block_width/2, y + self.data_block_height/2,
                    str(datablock), ha='center', va='center', fontsize=8 / self.font_scale,
                    fontproperties=font_arial,
                    zorder=z_order
                )
                self.data_block_texts[datablock.block_id] = text

                # 更新数据块位置信息
                datablock.position = (x, y)

        # 箭头样式设置
        arrow_style = '<->'  # 双向箭头（逻辑链路，非单向总线）
        arrow_color = 'black'
        arrow_width = 1.5 / self.canvas_height
        arrow_scale = 15 / self.font_scale  # 箭头大小

        # True表示DDR在左侧
        ddr_left = {"DDR1": True, "DDR2": False, "DDR3": False, "DDR4": True}

        for ddr_id, chip_id in HardwareConfig.ddr_chip_connections().items():
            ddr_pos = self.ddr_positions[ddr_id]
            chip_pos = self.chip_positions[chip_id]

            start_x = ddr_pos[0] + \
                self.ddr_width if ddr_left[ddr_id] else chip_pos[0] + \
                self.chip_width
            end_x = chip_pos[0] if ddr_left[ddr_id] else ddr_pos[0]

            self.arrow_patches.append(self.ax.add_patch(FancyArrowPatch(
                (start_x, ddr_pos[1] + self.ddr_height/2),
                (end_x, chip_pos[1] + self.chip_height/2),
                arrowstyle=arrow_style,
                mutation_scale=arrow_scale,
                linewidth=arrow_width,
                color=arrow_color,
                zorder=3
            )))

        # 芯片之间的水平连接
        for row in range(self.row_number_of_chip):
            for col in range(self.column_number_of_chip - 1):
                start_chip = f"Chip({row}, {col})"
                end_chip = f"Chip({row}, {col + 1})"

                start_pos = self.chip_positions[start_chip]
                end_pos = self.chip_positions[end_chip]

                self.arrow_patches.append(self.ax.add_patch(FancyArrowPatch(
                    (start_pos[0] + self.chip_width,
                     start_pos[1] + self.chip_height/2),
                    (end_pos[0], end_pos[1] + self.chip_height/2),
                    arrowstyle=arrow_style,
                    mutation_scale=arrow_scale,
                    linewidth=arrow_width,
                    color=arrow_color,
                    zorder=3
                )))

        # 芯片之间的垂直连接
        for row in range(self.row_number_of_chip - 1):
            for col in range(self.column_number_of_chip):
                top_chip = f"Chip({row}, {col})"
                bottom_chip = f"Chip({row + 1}, {col})"

                top_pos = self.chip_positions[top_chip]
                bottom_pos = self.chip_positions[bottom_chip]

                self.arrow_patches.append(self.ax.add_patch(FancyArrowPatch(
                    (top_pos[0] + self.chip_width/2, top_pos[1]),
                    (bottom_pos[0] + self.chip_width/2,
                     bottom_pos[1] + self.chip_height),
                    arrowstyle=arrow_style,
                    mutation_scale=arrow_scale,
                    linewidth=arrow_width,
                    color=arrow_color,
                    zorder=3
                )))

        # 返回所有图形对象
        patches_list = list(self.group_overlay_patches) + list(self.group_overlay_texts) + \
            list(self.chip_patches.values()) + list(self.ddr_patches.values()) + \
            list(self.buffer_patches.values()) + list(self.data_block_patches.values()) + \
            list(self.data_block_texts.values()) + \
            self.arrow_patches + [self.text_box, self.line]
        return patches_list

    def _get_pos(self, location: str, index: int, block_type: Optional[DataBlockType] = None) -> Tuple[float, float]:
        """根据位置与索引计算数据块左下角坐标。

        Args:
            location: 数据块所在位置（'DDR*' 或 'Chip(r, c)'）。
            index:    在对应缓冲区中的顺序索引（0 基）。可根据 DDR 折叠为多列显示。
            block_type: 数据块类型（仅在 Chip 上用于区分 Seq/Expert 缓冲区的 X 方向偏移）。

        Returns:
            (x, y): 数据块左下角坐标（归一化到 0~1 画布）。
        """

        if location.startswith("DDR"):
            # DDR中的数据块位置计算
            pos = self.ddr_positions[location]
            # 计算当前数据块应该在第几列
            column = index // self.ddr_buffer_size
            # 计算在当前列中的位置
            folded_index = index % self.ddr_buffer_size

            # 从右到左排列列，第一列在最右边
            total_columns = self.ddr_columns[location]
            column_from_right = total_columns - column
            # 计算x坐标，每列错开一定距离
            column_offset = column_from_right * self.columns_interval[location]
            x = pos[0] + column_offset

            # 计算y坐标
            y = pos[1] + self.buffer_offset_y + folded_index * \
                (self.data_block_height + self.data_block_spacing)
        else:
            # 芯片中的数据块位置计算
            pos = self.chip_positions[location]
            if block_type == DataBlockType.SEQ:
                x = pos[0] + self.seq_buffer_offset_x
            else:  # EXPERT
                x = pos[0] + self.expert_buffer_offset_x
            y = pos[1] + self.buffer_offset_y + index * \
                (self.data_block_height + self.data_block_spacing)

        return (x, y)

    def _create_data_block_patches(self, datablock: DataBlock) -> None:
        """在指定位置创建数据块矩形与文字标注。"""
        block_id = datablock.block_id
        location, index = datablock.location

        # 使用_get_pos计算数据块位置
        x, y = self._get_pos(location, index, datablock.block_type)

        # 创建数据块矩形
        color = self._get_data_block_color(datablock)
        rect = patches.Rectangle(
            (x, y), self.data_block_width, self.data_block_height,
            linewidth=1, edgecolor='black', facecolor=color,
            linestyle='-',
            zorder=4
        )
        self.ax.add_patch(rect)
        self.data_block_patches[block_id] = rect

        # 创建数据块文本
        text = self.ax.text(
            x + self.data_block_width/2, y + self.data_block_height/2,
            str(datablock), ha='center', va='center', fontsize=7 / self.font_scale,
            fontproperties=font_arial,
            zorder=5
        )
        self.data_block_texts[block_id] = text

        # 更新数据块位置信息
        datablock.position = (x, y)

    def _release_data_block_patches(self, datablock: DataBlock) -> None:
        """删除指定位置的数据块矩形与文字标注。"""
        block_id = datablock.block_id

        # 删除数据块矩形
        if block_id in self.data_block_patches:
            self.data_block_patches[block_id].remove()
            del self.data_block_patches[block_id]

        # 删除数据块文本
        if block_id in self.data_block_texts:
            self.data_block_texts[block_id].remove()
            del self.data_block_texts[block_id]

    # 兼容旧命名：保留 wrapper，不改变外部调用
    def _release_data_block_batches(self, datablock: DataBlock) -> None:
        self._release_data_block_patches(datablock)

    def _animate_move(self, action: Action, progress: float) -> None:
        """动画：在 source -> destination 之间插值移动数据块。"""
        # 确保progress在0-1范围内
        progress = max(0.0, min(1.0, progress))
        destination = action.destination
        source = action.source
        block = action.data_block
        block_id = block.block_id
        pre_block_id = "pre-" + block_id

        start_pos = self._get_pos(source[0], source[1], block.block_type)
        end_pos = self._get_pos(
            destination[0], destination[1], block.block_type)

        # 计算当前位置
        current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

        # 更新数据块位置
        if block_id in self.data_block_patches:
            self.data_block_patches[block_id].set_xy((current_x, current_y))
        if block_id in self.data_block_patches:
            self.data_block_patches[block_id].set_zorder(6)

        if block_id in self.data_block_texts:
            self.data_block_texts[block_id].set_position(
                (current_x + self.data_block_width/2, current_y + self.data_block_height/2))
            self.data_block_texts[block_id].set_zorder(7)

        # 在目的地创建预分配数据块
        if pre_block_id not in self.data_block_patches:
            # 创建预分配数据块矩形
            rect = patches.Rectangle(
                end_pos, self.data_block_width, self.data_block_height,
                linewidth=1, edgecolor='black', facecolor=self._get_data_block_color(block),
                linestyle='--',
                zorder=2
            )
            self.ax.add_patch(rect)
            self.data_block_patches[pre_block_id] = rect

            # 创建预分配数据块文本
            text = self.ax.text(
                end_pos[0] + self.data_block_width /
                2, end_pos[1] + self.data_block_height/2,
                pre_block_id, ha='center', va='center', fontsize=7 / self.font_scale,
                fontproperties=font_arial,
                zorder=3
            )
            self.data_block_texts[pre_block_id] = text

        if progress >= 1.0:
            if block_id in self.data_block_patches:
                self.data_block_patches[block_id].set_zorder(4)
            if block_id in self.data_block_texts:
                self.data_block_texts[block_id].set_zorder(5)

            # 移动完成时删除预分配数据块
            if pre_block_id in self.data_block_patches:
                self.data_block_patches[pre_block_id].remove()
                del self.data_block_patches[pre_block_id]
            if pre_block_id in self.data_block_texts:
                self.data_block_texts[pre_block_id].remove()
                del self.data_block_texts[pre_block_id]

    def _animate_allocated(self, action: Action, progress: float) -> None:
        """动画：预分配数据块（当前仅支持分配到芯片缓冲区）。"""

        if action.data_block.location[0].startswith("DDR"):
            raise ValueError("Not support DDR allocation yet")

        # 更新数据块的图形显示
        self._create_data_block_patches(action.data_block)

    def _animate_compute(self, action: Action, progress: float) -> None:
        """动画：计算高亮。

        在 progress < 0.9 时设置为 COMPUTING 状态，接近完成时恢复原类型颜色。
        """
        # 确保progress在0-1范围内
        progress = max(0.0, min(1.0, progress))
        blocks = [action.data_block] + action.other_blocks
        for block in blocks:
            # 更新数据块状态
            if progress < 0.9:
                # 开始计算
                block.state = DataBlockState.COMPUTING
            else:
                # 计算完成
                block.state = None

            # 更新数据块颜色
            if block.block_id in self.data_block_patches:
                self.data_block_patches[block.block_id].set_facecolor(
                    self._get_data_block_color(block))
            # else:
            #     raise ValueError(
            #         f"DataBlock {block.block_id} not found in patches")

    def _animate_release(self, action: Action, progress: float) -> None:
        """动画：释放数据块，逐步降低透明度，结束时移除图元。"""
        # 确保progress在0-1范围内
        progress = max(0.0, min(1.0, progress))
        block = action.data_block
        block_id = block.block_id
        # 计算当前透明度（确保在0-1范围内）
        current_alpha = max(0.0, min(1.0, 1.0 - progress))

        # 更新数据块透明度
        if block_id in self.data_block_patches:
            self.data_block_patches[block_id].set_alpha(current_alpha)

        if block_id in self.data_block_texts:
            self.data_block_texts[block_id].set_alpha(current_alpha)

        # 如果动画完成，移除数据块
        if progress >= 1.0:
            self._release_data_block_batches(block)

    def update(self, frame):
        """更新单帧动画。

        约定：
        - 每帧可从调度器拉取新动作，并将其加入当前执行队列。
        - 按动作类型驱动对应的动画子流程（移动/计算/释放/分配）。
        - 返回本帧需要重绘的图元集合（用于 blit）。
        """
        if not self.is_running:
            # 如果动画未开始，返回当前所有图形对象
            patches_list = list(self.group_overlay_patches) + list(self.group_overlay_texts) + \
                list(self.chip_patches.values()) + list(self.ddr_patches.values()) + \
                list(self.buffer_patches.values()) + list(self.data_block_patches.values()) + \
                list(self.data_block_texts.values()) + \
                self.arrow_patches + [self.line]
            return patches_list

        self.frame_count += 1

        # 每隔一定帧数从调度器获取新动作
        if frame % 1 == 0 or not self.current_actions:
            new_actions = self.scheduler.step()
            # 立即开始执行新动作
            for action in new_actions:
                action_text = f"Schedule {self.next_action_id}: {action}"
                # 更新文本框内容
                if not hasattr(self, 'action_log'):
                    self.action_log = []
                self.action_log.append(action_text)
                # 保持最新的10行
                if len(self.action_log) > 13:
                    self.action_log.pop(0)
                # 更新文本框显示
                if hasattr(self, 'text_box'):
                    self.text_box.set_text('\n'.join(self.action_log))

                action_id = self.next_action_id
                self.next_action_id += 1
                self.current_actions[action_id] = (
                    action, 0)  # (action, progress)

        # 更新所有正在执行的动作
        completed_actions = []
        max_duration = 1
        for action_id, (action, progress) in self.current_actions.items():
            # 根据动作类型获取持续时间
            if action.action_type != ActionType.FLAG:
                action_duration = HardwareConfig.action_durations.get(
                    action.action_type, 20)  # 默认20帧

                if action.action_type == ActionType.COMPUTE:
                    action_duration = max(
                        action_duration * len(action.other_blocks), 1)

                max_duration = max(max_duration, action_duration)
            else:
                action_duration = max_duration

            # 更新动作进度，确保不超过1.0
            new_progress = min(1.0, progress + 1.0 / action_duration)

            # 执行动作动画
            if action.action_type in [ActionType.CHIP_MOVE, ActionType.DDR_LOAD, ActionType.DDR_SAVE]:
                self._animate_move(action, new_progress)
            elif action.action_type == ActionType.COMPUTE:
                self._current_computing[action.destination[0]] = True
                self._animate_compute(action, new_progress)
            elif action.action_type == ActionType.RELEASE:
                self._animate_release(action, new_progress)
            elif action.action_type == ActionType.ALLOCATE:
                self._animate_allocated(action, new_progress)
            elif action.action_type == ActionType.FLAG and action.data_block.name == 'finish':
                if not self.finish:
                    self.finish = True
                    self._trigger_fireworks()

            # 更新进度
            self.current_actions[action_id] = (action, new_progress)

            # 如果动作执行完成，则标记为待移除
            if new_progress >= 1.0:
                completed_actions.append(action_id)
                # 通知scheduler该数据块的动作已完成
                self.scheduler.handle_action_complete(action)

         # 更新开始按钮显示当前帧数
        if self.is_running:
            self.start_button.label.set_text(f'Frame: {self.frame_count}')
            self.start_button_ax.figure.canvas.draw()

            # 更新折线图数据
            self.x_data.append(self.frame_count)
            # 计算当前正在计算的芯片比率
            computing_utilization = sum(
                1 for v in self._current_computing.values() if v) / len(self._current_computing)
            self.y_data.append(computing_utilization)

            # 更新折线图显示
            self.line.set_data(self.x_data, self.y_data)
            self.line_plot_ax.relim()
            self.line_plot_ax.autoscale_view()
            self.line_plot_ax.figure.canvas.draw()

        # 移除已完成的动作
        for action_id in completed_actions:
            action = self.current_actions[action_id][0]
            if action.action_type == ActionType.COMPUTE:
                self._current_computing[action.destination[0]] = False
            del self.current_actions[action_id]

        # 返回所有图形对象
        return list(self.group_overlay_patches) + list(self.group_overlay_texts) + \
            list(self.chip_patches.values()) + list(self.ddr_patches.values()) + \
            list(self.buffer_patches.values()) + list(self.data_block_patches.values()) + \
            list(self.data_block_texts.values()) + \
            self.arrow_patches + [self.text_box, self.line]

    def _trigger_fireworks(self):
        """触发烟花动画：在画布中心随机生成彩色散点，模拟烟花爆炸。"""
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        # 烟花中心点（画布中心）
        cx, cy = 0.5, 0.5
        # 烟花粒子数量
        particle_count = 200
        # 烟花持续时间（帧数）
        duration = 60

        # 生成随机颜色与初始速度
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
                  '#FFA500', '#FF1493', '#32CD32', '#1E90FF']
        particles = []
        for _ in range(particle_count):
            angle = random.uniform(0, 2 * 3.14159)
            speed = random.uniform(0.003, 0.008)
            vx = speed * random.uniform(-1, 1)
            vy = speed * random.uniform(-1, 1)
            color = random.choice(colors)
            size = random.uniform(0.002, 0.006)
            particles.append({'x': cx, 'y': cy, 'vx': vx, 'vy': vy,
                              'color': color, 'size': size, 'life': duration})

        # 存储烟花粒子图形对象
        firework_patches = []

        def animate_fireworks(frame):
            # 清除上一帧粒子
            for p in firework_patches:
                try:
                    p.remove()
                except Exception:
                    pass
            firework_patches.clear()

            # 更新粒子位置与生命周期
            for particle in particles:
                if particle['life'] <= 0:
                    continue
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                # 模拟重力下落
                particle['vy'] -= 0.0001
                particle['life'] -= 1
                # 创建散点圆表示粒子
                circle = patches.Circle((particle['x'], particle['y']),
                                        particle['size'],
                                        color=particle['color'],
                                        alpha=particle['life'] / duration,
                                        zorder=10)
                self.ax.add_patch(circle)
                firework_patches.append(circle)

        self.firework_anim = FuncAnimation(self.fig, animate_fireworks,
                                           frames=duration, interval=30, blit=False, repeat=False)
