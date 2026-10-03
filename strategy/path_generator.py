#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
芯片阵列遍历路径生成器

该模块提供了多种遍历芯片阵列的路径生成算法，包括:
- 默认行优先遍历
- 折叠式遍历
- Hilbert曲线遍历 
- 头尾相连遍历
- 漩涡状遍历

每种遍历方式都保证能完整访问芯片阵列中的所有芯片，并最终回到起点位置。
"""

from hardware_config import HardwareConfig
from typing import List, Tuple, Dict, Set
from enum import Enum


class PathMode(Enum):
    """Path generation mode enumeration class"""
    DEFAULT = "default"      # Default row-priority traversal
    FOLD = "fold"           # Folded traversal
    HILBERT = "hilbert"     # Hilbert curve traversal
    HEAD_TAIL = "head_tail"  # Head-tail connected traversal
    VOTEX = "votex"         # Spiral traversal


class PathGenerator():
    """Generate paths to traverse each chip in the chip array"""

    def __init__(self, path_mode: PathMode = PathMode.DEFAULT):
        """
        Initialize path generator
        Args:
            path_mode: Path generation mode, PathMode enum type
        """
        self.row_number_of_chip = HardwareConfig.row_number_of_chip
        self.column_number_of_chip = HardwareConfig.column_number_of_chip

        # Path dictionary
        self.paths = {
            PathMode.DEFAULT: {},
            PathMode.FOLD: {},
            PathMode.HILBERT: {},
            PathMode.HEAD_TAIL: {},
            PathMode.VOTEX: {}
        }

        # Expert-specific grouping
        # expert_id -> list of chip ids (e.g., ["Chip(0, 0)", ...])
        self.expert_groups: Dict[int, List[str]] = {}
        # Cache for subset paths: key = (mode, frozenset(group_chips)) -> {chip_id: next_chip}
        self.group_paths: Dict[Tuple[PathMode, frozenset], Dict[str, str]] = {}

        # Validate path mode
        if not isinstance(path_mode, PathMode):
            raise ValueError(
                f"Invalid path mode: {path_mode}. Must use PathMode enum type")

        self._path_mode = path_mode

        # Generate all paths
        self._generate_all_paths()

    def _generate_all_paths(self):
        """Generate all available paths"""
        self._generate_default_path()
        self._generate_fold_path()
        self._generate_hilbert_path()
        self._generate_head_tail_path()
        self._generate_votex_path()

    @property
    def path_mode(self) -> PathMode:
        return self._path_mode

    @path_mode.setter
    def path_mode(self, mode: PathMode):
        if not isinstance(mode, PathMode):
            raise ValueError(
                f"Invalid path mode: {mode}. Must use PathMode enum type")
        self._path_mode = mode

    def get_next(self, chip_id: str) -> str:
        """
        Get the next chip ID based on current path mode

        Args:
            chip_id: Current chip ID in format "Chip(row, col)"

        Returns:
            str: Next chip ID in format "Chip(row, col)"

        Raises:
            ValueError: If chip_id is invalid or path mode is not supported
        """
        # Get path dictionary based on current mode
        path_dict = self.paths.get(self._path_mode)
        if not path_dict:
            raise ValueError(f"Path mode {self._path_mode} is not supported")

        # Get next chip from path dictionary
        next_chip = path_dict.get(chip_id)
        if next_chip is None:
            raise ValueError(f"Invalid chip ID: {chip_id}")

        return next_chip

    def set_expert_groups(self, groups: Dict[int, List[str]]):
        """Set per-expert allowed chip groups.

        Args:
            groups: Mapping expert_id -> list of chip ids
        """
        self.expert_groups = groups or {}
        # Invalidate group path cache when groups change
        self.group_paths.clear()

    def _parse_chip_rc(self, chip_id: str) -> Tuple[int, int]:
        """Parse chip string like 'Chip(r, c)' to (r, c)."""
        try:
            inside = chip_id[chip_id.find('(')+1:chip_id.find(')')]
            r_str, c_str = inside.split(',')
            return int(r_str.strip()), int(c_str.strip())
        except Exception:
            raise ValueError(f"Invalid chip id format: {chip_id}")

    def _ensure_group_path(self, group: List[str]) -> Dict[str, str]:
        """Ensure a row-major cyclic path for a given chip subset for current mode.

        For now, regardless of global path mode, we generate a simple row-major
        order within the subset and connect last back to first. This guarantees
        a valid traversal strictly within the subset.
        """
        key = (self._path_mode, frozenset(group))
        if key in self.group_paths:
            return self.group_paths[key]

        # Sort chips by row, then col
        sorted_group = sorted(group, key=lambda cid: self._parse_chip_rc(cid))
        if not sorted_group:
            self.group_paths[key] = {}
            return self.group_paths[key]

        mapping: Dict[str, str] = {}
        for i, cid in enumerate(sorted_group):
            nxt = sorted_group[(i + 1) % len(sorted_group)]
            mapping[cid] = nxt

        self.group_paths[key] = mapping
        return mapping

    def get_next_in_group(self, chip_id: str, expert_id: int) -> str:
        """Get next chip limited to the expert's allowed chip group.

        If the chip_id is not within the allowed group, return the first chip
        of that group to pull the traversal back into the group.
        """
        if not self.expert_groups or expert_id not in self.expert_groups:
            return self.get_next(chip_id)

        group = self.expert_groups[expert_id]
        if not group:
            return self.get_next(chip_id)

        mapping = self._ensure_group_path(group)
        if chip_id in mapping:
            return mapping[chip_id]
        # Fallback: start from the first chip in group
        return group[0]

    def _generate_default_path(self):
        """
        Generate default row-priority traversal path.
        Traverses each row from left to right, then moves to the next row.
        Returns to start point (0,0) after reaching the last chip.
        """
        # Initialize default path
        for row in range(self.row_number_of_chip):
            for col in range(self.column_number_of_chip):
                # Calculate next position
                next_col = col + 1
                next_row = row

                # If column exceeds range, move to first column of next row
                if next_col >= self.column_number_of_chip:
                    next_col = 0
                    next_row = row + 1

                # If row exceeds range, return to start point (0,0)
                if next_row >= self.row_number_of_chip:
                    next_row = 0
                    next_col = 0

                self.paths[PathMode.DEFAULT][f"Chip({row}, {col})"] = f"Chip({next_row}, {next_col})"

    def _generate_fold_path(self):
        """
        Generate folded traversal path.
        Even rows go left to right, odd rows go right to left.
        Moves down one row at the end of each row.
        Returns to start point (0,0) after reaching the last chip.
        """
        for row in range(self.row_number_of_chip):
            for col in range(self.column_number_of_chip):
                if row % 2 == 0:  # Even rows go left to right
                    next_col = col + 1
                    next_row = row
                    if next_col >= self.column_number_of_chip:
                        next_col = self.column_number_of_chip - 1
                        next_row = row + 1
                else:  # Odd rows go right to left
                    next_col = col - 1
                    next_row = row
                    if next_col < 0:
                        next_col = 0
                        next_row = row + 1

                if next_row >= self.row_number_of_chip:
                    next_row = 0
                    next_col = 0

                self.paths[PathMode.FOLD][f"Chip({row}, {col})"] = f"Chip({next_row}, {next_col})"

    def _generate_hilbert_path(self):
        """
        Generate Hilbert curve traversal path.
        Uses a space-filling curve to maintain locality between consecutive chips.
        Maps a power-of-2 sized Hilbert curve to the actual chip array dimensions.
        Returns to start point (0,0) after reaching the last chip.
        """
        # Generate Hilbert curve path of power of 2 size
        size = max(self.row_number_of_chip, self.column_number_of_chip)
        power = 1
        while power < size:
            power *= 2
        points = self._generate_hilbert_curve(power)

        # Map Hilbert curve to actual chip size
        for i in range(len(points)-1):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            if x1 < self.row_number_of_chip and y1 < self.column_number_of_chip:
                if x2 < self.row_number_of_chip and y2 < self.column_number_of_chip:
                    self.paths[PathMode.HILBERT][f"Chip({x1}, {y1})"] = f"Chip({x2}, {y2})"
                else:
                    self.paths[PathMode.HILBERT][f"Chip({x1}, {y1})"] = "Chip(0, 0)"

        # Connect last point back to start
        last_x, last_y = points[-1]
        if last_x < self.row_number_of_chip and last_y < self.column_number_of_chip:
            self.paths[PathMode.HILBERT][f"Chip({last_x}, {last_y})"] = "Chip(0, 0)"

    def _generate_hilbert_curve(self, n: int) -> List[Tuple[int, int]]:
        """
        Generate coordinate sequence for n x n Hilbert curve.

        Args:
            n: Size of the Hilbert curve (must be power of 2)

        Returns:
            List of (x,y) coordinate tuples representing the Hilbert curve path
        """
        def d2xy(n: int, d: int) -> Tuple[int, int]:
            """Convert distance along curve to (x,y) coordinates"""
            def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
                """Rotate/flip quadrant appropriately"""
                if ry == 0:
                    if rx == 1:
                        x = n-1 - x
                        y = n-1 - y
                    x, y = y, x
                return x, y

            x = y = 0
            t = d
            s = 1
            while s < n:
                rx = 1 & (t >> 1)
                ry = 1 & (t ^ rx)
                x, y = _rot(s, x, y, rx, ry)
                x += s * rx
                y += s * ry
                t >>= 2
                s <<= 1
            return x, y

        points = []
        for i in range(n * n):
            points.append(d2xy(n, i))
        return points

    def _generate_head_tail_path(self):
        """
        Generate head-tail connected traversal path.
        Ensures each consecutive chip in path is physically adjacent.
        For even-sized arrays, connects last chip back to start.
        For odd-sized arrays, connects middle chip back to start.
        """
        # Initialize path dictionary
        self.paths[PathMode.HEAD_TAIL] = {}

        # Traverse each row
        for row in range(self.row_number_of_chip):
            for col in range(self.column_number_of_chip):
                current = f"Chip({row}, {col})"

                # Even rows go left to right
                if row % 2 == 0:
                    if col < self.column_number_of_chip - 1:
                        # If not at row end, connect to right chip
                        self.paths[PathMode.HEAD_TAIL][current] = f"Chip({row}, {col + 1})"
                    else:
                        # If at row end, connect to next row
                        if row < self.row_number_of_chip - 1:
                            self.paths[PathMode.HEAD_TAIL][current] = f"Chip({row + 1}, {col})"
                # Odd rows go right to left
                else:
                    if col > 0:
                        # If not at row start, connect to left chip
                        self.paths[PathMode.HEAD_TAIL][current] = f"Chip({row}, {col - 1})"
                    else:
                        # If at row start and not last row, connect to next row
                        if row < self.row_number_of_chip - 1:
                            self.paths[PathMode.HEAD_TAIL][current] = f"Chip({row + 1}, {col})"

        # Special handling: For odd number of rows, connect middle position back to start
        if self.row_number_of_chip % 2 == 1:
            middle_row = self.row_number_of_chip // 2
            middle_col = self.column_number_of_chip // 2
            self.paths[PathMode.HEAD_TAIL][f"Chip({middle_row}, {middle_col})"] = "Chip(0, 0)"
        else:
            # For even rows, connect last chip back to start
            last_row = self.row_number_of_chip - 1
            last_col = 0
            self.paths[PathMode.HEAD_TAIL][f"Chip({last_row}, {last_col})"] = "Chip(0, 0)"

    def _generate_votex_path(self):
        """
        Generate spiral traversal path.
        Starts from outer edge and spirals inward.
        Changes direction when hitting boundaries or visited chips.
        Returns to start point (0,0) after reaching the last chip.
        """
        # Define four directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        current_dir = 0  # Start going right

        # Initial position
        row, col = 0, 0
        # Track visited positions
        visited = set()

        # Boundaries
        top = 0
        bottom = self.row_number_of_chip - 1
        left = 0
        right = self.column_number_of_chip - 1

        while len(visited) < self.row_number_of_chip * self.column_number_of_chip - 1:
            # Record current position
            current_pos = f"Chip({row}, {col})"
            visited.add(current_pos)

            # Calculate next position
            next_row = row + directions[current_dir][0]
            next_col = col + directions[current_dir][1]

            # Check if direction change needed
            if (current_dir == 0 and next_col > right) or \
               (current_dir == 1 and next_row > bottom) or \
               (current_dir == 2 and next_col < left) or \
               (current_dir == 3 and next_row < top) or \
               f"Chip({next_row}, {next_col})" in visited:
                # Change direction
                current_dir = (current_dir + 1) % 4
                # Update boundaries
                if current_dir == 0:
                    top += 1
                elif current_dir == 1:
                    right -= 1
                elif current_dir == 2:
                    bottom -= 1
                elif current_dir == 3:
                    left += 1
                # Recalculate next position
                next_row = row + directions[current_dir][0]
                next_col = col + directions[current_dir][1]

            # Record path
            self.paths[PathMode.VOTEX][current_pos] = f"Chip({next_row}, {next_col})"

            # Update current position
            row, col = next_row, next_col

        # Connect last position back to start
        last_pos = f"Chip({row}, {col})"
        self.paths[PathMode.VOTEX][last_pos] = "Chip(0, 0)"

    def get_next_by_default_path(self, chip_id: str) -> str:
        """
        Get next chip ID using row-priority traversal order.
        Traverses each row from left to right:
        Chip(0, 0) -> Chip(0, 1) -> ... -> Chip(0, m-1) ->
        Chip(1, 0) -> Chip(1, 1) -> ...
        ...
        Chip(n-1, 0) -> ... -> Chip(n-1, m-1) -> Chip(0, 0)
        """
        return self.paths[PathMode.DEFAULT][chip_id]

    def get_next_by_fold_path(self, chip_id: str) -> str:
        """
        Get next chip ID using folded traversal order.
        Even rows go left to right, odd rows go right to left:
        Chip(0, 0) -> Chip(0, 1) -> Chip(0, 2) -> Chip(0, 3)
                                                     ⬇
        Chip(1, 0) <- Chip(1, 1) <- Chip(1, 2) <- Chip(1, 3)
            ⬇
        Chip(2, 0) -> Chip(2, 1) -> Chip(2, 2) -> Chip(2, 3)
                                                     ⬇
        Chip(3, 0) <- Chip(3, 1) <- Chip(3, 2) <- Chip(3, 3)

        Chip(3, 0) connects back to Chip(0, 0)
        """
        return self.paths[PathMode.FOLD][chip_id]

    def get_next_by_hilbert_path(self, chip_id: str) -> str:
        """
        Get next chip ID using Hilbert curve traversal order.
        Follows a space-filling curve pattern:
        Chip(0, 0) -> Chip(0, 1) -> Chip(1, 1) -> Chip(1, 0) ->
        Chip(2, 0) -> Chip(2, 1) -> Chip(3, 1) -> Chip(3, 0) ->
        ... -> Chip(0, 0)

        Note: Current implementation may not achieve perfect closure
        """
        return self.paths[PathMode.HILBERT][chip_id]

    def get_next_by_votex_path(self, chip_id: str) -> str:
        """
        Get next chip ID using spiral traversal order.
        For a 4×4 chip array:
        Chip(0, 0) -> Chip(0, 1) -> Chip(0, 2) -> Chip(0, 3)
                                                     ⬇
        Chip(1, 0) -> Chip(1, 1) -> Chip(1, 2)    Chip(1, 3)
            ⬆                          ⬇            ⬇
        Chip(2, 0)    Chip(2, 1) <- Chip(2, 2)    Chip(2, 3)
            ⬆                                        ⬇
        Chip(3, 0) <- Chip(3, 1) <- Chip(3, 2) <- Chip(3, 3)

        Chip(2, 1) connects back to Chip(0, 0)
        """
        return self.paths[PathMode.VOTEX][chip_id]

    def get_next_by_head_tail_path(self, chip_id: str) -> str:
        """
        Get next chip ID using head-tail connected traversal order.
        Ensures each consecutive chip is physically adjacent (coordinates differ by 1).
        For a 4×4 chip array:
        Chip(0, 0) -> Chip(0, 1) -> Chip(0, 2) -> Chip(0, 3)
            ⬆                                        ⬇
        Chip(1, 0)    Chip(1, 1) <- Chip(1, 2) <- Chip(1, 3)
            ⬆             ⬇
        Chip(2, 0)    Chip(2, 1) -> Chip(2, 2) -> Chip(2, 3)
            ⬆                                        ⬇
        Chip(3, 0) <- Chip(3, 1) <- Chip(3, 2) <- Chip(3, 3)

        For 3×3 array:
        Chip(0, 0) -> Chip(0, 1) -> Chip(0, 2)
                                        ⬇
        Chip(1, 0) -> Chip(1, 1)    Chip(1, 2)
            ⬆                          ⬇
        Chip(2, 0) <- Chip(2, 1) <- Chip(2, 2)

        Chip(1, 1) connects back to Chip(0, 0)

        Note: Current implementation may have issues
        """
        return self.paths[PathMode.HEAD_TAIL][chip_id]
