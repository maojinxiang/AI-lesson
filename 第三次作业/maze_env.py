from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Tuple


State = Tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    next_state: State
    reward: float
    done: bool


class MazeEnv:
    """按课程示意图构建的 5x5 离散迷宫环境。"""

    ACTIONS = {
        0: (-1, 0),  # 上
        1: (1, 0),   # 下
        2: (0, -1),  # 左
        3: (0, 1),   # 右
    }

    def __init__(self) -> None:
        self.height = 5
        self.width = 5

        # 课程图片坐标使用 1-based，下方状态使用 0-based。
        self.start_state: State = (1, 0)  # [2,1]
        self.goal_state: State = (4, 4)   # [5,5]

        # 黑色障碍块：[(3,3), (3,4), (3,5), (4,3)] (1-based)
        self.obstacles: Set[State] = {
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 2),
        }

        # 特殊奖励跃迁：[2,4] -> [4,4]，奖励 +5（1-based）
        self.jump_from: State = (1, 3)
        self.jump_to: State = (3, 3)

        self.current_state = self.start_state

    def reset(self) -> State:
        self.current_state = self.start_state
        return self.current_state

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def is_obstacle(self, state: State) -> bool:
        return state in self.obstacles

    def step(self, action: int) -> StepResult:
        # 规则 6：除特殊奖励与终点奖励外，默认每步 -1。
        dr, dc = self.ACTIONS[action]
        nr = self.current_state[0] + dr
        nc = self.current_state[1] + dc

        # 越界或撞障碍，位置不变，奖励 -1。
        if not self.in_bounds(nr, nc) or self.is_obstacle((nr, nc)):
            return StepResult(self.current_state, -1.0, False)

        next_state = (nr, nc)

        # 规则 4：从 [2,4] 进入后立即跃迁到 [4,4]，奖励 +5。
        if next_state == self.jump_from:
            self.current_state = self.jump_to
            return StepResult(self.jump_to, 5.0, False)

        self.current_state = next_state

        # 规则 3：到达 [5,5] 奖励 +10，回合结束。
        if next_state == self.goal_state:
            return StepResult(next_state, 10.0, True)

        return StepResult(next_state, -1.0, False)

    @property
    def action_space_n(self) -> int:
        return 4
