from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Tuple

from maze_env import MazeEnv
from rl_brain import QLearningAgent

class MazeTrainerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("强化学习迷宫可视化 (Q-Learning)")
        self.cell_size = 72

        self.env = MazeEnv()
        self.agent = QLearningAgent(self.env.action_space_n)

        self.episode = 1
        self.step_count = 0
        self.episode_reward = 0.0
        self.max_steps_per_episode = 200
        self.running = False

        self._build_ui()
        self._draw_maze()
        self._draw_agent(self.env.current_state)
        self._update_metrics()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X)

        self.btn_start = ttk.Button(controls, text="开始训练", command=self.start_training)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_pause = ttk.Button(controls, text="暂停", command=self.pause_training)
        self.btn_pause.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_reset = ttk.Button(controls, text="重置", command=self.reset_training)
        self.btn_reset.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_demo = ttk.Button(controls, text="演示最优路径", command=self.demo_best_path)
        self.btn_demo.pack(side=tk.LEFT)

        ttk.Label(controls, text="速度(ms)").pack(side=tk.LEFT, padx=(16, 4))
        self.speed_var = tk.IntVar(value=80)
        self.speed_scale = ttk.Scale(controls, from_=20, to=400, variable=self.speed_var)
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        info = ttk.Frame(container)
        info.pack(fill=tk.X, pady=(10, 8))

        self.status_var = tk.StringVar(value="状态: 待开始")
        self.metrics_var = tk.StringVar(value="")

        ttk.Label(info, textvariable=self.status_var).pack(anchor=tk.W)
        ttk.Label(info, textvariable=self.metrics_var).pack(anchor=tk.W)

        self.margin = 36
        canvas_w = self.env.width * self.cell_size + self.margin
        canvas_h = self.env.height * self.cell_size + self.margin
        self.canvas = tk.Canvas(container, width=canvas_w, height=canvas_h, bg="#ffffff", highlightthickness=0)
        self.canvas.pack()

        legend_text = "图例: 蓝色=智能体, 黑色=障碍, 青色=终点[5,5]+10, 跳跃: [2,4]->[4,4] +5"
        ttk.Label(container, text=legend_text).pack(anchor=tk.W, pady=(8, 0))

    def _draw_maze(self) -> None:
        self.canvas.delete("cell")
        for r in range(self.env.height):
            for c in range(self.env.width):
                x1, y1 = self.margin + c * self.cell_size, self.margin + r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size

                color = "#f2f2f2"
                state = (r, c)
                if self.env.is_obstacle(state):
                    color = "#202020"
                elif state == self.env.goal_state:
                    color = "#63d7d1"
                elif state == self.env.start_state:
                    color = "#d8eeff"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#9a9a9a", tags="cell")
                if state == self.env.goal_state:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="+10", fill="#0a5550", font=("Consolas", 14, "bold"), tags="cell")
                elif state == self.env.start_state:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="S", fill="#1a1a1a", font=("Consolas", 14, "bold"), tags="cell")

        # 顶部与左侧坐标轴（1-based），与课件一致。
        for i in range(self.env.width):
            x = self.margin + i * self.cell_size + self.cell_size / 2
            self.canvas.create_text(x, self.margin / 2, text=str(i + 1), fill="#404040", font=("Consolas", 12, "bold"), tags="cell")
        for i in range(self.env.height):
            y = self.margin + i * self.cell_size + self.cell_size / 2
            self.canvas.create_text(self.margin / 2, y, text=str(i + 1), fill="#404040", font=("Consolas", 12, "bold"), tags="cell")

        # 跳跃标注：[2,4] -> [4,4]，+5。
        fr, fc = self.env.jump_from
        tr, tc = self.env.jump_to
        fx = self.margin + fc * self.cell_size + self.cell_size / 2
        fy = self.margin + fr * self.cell_size + self.cell_size / 2
        tx = self.margin + tc * self.cell_size + self.cell_size / 2
        ty = self.margin + tr * self.cell_size + self.cell_size / 2
        self.canvas.create_line(fx, fy, tx, ty, arrow=tk.LAST, width=2, fill="#3f8dd1", tags="cell")
        self.canvas.create_text((fx + tx) / 2 + 12, (fy + ty) / 2 - 12, text="+5", fill="#3f8dd1", font=("Consolas", 12, "bold"), tags="cell")

    def _draw_agent(self, state: Tuple[int, int]) -> None:
        self.canvas.delete("agent")
        r, c = state
        pad = 10
        x1 = self.margin + c * self.cell_size + pad
        y1 = self.margin + r * self.cell_size + pad
        x2 = self.margin + (c + 1) * self.cell_size - pad
        y2 = self.margin + (r + 1) * self.cell_size - pad
        self.canvas.create_oval(x1, y1, x2, y2, fill="#2f78dd", outline="", tags="agent")

    def _update_metrics(self) -> None:
        self.metrics_var.set(
            f"Episode: {self.episode} | Step: {self.step_count} | "
            f"Epsilon: {self.agent.epsilon:.3f} | Reward: {self.episode_reward:.3f}"
        )

    def start_training(self) -> None:
        if self.running:
            return
        self.running = True
        self.status_var.set("状态: 训练中")
        self._run_step_loop()

    def pause_training(self) -> None:
        self.running = False
        self.status_var.set("状态: 已暂停")

    def reset_training(self) -> None:
        self.running = False
        self.env.reset()
        self.agent = QLearningAgent(self.env.action_space_n)
        self.episode = 1
        self.step_count = 0
        self.episode_reward = 0.0
        self._draw_maze()
        self._draw_agent(self.env.current_state)
        self._update_metrics()
        self.status_var.set("状态: 已重置")

    def _next_episode(self) -> None:
        self.agent.decay_epsilon()
        self.episode += 1
        self.step_count = 0
        self.episode_reward = 0.0
        self.env.reset()

    def _run_step_loop(self) -> None:
        if not self.running:
            return

        state = self.env.current_state
        action = self.agent.choose_action(state)

        # 用于可视化：先计算动作落点，若命中 jump_from，则短暂停留后再显示 jump_to。
        dr, dc = self.env.ACTIONS[action]
        entered_state = (state[0] + dr, state[1] + dc)

        result = self.env.step(action)
        self.agent.learn(state, action, result.reward, result.next_state, result.done)

        self.step_count += 1
        self.episode_reward += result.reward

        self._draw_maze()
        if entered_state == self.env.jump_from and result.next_state == self.env.jump_to:
            self._draw_agent(self.env.jump_from)
            self.root.after(90, lambda: self._draw_agent(result.next_state))
            self.status_var.set("状态: 触发跳跃 [2,4] -> [4,4] (+5)")
        else:
            self._draw_agent(result.next_state)
        self._update_metrics()

        if result.done or self.step_count >= self.max_steps_per_episode:
            if result.next_state == self.env.goal_state:
                self.status_var.set(f"状态: 第 {self.episode} 回合到达终点")
            else:
                self.status_var.set(f"状态: 第 {self.episode} 回合结束")
            self._next_episode()

        delay = int(self.speed_var.get())
        self.root.after(delay, self._run_step_loop)

    def demo_best_path(self) -> None:
        self.running = False
        self.status_var.set("状态: 演示当前最优路径")

        self.env.reset()
        self._draw_maze()
        self._draw_agent(self.env.current_state)

        path: List[Tuple[int, int]] = [self.env.current_state]
        for _ in range(self.max_steps_per_episode):
            state = self.env.current_state
            action = self.agent.greedy_action(state)
            dr, dc = self.env.ACTIONS[action]
            entered_state = (state[0] + dr, state[1] + dc)
            result = self.env.step(action)
            # 与训练可视化一致：触发跳跃时先显示 jump_from，再显示 jump_to。
            if entered_state == self.env.jump_from and result.next_state == self.env.jump_to:
                path.append(self.env.jump_from)
            path.append(result.next_state)
            if result.done:
                break

        self._animate_path(path, 0)

    def _animate_path(self, path: List[Tuple[int, int]], idx: int) -> None:
        if idx >= len(path):
            return

        self._draw_maze()
        self._draw_agent(path[idx])

        if idx == len(path) - 1:
            if path[idx] == self.env.goal_state:
                self.status_var.set("状态: 演示结束，成功到达终点")
            else:
                self.status_var.set("状态: 演示结束，未到达终点")
            return

        # 演示速度比训练稍慢，方便观察路径与特殊跳跃。
        demo_delay = max(220, int(self.speed_var.get()) + 120)
        self.root.after(demo_delay, lambda: self._animate_path(path, idx + 1))


def run() -> None:
    root = tk.Tk()
    MazeTrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()
