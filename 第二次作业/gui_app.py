from __future__ import annotations

import io
import json
import threading
import traceback
from contextlib import redirect_stdout
from datetime import datetime
import tkinter as tk
from tkinter import ttk

import main as hw
from llm_client import LLMClient
from prompts import SIMPLE_PROMPT, SYSTEM_ROLE


class AssignmentGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("第二次作业 Prompt 实验 GUI")
        self.geometry("1120x760")
        self.minsize(980, 620)

        self.task_client = LLMClient()
        self.chat_client = LLMClient()
        self.chat_messages = [{"role": "system", "content": SYSTEM_ROLE}]
        self.task_running = False

        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.task_tab = ttk.Frame(notebook)
        self.chat_tab = ttk.Frame(notebook)

        notebook.add(self.task_tab, text="任务运行")
        notebook.add(self.chat_tab, text="角色聊天（任务4）")

        self._build_task_tab()
        self._build_chat_tab()

    def _build_task_tab(self) -> None:
        toolbar = ttk.Frame(self.task_tab)
        toolbar.pack(fill="x", padx=10, pady=10)

        self.task_buttons = []
        button_specs = [
            ("任务1", lambda: self.run_single_task("1")),
            ("任务2", lambda: self.run_single_task("2")),
            ("任务3", lambda: self.run_single_task("3")),
            ("任务5", lambda: self.run_single_task("5")),
            ("任务6", lambda: self.run_single_task("6")),
            ("运行全部(1,2,3,5,6)", self.run_all_tasks),
        ]

        for text, command in button_specs:
            btn = ttk.Button(toolbar, text=text, command=command)
            btn.pack(side="left", padx=4)
            self.task_buttons.append(btn)

        ttk.Button(toolbar, text="清空输出", command=self.clear_task_output).pack(
            side="left", padx=8
        )

        self.task_status_var = tk.StringVar(value="就绪")
        ttk.Label(toolbar, textvariable=self.task_status_var).pack(side="right")

        self.task_output = tk.Text(self.task_tab, wrap="word", font=("Consolas", 11))
        self.task_output.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _build_chat_tab(self) -> None:
        top = ttk.Frame(self.chat_tab)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="输入后点击发送，模型将以‘刻薄但专业的影评家’身份回复。").pack(
            side="left"
        )
        ttk.Button(top, text="重置聊天", command=self.reset_chat).pack(side="right")

        self.chat_output = tk.Text(self.chat_tab, wrap="word", font=("Consolas", 11))
        self.chat_output.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.chat_output.insert("end", "系统：聊天已初始化。\n")
        self.chat_output.configure(state="disabled")

        bottom = ttk.Frame(self.chat_tab)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        self.chat_input = ttk.Entry(bottom)
        self.chat_input.pack(side="left", fill="x", expand=True)
        self.chat_input.bind("<Return>", self._on_enter_send)

        self.chat_send_btn = ttk.Button(bottom, text="发送", command=self.send_chat)
        self.chat_send_btn.pack(side="left", padx=8)

    def _set_task_running(self, running: bool) -> None:
        self.task_running = running
        state = "disabled" if running else "normal"
        for btn in self.task_buttons:
            btn.configure(state=state)

    def _append_task_output(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.task_output.insert("end", f"[{timestamp}]\n{text}\n")
        self.task_output.see("end")

    def clear_task_output(self) -> None:
        self.task_output.delete("1.0", "end")

    def run_single_task(self, task_id: str) -> None:
        if self.task_running:
            self._append_task_output("已有任务在运行，请稍后再试。")
            return

        self._set_task_running(True)
        self.task_status_var.set(f"运行中：任务{task_id}")
        thread = threading.Thread(target=self._worker_single_task, args=(task_id,), daemon=True)
        thread.start()

    def run_all_tasks(self) -> None:
        if self.task_running:
            self._append_task_output("已有任务在运行，请稍后再试。")
            return

        self._set_task_running(True)
        self.task_status_var.set("运行中：任务1,2,3,5,6")
        thread = threading.Thread(target=self._worker_all_tasks, daemon=True)
        thread.start()

    def _worker_single_task(self, task_id: str) -> None:
        try:
            output = self._execute_task(task_id)
        except Exception:
            output = "发生异常：\n" + traceback.format_exc()

        self.after(0, lambda: self._finish_task_run(output))

    def _worker_all_tasks(self) -> None:
        try:
            chunks = [
                self._execute_task("1"),
                self._execute_task("2"),
                self._execute_task("3"),
                self._execute_task("5"),
                self._execute_task("6"),
            ]
            output = "\n\n".join(chunks)
        except Exception:
            output = "发生异常：\n" + traceback.format_exc()

        self.after(0, lambda: self._finish_task_run(output))

    def _execute_task(self, task_id: str) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            if task_id == "1":
                hw.task1_zero_vs_few_shot(self.task_client)
            elif task_id == "2":
                hw.task2_force_json(self.task_client)
            elif task_id == "3":
                hw.task3_cot(self.task_client)
            elif task_id == "5":
                print("=" * 20, "任务5：提示词打分器", "=" * 20)
                report = hw.evaluate_prompt(self.task_client, SIMPLE_PROMPT)
                print("SIMPLE_PROMPT 评分:")
                print(json.dumps(report, ensure_ascii=False, indent=2))
            elif task_id == "6":
                hw.task6_grid_search(self.task_client)
            else:
                print(f"未知任务编号：{task_id}")

        return buf.getvalue().strip() or f"任务{task_id}执行完毕（无输出）"

    def _finish_task_run(self, output: str) -> None:
        self._append_task_output(output)
        self.task_status_var.set("就绪")
        self._set_task_running(False)

    def _append_chat(self, speaker: str, text: str) -> None:
        self.chat_output.configure(state="normal")
        self.chat_output.insert("end", f"{speaker}: {text}\n\n")
        self.chat_output.see("end")
        self.chat_output.configure(state="disabled")

    def reset_chat(self) -> None:
        self.chat_messages = [{"role": "system", "content": SYSTEM_ROLE}]
        self.chat_output.configure(state="normal")
        self.chat_output.delete("1.0", "end")
        self.chat_output.insert("end", "系统：聊天已重置。\n\n")
        self.chat_output.configure(state="disabled")

    def _on_enter_send(self, event: tk.Event) -> None:
        self.send_chat()

    def send_chat(self) -> None:
        user_text = self.chat_input.get().strip()
        if not user_text:
            return

        self.chat_input.delete(0, "end")
        self._append_chat("你", user_text)
        self.chat_send_btn.configure(state="disabled")

        thread = threading.Thread(target=self._worker_chat, args=(user_text,), daemon=True)
        thread.start()

    def _worker_chat(self, user_text: str) -> None:
        try:
            self.chat_messages.append({"role": "user", "content": user_text})
            reply = self.chat_client.chat(self.chat_messages)
            self.chat_messages.append({"role": "assistant", "content": reply})
            self.after(0, lambda: self._finish_chat(reply))
        except Exception:
            err = "调用失败：\n" + traceback.format_exc()
            self.after(0, lambda: self._finish_chat(err))

    def _finish_chat(self, text: str) -> None:
        self._append_chat("影评家", text)
        self.chat_send_btn.configure(state="normal")


if __name__ == "__main__":
    app = AssignmentGUI()
    app.mainloop()
