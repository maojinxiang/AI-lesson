import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch

from dataset import extract_feature
from model import SpeechEmotionTransformer


class ProcessController:
    def __init__(self, log_callback, done_callback):
        self.log_callback = log_callback
        self.done_callback = done_callback
        self.process = None
        self.reader_thread = None
        self._queue = queue.Queue()

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, cmd, cwd):
        if self.is_running():
            raise RuntimeError("已有任务正在运行，请先停止当前任务")

        self.process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()

    def _reader(self):
        assert self.process is not None
        for line in self.process.stdout:
            self._queue.put(line)
        self.process.wait()
        self._queue.put(None)

    def poll(self):
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            if item is None:
                rc = self.process.returncode if self.process else -1
                self.done_callback(rc)
                self.process = None
                break
            self.log_callback(item)

    def stop(self):
        if self.is_running():
            self.process.terminate()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("语音情感识别工具箱")
        self.geometry("1180x820")
        self.minsize(1080, 760)

        self.workspace = Path(__file__).resolve().parent
        self.controller = ProcessController(self.append_log, self.on_process_done)

        self._setup_theme()
        self._build_ui()
        self.after(150, self._poll_logs)

    def _setup_theme(self):
        self.colors = {
            "bg": "#f3f5f7",
            "panel": "#ffffff",
            "text": "#1f2937",
            "subtext": "#667085",
            "accent": "#0f766e",
            "accent_hover": "#115e59",
            "danger": "#b42318",
            "danger_hover": "#912018",
            "log_bg": "#111827",
            "log_fg": "#d1e0ff",
        }

        self.configure(bg=self.colors["bg"])

        style = ttk.Style(self)
        style.theme_use("clam")

        base_font = ("Microsoft YaHei UI", 10)
        title_font = ("Microsoft YaHei UI", 18, "bold")
        sub_title_font = ("Microsoft YaHei UI", 10)

        style.configure(".", font=base_font)
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Card.TFrame", background=self.colors["panel"], relief="solid", borderwidth=1)
        style.configure("Panel.TLabelframe", background=self.colors["panel"], relief="solid", borderwidth=1)
        style.configure("Panel.TLabelframe.Label", background=self.colors["panel"], foreground=self.colors["text"])

        style.configure("Title.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=title_font)
        style.configure(
            "SubTitle.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["subtext"],
            font=sub_title_font,
        )
        style.configure("Card.TLabel", background=self.colors["panel"], foreground=self.colors["text"])
        style.configure(
            "Status.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["accent"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )

        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure("TCombobox", fieldbackground="#ffffff")

        style.configure("Accent.TButton", background=self.colors["accent"], foreground="#ffffff", padding=(12, 7))
        style.map(
            "Accent.TButton",
            background=[("active", self.colors["accent_hover"]), ("pressed", self.colors["accent_hover"])],
            foreground=[("active", "#ffffff"), ("pressed", "#ffffff")],
        )

        style.configure("Danger.TButton", background=self.colors["danger"], foreground="#ffffff", padding=(12, 7))
        style.map(
            "Danger.TButton",
            background=[("active", self.colors["danger_hover"]), ("pressed", self.colors["danger_hover"])],
            foreground=[("active", "#ffffff"), ("pressed", "#ffffff")],
        )

        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), background="#e7ecef", foreground=self.colors["text"])
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.colors["panel"]), ("active", "#d9e2e7")],
            foreground=[("selected", self.colors["accent"])],
        )

        style.configure("Treeview", rowheight=26)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_ui(self):
        top = ttk.Frame(self, style="Card.TFrame", padding=14)
        top.pack(fill=tk.X, padx=12, pady=(12, 8))

        ttk.Label(top, text="语音情感识别工具箱", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            top,
            text="一站式管理训练、测试与单条音频推理（CASIA）",
            style="SubTitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 6))
        ttk.Label(top, text=f"工作目录: {self.workspace}", style="SubTitle.TLabel").pack(anchor=tk.W)

        actions = ttk.Frame(self, style="Card.TFrame", padding=10)
        actions.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.run_btn = ttk.Button(actions, text="运行当前页任务", style="Accent.TButton", command=self.run_current_task)
        self.run_btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(actions, text="停止任务", style="Danger.TButton", command=self.stop_task)
        self.stop_btn.pack(side=tk.LEFT, padx=8)

        self.clear_btn = ttk.Button(actions, text="清空日志", command=self.clear_log)
        self.clear_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="状态: 就绪")
        ttk.Label(actions, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12)

        self.train_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        self.test_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        self.infer_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)

        self.notebook.add(self.train_tab, text="训练")
        self.notebook.add(self.test_tab, text="测试")
        self.notebook.add(self.infer_tab, text="单条推理")

        self._build_train_tab()
        self._build_test_tab()
        self._build_infer_tab()

        log_frame = ttk.LabelFrame(self, text="运行日志", style="Panel.TLabelframe", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            height=16,
            bg=self.colors["log_bg"],
            fg=self.colors["log_fg"],
            insertbackground="#ffffff",
            relief="flat",
            padx=10,
            pady=8,
            font=("Consolas", 10),
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scroll.set)

    def _row_entry(self, parent, row, label, var, width=28, browse=None):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky=tk.W, padx=6, pady=4)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=6, pady=4)
        if browse:
            ttk.Button(parent, text="浏览", command=browse).grid(row=row, column=2, sticky=tk.W, padx=6, pady=4)

    def _build_train_tab(self):
        f = self.train_tab

        self.train_data_dir = tk.StringVar(value="CASIA database")
        self.train_feature = tk.StringVar(value="mel")
        self.train_epochs = tk.StringVar(value="20")
        self.train_batch = tk.StringVar(value="16")
        self.train_lr = tk.StringVar(value="1e-4")
        self.train_d_model = tk.StringVar(value="128")
        self.train_nhead = tk.StringVar(value="4")
        self.train_layers = tk.StringVar(value="2")
        self.train_ffn = tk.StringVar(value="256")
        self.train_dropout = tk.StringVar(value="0.1")
        self.train_ratio = tk.StringVar(value="0.7")
        self.val_ratio = tk.StringVar(value="0.15")
        self.test_ratio = tk.StringVar(value="0.15")
        self.train_seed = tk.StringVar(value="42")
        self.train_ckpt = tk.StringVar(value="checkpoints/best_model.pt")
        self.train_vis = tk.StringVar(value="outputs/train")
        self.train_split = tk.StringVar(value="outputs/train/data_split.json")
        self.train_specaug = tk.BooleanVar(value=True)
        self.train_class_weight = tk.BooleanVar(value=False)

        self._row_entry(
            f,
            0,
            "数据目录",
            self.train_data_dir,
            browse=lambda: self._choose_dir(self.train_data_dir),
        )

        ttk.Label(f, text="特征类型", style="Card.TLabel").grid(row=1, column=0, sticky=tk.W, padx=6, pady=4)
        ttk.Combobox(f, textvariable=self.train_feature, values=["mel", "mfcc"], width=10, state="readonly").grid(
            row=1, column=1, sticky=tk.W, padx=6, pady=4
        )

        self._row_entry(f, 2, "epochs", self.train_epochs, width=12)
        self._row_entry(f, 3, "batch_size", self.train_batch, width=12)
        self._row_entry(f, 4, "lr", self.train_lr, width=12)
        self._row_entry(f, 5, "d_model", self.train_d_model, width=12)
        self._row_entry(f, 6, "nhead", self.train_nhead, width=12)
        self._row_entry(f, 7, "num_layers", self.train_layers, width=12)
        self._row_entry(f, 8, "dim_feedforward", self.train_ffn, width=12)
        self._row_entry(f, 9, "dropout", self.train_dropout, width=12)
        self._row_entry(f, 10, "train_ratio", self.train_ratio, width=12)
        self._row_entry(f, 11, "val_ratio", self.val_ratio, width=12)
        self._row_entry(f, 12, "test_ratio", self.test_ratio, width=12)
        self._row_entry(f, 13, "seed", self.train_seed, width=12)

        self._row_entry(
            f,
            14,
            "模型保存路径",
            self.train_ckpt,
            browse=lambda: self._choose_save_file(self.train_ckpt, [("PyTorch", "*.pt"), ("All", "*.*")]),
        )
        self._row_entry(f, 15, "训练输出目录", self.train_vis, browse=lambda: self._choose_dir(self.train_vis))
        self._row_entry(
            f,
            16,
            "划分清单路径",
            self.train_split,
            browse=lambda: self._choose_save_file(self.train_split, [("JSON", "*.json"), ("All", "*.*")]),
        )

        ttk.Checkbutton(f, text="启用 SpecAugment", variable=self.train_specaug).grid(
            row=17, column=0, sticky=tk.W, padx=6, pady=6
        )
        ttk.Checkbutton(f, text="启用类别权重", variable=self.train_class_weight).grid(
            row=17, column=1, sticky=tk.W, padx=6, pady=6
        )

        for i in range(3):
            f.grid_columnconfigure(i, weight=1 if i == 1 else 0)

    def _build_test_tab(self):
        f = self.test_tab

        self.test_ckpt = tk.StringVar(value="checkpoints/best_model.pt")
        self.test_data_dir = tk.StringVar(value="CASIA database")
        self.test_split_json = tk.StringVar(value="outputs/train/data_split.json")
        self.test_split_name = tk.StringVar(value="test")
        self.test_batch = tk.StringVar(value="32")
        self.test_vis = tk.StringVar(value="outputs/test")

        self._row_entry(
            f,
            0,
            "模型路径",
            self.test_ckpt,
            browse=lambda: self._choose_file(self.test_ckpt, [("PyTorch", "*.pt"), ("All", "*.*")]),
        )
        self._row_entry(
            f,
            1,
            "数据目录",
            self.test_data_dir,
            browse=lambda: self._choose_dir(self.test_data_dir),
        )
        self._row_entry(
            f,
            2,
            "划分清单",
            self.test_split_json,
            browse=lambda: self._choose_file(self.test_split_json, [("JSON", "*.json"), ("All", "*.*")]),
        )

        ttk.Label(f, text="评估集合", style="Card.TLabel").grid(row=3, column=0, sticky=tk.W, padx=6, pady=4)
        ttk.Combobox(
            f,
            textvariable=self.test_split_name,
            values=["train", "val", "test"],
            width=10,
            state="readonly",
        ).grid(row=3, column=1, sticky=tk.W, padx=6, pady=4)

        self._row_entry(f, 4, "batch_size", self.test_batch, width=12)
        self._row_entry(f, 5, "测试输出目录", self.test_vis, browse=lambda: self._choose_dir(self.test_vis))

        for i in range(3):
            f.grid_columnconfigure(i, weight=1 if i == 1 else 0)

    def _build_infer_tab(self):
        f = self.infer_tab

        self.infer_wav = tk.StringVar(value="")
        self.infer_ckpt = tk.StringVar(value="checkpoints/best_model.pt")
        self.infer_topk = tk.StringVar(value="3")
        self.infer_device = tk.StringVar(value="")
        self.infer_pred = tk.StringVar(value="预测结果: -")
        self.infer_conf = tk.StringVar(value="置信度: -")

        self._row_entry(
            f,
            0,
            "WAV 文件",
            self.infer_wav,
            browse=lambda: self._choose_file(
                self.infer_wav,
                [("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All", "*.*")],
            ),
        )
        self._row_entry(
            f,
            1,
            "模型路径",
            self.infer_ckpt,
            browse=lambda: self._choose_file(self.infer_ckpt, [("PyTorch", "*.pt"), ("All", "*.*")]),
        )
        self._row_entry(f, 2, "topk", self.infer_topk, width=12)

        ttk.Label(f, text="device", style="Card.TLabel").grid(row=3, column=0, sticky=tk.W, padx=6, pady=4)
        ttk.Combobox(
            f,
            textvariable=self.infer_device,
            values=["", "cpu", "cuda"],
            width=10,
            state="readonly",
        ).grid(row=3, column=1, sticky=tk.W, padx=6, pady=4)

        tip = (
            "说明: device 为空表示自动选择 (有 CUDA 则用 CUDA, 否则 CPU)。"
        )
        ttk.Label(f, text=tip, style="SubTitle.TLabel").grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=6, pady=10)

        infer_btn = ttk.Button(f, text="在界面推理并显示百分比", command=self.run_infer_local)
        infer_btn.grid(row=5, column=0, sticky=tk.W, padx=6, pady=6)

        ttk.Label(f, textvariable=self.infer_pred, style="Card.TLabel").grid(
            row=6, column=0, columnspan=3, sticky=tk.W, padx=6, pady=2
        )
        ttk.Label(f, textvariable=self.infer_conf, style="Card.TLabel").grid(
            row=7, column=0, columnspan=3, sticky=tk.W, padx=6, pady=2
        )

        table_frame = ttk.LabelFrame(f, text="复杂情感百分比（全部类别）")
        table_frame.grid(row=8, column=0, columnspan=3, sticky=tk.NSEW, padx=6, pady=8)

        self.prob_table = ttk.Treeview(table_frame, columns=("label", "prob"), show="headings", height=8)
        self.prob_table.heading("label", text="情感类别")
        self.prob_table.heading("prob", text="概率(%)")
        self.prob_table.column("label", width=200, anchor=tk.W)
        self.prob_table.column("prob", width=120, anchor=tk.CENTER)
        self.prob_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        table_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.prob_table.yview)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.prob_table.configure(yscrollcommand=table_scroll.set)

        for i in range(3):
            f.grid_columnconfigure(i, weight=1 if i == 1 else 0)
        f.grid_rowconfigure(8, weight=1)

    def _choose_file(self, var, filetypes):
        p = filedialog.askopenfilename(initialdir=str(self.workspace), filetypes=filetypes)
        if p:
            var.set(self._to_rel(p))

    def _choose_save_file(self, var, filetypes):
        p = filedialog.asksaveasfilename(initialdir=str(self.workspace), filetypes=filetypes)
        if p:
            var.set(self._to_rel(p))

    def _choose_dir(self, var):
        p = filedialog.askdirectory(initialdir=str(self.workspace))
        if p:
            var.set(self._to_rel(p))

    def _to_rel(self, path_str):
        p = Path(path_str)
        try:
            return str(p.resolve().relative_to(self.workspace))
        except Exception:
            return str(p)

    def append_log(self, text):
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def run_current_task(self):
        if self.controller.is_running():
            messagebox.showwarning("提示", "已有任务正在运行")
            return

        tab_idx = self.notebook.index(self.notebook.select())
        if tab_idx == 0:
            cmd = self.build_train_cmd()
            title = "训练"
        elif tab_idx == 1:
            cmd = self.build_test_cmd()
            title = "测试"
        else:
            cmd = self.build_infer_cmd()
            title = "单条推理"

        self.append_log(f"\n===== 开始 {title} =====\n")
        self.append_log("命令: " + " ".join(cmd) + "\n\n")
        self.status_var.set(f"状态: 运行中 ({title})")
        try:
            self.controller.start(cmd, cwd=str(self.workspace))
        except Exception as e:
            messagebox.showerror("运行失败", str(e))
            self.status_var.set("状态: 就绪")

    def stop_task(self):
        if self.controller.is_running():
            self.controller.stop()
            self.append_log("\n[GUI] 已发送停止信号。\n")
            self.status_var.set("状态: 正在停止任务")

    def on_process_done(self, return_code):
        self.append_log(f"\n===== 任务结束，退出码: {return_code} =====\n")
        self.status_var.set("状态: 就绪")

    def _poll_logs(self):
        self.controller.poll()
        self.after(150, self._poll_logs)

    def build_train_cmd(self):
        cmd = [
            sys.executable,
            "train.py",
            "--data_dir",
            self.train_data_dir.get().strip(),
            "--feature_type",
            self.train_feature.get().strip(),
            "--epochs",
            self.train_epochs.get().strip(),
            "--batch_size",
            self.train_batch.get().strip(),
            "--lr",
            self.train_lr.get().strip(),
            "--d_model",
            self.train_d_model.get().strip(),
            "--nhead",
            self.train_nhead.get().strip(),
            "--num_layers",
            self.train_layers.get().strip(),
            "--dim_feedforward",
            self.train_ffn.get().strip(),
            "--dropout",
            self.train_dropout.get().strip(),
            "--train_ratio",
            self.train_ratio.get().strip(),
            "--val_ratio",
            self.val_ratio.get().strip(),
            "--test_ratio",
            self.test_ratio.get().strip(),
            "--seed",
            self.train_seed.get().strip(),
            "--save_path",
            self.train_ckpt.get().strip(),
            "--vis_dir",
            self.train_vis.get().strip(),
            "--split_path",
            self.train_split.get().strip(),
        ]

        if self.train_specaug.get():
            cmd.append("--use_specaug")
        if self.train_class_weight.get():
            cmd.append("--use_class_weight")
        return cmd

    def build_test_cmd(self):
        cmd = [
            sys.executable,
            "test.py",
            "--ckpt",
            self.test_ckpt.get().strip(),
            "--data_dir",
            self.test_data_dir.get().strip(),
            "--split_json",
            self.test_split_json.get().strip(),
            "--split_name",
            self.test_split_name.get().strip(),
            "--batch_size",
            self.test_batch.get().strip(),
            "--vis_dir",
            self.test_vis.get().strip(),
        ]
        return cmd

    def build_infer_cmd(self):
        wav = self.infer_wav.get().strip()
        if not wav:
            raise RuntimeError("请先选择 wav 文件")

        cmd = [
            sys.executable,
            "infer_wav.py",
            "--wav_path",
            wav,
            "--ckpt",
            self.infer_ckpt.get().strip(),
            "--topk",
            self.infer_topk.get().strip(),
        ]
        device = self.infer_device.get().strip()
        if device:
            cmd.extend(["--device", device])
        return cmd

    def _build_model_from_checkpoint(self, checkpoint, device):
        train_args = checkpoint["args"]
        label2id = checkpoint["label2id"]

        input_dim = train_args["n_mels"] if train_args["feature_type"] == "mel" else train_args["n_mfcc"]
        model = SpeechEmotionTransformer(
            input_dim=input_dim,
            num_classes=len(label2id),
            d_model=train_args["d_model"],
            nhead=train_args["nhead"],
            num_layers=train_args["num_layers"],
            dim_feedforward=train_args["dim_feedforward"],
            dropout=train_args["dropout"],
            max_len=train_args["seq_len"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, train_args, label2id

    def _preprocess_single_wav(self, wav_path: str, train_args: dict):
        feat = extract_feature(
            wav_path,
            sr=train_args["sr"],
            n_mels=train_args["n_mels"],
            n_mfcc=train_args["n_mfcc"],
            feature_type=train_args["feature_type"],
        )

        t = feat.shape[1]
        seq_len = train_args["seq_len"]
        if t >= seq_len:
            feat = feat[:, :seq_len]
        else:
            feat = np.pad(feat, ((0, 0), (0, seq_len - t)), mode="constant")

        x = torch.tensor(feat.T, dtype=torch.float32).unsqueeze(0)
        return x

    def run_infer_local(self):
        wav = self.infer_wav.get().strip()
        ckpt = self.infer_ckpt.get().strip()
        if not wav:
            messagebox.showwarning("提示", "请先选择 wav 文件")
            return

        wav_path = (self.workspace / wav).resolve() if not Path(wav).is_absolute() else Path(wav)
        ckpt_path = (self.workspace / ckpt).resolve() if not Path(ckpt).is_absolute() else Path(ckpt)

        if not wav_path.exists():
            messagebox.showerror("错误", f"Wav 文件不存在: {wav_path}")
            return
        if not ckpt_path.exists():
            messagebox.showerror("错误", f"模型文件不存在: {ckpt_path}")
            return

        try:
            device_value = self.infer_device.get().strip()
            device = torch.device(device_value if device_value else ("cuda" if torch.cuda.is_available() else "cpu"))

            checkpoint = torch.load(str(ckpt_path), map_location=device)
            model, train_args, label2id = self._build_model_from_checkpoint(checkpoint, device)
            id2label = {v: k for k, v in label2id.items()}

            x = self._preprocess_single_wav(str(wav_path), train_args).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1).squeeze(0)

            pred_id = int(torch.argmax(probs).item())
            pred_label = id2label[pred_id]
            pred_prob = float(probs[pred_id].item()) * 100.0

            self.infer_pred.set(f"预测结果: {pred_label}")
            self.infer_conf.set(f"置信度: {pred_prob:.2f}%")

            for item in self.prob_table.get_children():
                self.prob_table.delete(item)

            sorted_items = sorted(
                [(id2label[i], float(probs[i].item()) * 100.0) for i in range(probs.numel())],
                key=lambda x: x[1],
                reverse=True,
            )
            for label, p in sorted_items:
                self.prob_table.insert("", tk.END, values=(label, f"{p:.2f}"))

            self.append_log(
                f"\n[GUI推理] wav={wav_path.name}, pred={pred_label}, conf={pred_prob:.2f}%\n"
            )
        except Exception as e:
            messagebox.showerror("推理失败", str(e))


if __name__ == "__main__":
    app = App()
    app.mainloop()
