from __future__ import annotations

import csv
import os
import queue
import json
import subprocess
import sys
import threading
import tkinter as tk
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
VENV_PYTHONW = PROJECT_ROOT / ".venv" / "Scripts" / "pythonw.exe"
CONFIG = PROJECT_ROOT / "config.toml"
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

LLM_PRESETS = {
    "Local heuristic": {
        "provider": "heuristic",
        "model": "local-heuristic-v1",
        "base_url": "local",
        "api_key_env": "",
        "rate_limit_per_minute": "3000",
    },
    "OpenAI-compatible": {
        "provider": "openai_compatible",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "rate_limit_per_minute": "60",
    },
    "DeepSeek-compatible": {
        "provider": "openai_compatible",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "rate_limit_per_minute": "60",
    },
    "Qwen-compatible": {
        "provider": "openai_compatible",
        "model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "rate_limit_per_minute": "60",
    },
    "OpenRouter-compatible": {
        "provider": "openai_compatible",
        "model": "openai/gpt-4o-mini",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "rate_limit_per_minute": "60",
    },
    "Custom OpenAI-compatible": {
        "provider": "openai_compatible",
        "model": "",
        "base_url": "",
        "api_key_env": "OPENAI_API_KEY",
        "rate_limit_per_minute": "60",
    },
}


def configure_launch_environment() -> None:
    os.chdir(PROJECT_ROOT)
    os.environ["PYTHONPATH"] = str(SRC_DIR)
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def relaunch_with_project_venv() -> None:
    if os.environ.get("AIQRS_UI_RELAUNCHED") == "1":
        return
    launcher = VENV_PYTHONW if VENV_PYTHONW.exists() else VENV_PYTHON
    if not launcher.exists():
        return
    current = Path(sys.executable).resolve()
    if current == launcher.resolve():
        return
    env = os.environ.copy()
    env["AIQRS_UI_RELAUNCHED"] = "1"
    env["PYTHONPATH"] = str(SRC_DIR)
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    subprocess.Popen([str(launcher), str(Path(__file__).resolve())], cwd=PROJECT_ROOT, env=env, startupinfo=startupinfo)
    raise SystemExit(0)


class PipelineUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Quant Research System")
        self.geometry("1280x900")
        self.minsize(1080, 760)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running = False
        self.worker: threading.Thread | None = None
        self.active_process: subprocess.Popen[str] | None = None
        self.log_window: tk.Toplevel | None = None
        self.log_text: tk.Text | None = None
        self.report_image: tk.PhotoImage | None = None
        self.report_canvas_image_id: int | None = None
        self.report_image_path: Path | None = None
        self.mpl_canvas = None
        self.mpl_toolbar = None
        self.hover_annotation = None
        self.hover_vlines = []
        self.right_scroll_target: tk.Widget | None = None
        self.right_scroll_target_kind: str | None = None
        self.total_steps = 1
        self._build_style()
        self._build_layout()
        self._load_ui_config()
        self.refresh_universe_status()
        self.after(100, self._drain_log_queue)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Muted.TLabel", font=("Segoe UI", 9), foreground="#5f6b7a")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=24)

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="AI Quant Research System", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            container,
            text="Local pipeline for data collection, news scoring, factor building, and prediction modeling.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        body = ttk.Frame(container)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 14))

        controls = ttk.LabelFrame(left, text="Workflow", padding=12)
        controls.pack(fill="x")
        ttk.Button(controls, text="1. Fetch / Update Data", style="Primary.TButton", command=self.run_data_pipeline).pack(fill="x", pady=(0, 6))
        ttk.Button(controls, text="2. Run Modeling Pipeline", style="Primary.TButton", command=self.run_modeling_pipeline).pack(fill="x", pady=(0, 10))
        ttk.Button(controls, text="Full Pipeline", command=self.run_full_pipeline).pack(fill="x", pady=3)
        ttk.Separator(controls).pack(fill="x", pady=8)

        for label, command in [
            ("Initialize DB", self.run_init_db),
            ("Fetch Prices", self.run_prices),
            ("Fetch S&P 500 News", self.run_news),
            ("Build News Factors", self.run_news_factors),
            ("Build Factor Table", self.run_factor_table),
            ("Train Models", self.run_models),
            ("Train Walk-Forward", self.run_walk_forward_models),
            ("Run Backtest", self.run_backtest),
            ("Optimize Portfolio", self.run_optimize_portfolio),
            ("Run Trust Audit", self.run_trust_audit),
            ("Import Historical Universe", self.import_historical_universe),
            ("Check Universe", self.refresh_universe_status),
            ("Refresh Row Counts", self.run_counts),
        ]:
            ttk.Button(controls, text=label, command=command).pack(fill="x", pady=3)

        settings = ttk.LabelFrame(left, text="Data Settings", padding=12)
        settings.pack(fill="x", pady=(14, 0))
        self.history_days = tk.StringVar(value="730")
        self.yahoo_news_limit = tk.StringVar(value="3")
        self.sec_days = tk.StringVar(value="90")
        self.sec_limit = tk.StringVar(value="2")
        self.score_limit = tk.StringVar(value="5000")
        self.force_refresh = tk.BooleanVar(value=False)
        self.require_pit_universe = tk.BooleanVar(value=False)
        self._field(settings, "Price days", self.history_days)
        self._field(settings, "Yahoo news/ticker", self.yahoo_news_limit)
        self._field(settings, "SEC lookback days", self.sec_days)
        self._field(settings, "SEC filings/ticker", self.sec_limit)
        self._field(settings, "Scoring limit", self.score_limit)
        ttk.Checkbutton(settings, text="Force refresh existing data", variable=self.force_refresh).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(settings, text="Require historical PIT universe", variable=self.require_pit_universe).pack(anchor="w", pady=(4, 0))

        scoring = ttk.LabelFrame(left, text="News Scoring", padding=12)
        scoring.pack(fill="x", pady=(14, 0))
        self.scorer_provider = tk.StringVar(value="heuristic")
        self.llm_preset = tk.StringVar(value="Local heuristic")
        self.llm_model = tk.StringVar(value="local-heuristic-v1")
        self.llm_base_url = tk.StringVar(value="local")
        self.llm_api_key_env = tk.StringVar(value="")
        self.llm_rate_limit = tk.StringVar(value="3000")
        ttk.Radiobutton(scoring, text="Local heuristic", variable=self.scorer_provider, value="heuristic").pack(anchor="w")
        ttk.Radiobutton(scoring, text="External API (OpenAI-compatible)", variable=self.scorer_provider, value="openai_compatible").pack(anchor="w")
        preset_row = ttk.Frame(scoring)
        preset_row.pack(fill="x", pady=(6, 4))
        ttk.Label(preset_row, text="LLM preset", width=18).pack(side="left")
        preset_box = ttk.Combobox(
            preset_row,
            textvariable=self.llm_preset,
            values=tuple(LLM_PRESETS.keys()),
            state="readonly",
            width=24,
        )
        preset_box.pack(side="right", fill="x", expand=True)
        preset_box.bind("<<ComboboxSelected>>", self._apply_llm_preset)
        self._wide_field(scoring, "LLM model", self.llm_model)
        self._wide_field(scoring, "Base URL", self.llm_base_url)
        self._wide_field(scoring, "API key env", self.llm_api_key_env)
        self._field(scoring, "Rate limit/min", self.llm_rate_limit)
        ttk.Label(scoring, text="API keys are read from environment variables or config.toml [llm].", style="Muted.TLabel").pack(anchor="w", pady=(6, 0))

        models = ttk.LabelFrame(left, text="Model Selection", padding=12)
        models.pack(fill="x", pady=(14, 0))
        self.use_lgbm = tk.BooleanVar(value=True)
        self.use_xgb = tk.BooleanVar(value=True)
        self.use_rf = tk.BooleanVar(value=True)
        ttk.Checkbutton(models, text="LightGBM slot", variable=self.use_lgbm).pack(anchor="w")
        ttk.Checkbutton(models, text="XGBoost slot", variable=self.use_xgb).pack(anchor="w")
        ttk.Checkbutton(models, text="RandomForest", variable=self.use_rf).pack(anchor="w")

        backtest = ttk.LabelFrame(left, text="Backtest Settings", padding=12)
        backtest.pack(fill="x", pady=(14, 0))
        self.top_pct = tk.StringVar(value="0.10")
        self.max_weight = tk.StringVar(value="0.05")
        self.min_weight = tk.StringVar(value="0.005")
        self.max_dd_limit = tk.StringVar(value="0.10")
        self.rebalance_days = tk.StringVar(value="5")
        self.transaction_bps = tk.StringVar(value="10")
        self.slippage_bps = tk.StringVar(value="5")
        self.initial_capital = tk.StringVar(value="1000000")
        self.execution_price_model = tk.StringVar(value="open")
        self._field(backtest, "Buy top fraction", self.top_pct)
        self._field(backtest, "Max stock weight", self.max_weight)
        self._field(backtest, "Min stock weight", self.min_weight)
        self._field(backtest, "Max DD limit", self.max_dd_limit)
        self._field(backtest, "Rebalance days", self.rebalance_days)
        self._field(backtest, "Transaction bps", self.transaction_bps)
        self._field(backtest, "Slippage bps", self.slippage_bps)
        self._field(backtest, "Initial capital", self.initial_capital)
        exec_row = ttk.Frame(backtest)
        exec_row.pack(fill="x", pady=4)
        ttk.Label(exec_row, text="Execution price", width=18).pack(side="left")
        ttk.Combobox(
            exec_row,
            textvariable=self.execution_price_model,
            values=("open", "vwap", "close"),
            width=8,
            state="readonly",
        ).pack(side="right")

        paths = ttk.LabelFrame(left, text="Project", padding=12)
        paths.pack(fill="x", pady=(14, 0))
        ttk.Label(paths, text=str(PROJECT_ROOT), wraplength=270, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(paths, text=str(PROJECT_ROOT / "data" / "ai_quant.duckdb"), wraplength=270).pack(anchor="w", pady=(8, 0))

        right_outer = ttk.Frame(body)
        right_outer.grid(row=0, column=1, sticky="nsew")
        right_outer.rowconfigure(0, weight=1)
        right_outer.columnconfigure(0, weight=1)
        self.right_canvas = tk.Canvas(right_outer, highlightthickness=0, bg=self.cget("background"))
        self.right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scroll = ttk.Scrollbar(right_outer, orient="vertical", command=self.right_canvas.yview)
        right_scroll.grid(row=0, column=1, sticky="ns")
        self.right_canvas.configure(yscrollcommand=right_scroll.set)
        right = ttk.Frame(self.right_canvas)
        self.right_window_id = self.right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", self._update_right_scrollregion)
        self.right_canvas.bind("<Configure>", self._resize_right_panel)
        self.right_canvas.bind("<Enter>", self._bind_right_mousewheel)
        self.right_canvas.bind("<Leave>", self._unbind_right_mousewheel)
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        status_bar = ttk.Frame(right)
        status_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.status_text = tk.StringVar(value="Ready")
        ttk.Label(status_bar, textvariable=self.status_text, style="Status.TLabel").pack(side="left")
        self.busy_progress = ttk.Progressbar(status_bar, mode="indeterminate", length=180)
        self.busy_progress.pack(side="right")

        progress_frame = ttk.LabelFrame(right, text="Pipeline Status", padding=12)
        progress_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        progress_frame.columnconfigure(0, weight=1)
        self.step_text = tk.StringVar(value="No task is running.")
        ttk.Label(progress_frame, textvariable=self.step_text).grid(row=0, column=0, sticky="w")
        self.step_progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=1)
        self.step_progress.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        universe_frame = ttk.LabelFrame(right, text="Universe Quality", padding=12)
        universe_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        universe_frame.columnconfigure(0, weight=1)
        self.universe_status_text = tk.StringVar(value="Not checked")
        ttk.Label(universe_frame, textvariable=self.universe_status_text, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        self.universe_table = ttk.Treeview(universe_frame, columns=("field", "value"), show="headings", height=6)
        self.universe_table.heading("field", text="Field")
        self.universe_table.heading("value", text="Value")
        self.universe_table.column("field", width=170, anchor="w")
        self.universe_table.column("value", width=220, anchor="e")
        self.universe_table.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self._register_inner_scroll_target(self.universe_table, "yview")

        viz_frame = ttk.LabelFrame(right, text="Out-of-sample Visualization", padding=12)
        viz_frame.grid(row=3, column=0, sticky="nsew")
        viz_frame.rowconfigure(0, weight=1, minsize=580)
        viz_frame.columnconfigure(0, weight=1)
        self.viz_holder = ttk.Frame(viz_frame)
        self.viz_holder.grid(row=0, column=0, sticky="nsew")
        self.viz_holder.rowconfigure(0, weight=1)
        self.viz_holder.columnconfigure(0, weight=1)
        table_area = ttk.Frame(viz_frame)
        table_area.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        table_area.columnconfigure(0, weight=1)
        table_area.columnconfigure(1, weight=1)

        metrics_box = ttk.LabelFrame(table_area, text="Backtest Metrics", padding=6)
        metrics_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.metrics_table = ttk.Treeview(metrics_box, columns=("metric", "value"), show="headings", height=4)
        self.metrics_table.heading("metric", text="Metric")
        self.metrics_table.heading("value", text="Value")
        self.metrics_table.column("metric", width=150, anchor="w")
        self.metrics_table.column("value", width=120, anchor="e")
        self.metrics_table.pack(fill="x")
        self._register_inner_scroll_target(self.metrics_table, "yview")

        holdings_box = ttk.LabelFrame(table_area, text="Latest Holdings", padding=6)
        holdings_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.holdings_table = ttk.Treeview(holdings_box, columns=("ticker", "weight"), show="headings", height=4)
        self.holdings_table.heading("ticker", text="Ticker")
        self.holdings_table.heading("weight", text="Target Weight")
        self.holdings_table.column("ticker", width=120, anchor="w")
        self.holdings_table.column("weight", width=120, anchor="e")
        self.holdings_table.pack(fill="x")
        self._register_inner_scroll_target(self.holdings_table, "yview")
        self.load_latest_report_image()

        footer = ttk.Frame(right_outer)
        footer.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(footer, text="Show Log", command=self.show_log_window).pack(side="left")
        self.force_stop_button = ttk.Button(footer, text="Force Stop All Tasks", command=self.force_stop_all_tasks)
        self.force_stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(footer, text="Export Holdings Excel", command=self.export_holdings_excel).pack(side="left", padx=(8, 0))
        ttk.Button(footer, text="Open Project Folder", command=self.open_project_folder).pack(side="right")

    def _field(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=18).pack(side="left")
        ttk.Entry(row, textvariable=variable, width=10).pack(side="right")

    def _wide_field(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=18).pack(side="left")
        ttk.Entry(row, textvariable=variable, width=24).pack(side="right", fill="x", expand=True)

    def _apply_llm_preset(self, _event: tk.Event | None = None) -> None:
        preset = LLM_PRESETS.get(self.llm_preset.get())
        if not preset:
            return
        self.scorer_provider.set(str(preset["provider"]))
        self.llm_model.set(str(preset["model"]))
        self.llm_base_url.set(str(preset["base_url"]))
        self.llm_api_key_env.set(str(preset["api_key_env"]))
        self.llm_rate_limit.set(str(preset["rate_limit_per_minute"]))

    def _load_ui_config(self) -> None:
        if not CONFIG.exists():
            return
        try:
            from ai_quant_research_system.config import load_config

            cfg = load_config(CONFIG)
        except Exception:
            return
        self.scorer_provider.set(cfg.llm.provider)
        self.llm_model.set(cfg.llm.model)
        self.llm_base_url.set(cfg.llm.base_url if cfg.llm.provider != "heuristic" else "local")
        self.llm_api_key_env.set(cfg.llm.api_key_env)
        self.llm_rate_limit.set(str(cfg.llm.rate_limit_per_minute))
        matched = "Custom OpenAI-compatible"
        for name, preset in LLM_PRESETS.items():
            if (
                preset["provider"] == cfg.llm.provider
                and preset["model"] == cfg.llm.model
                and (preset["base_url"] == cfg.llm.base_url or preset["base_url"] == "local")
            ):
                matched = name
                break
        self.llm_preset.set(matched)

    def _update_right_scrollregion(self, _event: tk.Event) -> None:
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def _resize_right_panel(self, event: tk.Event) -> None:
        self.right_canvas.itemconfigure(self.right_window_id, width=event.width)

    def _bind_right_mousewheel(self, _event: tk.Event) -> None:
        self.right_canvas.bind_all("<MouseWheel>", self._on_right_mousewheel)
        self.right_canvas.bind_all("<Button-4>", self._on_right_mousewheel)
        self.right_canvas.bind_all("<Button-5>", self._on_right_mousewheel)
        self.right_canvas.bind_all("<Button-3>", self._clear_inner_scroll_target)

    def _unbind_right_mousewheel(self, _event: tk.Event) -> None:
        self.right_canvas.unbind_all("<MouseWheel>")
        self.right_canvas.unbind_all("<Button-4>")
        self.right_canvas.unbind_all("<Button-5>")
        self.right_canvas.unbind_all("<Button-3>")

    def _on_right_mousewheel(self, event: tk.Event) -> None:
        if getattr(event, "num", None) == 4:
            delta = -3
        elif getattr(event, "num", None) == 5:
            delta = 3
        else:
            delta = -1 * int(getattr(event, "delta", 0) / 120)
        if self._scroll_selected_inner_target(delta):
            return "break"
        self.right_canvas.yview_scroll(delta, "units")
        return "break"

    def _register_inner_scroll_target(self, widget: tk.Widget, kind: str) -> None:
        widget.bind("<Button-1>", lambda _event, target=widget, target_kind=kind: self._select_inner_scroll_target(target, target_kind), add="+")

    def _select_inner_scroll_target(self, widget: tk.Widget, kind: str) -> None:
        self.right_scroll_target = widget
        self.right_scroll_target_kind = kind

    def _clear_inner_scroll_target(self, _event: tk.Event | None = None) -> str:
        self.right_scroll_target = None
        self.right_scroll_target_kind = None
        return "break"

    def _scroll_selected_inner_target(self, delta: int) -> bool:
        target = self.right_scroll_target
        if target is None or not target.winfo_exists():
            self.right_scroll_target = None
            self.right_scroll_target_kind = None
            return False
        if self.right_scroll_target_kind == "yview" and hasattr(target, "yview_scroll"):
            target.yview_scroll(delta, "units")  # type: ignore[attr-defined]
            return True
        if self.right_scroll_target_kind == "chart":
            return True
        return False

    def show_log_window(self) -> None:
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.lift()
            return
        self.log_window = tk.Toplevel(self)
        self.log_window.title("Live Log")
        self.log_window.geometry("980x620")
        frame = ttk.Frame(self.log_window, padding=8)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(frame, wrap="word", font=("Consolas", 10), bg="#101418", fg="#e6edf3", insertbackground="#e6edf3", relief="flat")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        ttk.Button(frame, text="Clear Log", command=self.clear_log).grid(row=1, column=0, sticky="w", pady=(8, 0))

    def clear_log(self) -> None:
        if self.log_text is not None and self.log_text.winfo_exists():
            self.log_text.delete("1.0", "end")

    def open_project_folder(self) -> None:
        os.startfile(PROJECT_ROOT)  # type: ignore[attr-defined]

    def log_line(self, text: str) -> None:
        self.log_queue.put(text)

    def _drain_log_queue(self) -> None:
        while True:
            try:
                text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if self.log_text is not None and self.log_text.winfo_exists():
                self.log_text.insert("end", text)
                self.log_text.see("end")
        self.after(100, self._drain_log_queue)

    def set_running(self, running: bool, status: str) -> None:
        self.running = running
        self.status_text.set(status)
        state = "disabled" if running else "normal"
        for child in self.winfo_children():
            self._set_button_state(child, state)
        if running:
            self.busy_progress.start(12)
        else:
            self.busy_progress.stop()
            self.load_latest_report_image()
            self.refresh_universe_status()

    def _set_button_state(self, widget: tk.Widget, state: str) -> None:
        if getattr(self, "force_stop_button", None) is widget:
            widget.configure(state="normal")
            return
        if isinstance(widget, ttk.Button):
            widget.configure(state=state)
        for child in widget.winfo_children():
            self._set_button_state(child, state)

    def ensure_config(self) -> None:
        if not CONFIG.exists() and (PROJECT_ROOT / "config.example.toml").exists():
            CONFIG.write_text((PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8"), encoding="utf-8")

    def command(self, *args: str) -> list[str]:
        python = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
        return [str(python), "-m", "ai_quant_research_system.cli", *args]

    def run_commands(self, title: str, commands: list[list[str]], install_deps: bool = False) -> None:
        if self.running:
            messagebox.showinfo("Task running", "A task is already running. Please wait for it to finish.")
            return
        self.worker = threading.Thread(target=self._run_commands_worker, args=(title, commands, install_deps), daemon=True)
        self.worker.start()

    def _run_commands_worker(self, title: str, commands: list[list[str]], install_deps: bool) -> None:
        self.after(0, self.set_running, True, f"Running: {title}")
        try:
            self.ensure_config()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
            self.log_line(f"\n== {title} ==\n")
            if install_deps:
                if not VENV_PYTHON.exists():
                    self.log_line("Creating virtual environment...\n")
                    self._run_process([sys.executable, "-m", "venv", str(PROJECT_ROOT / ".venv")], env)
                self.log_line("Installing dependencies...\n")
                self._run_process([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], env)
                self._run_process([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")], env)
            self._apply_ui_config()
            self.after(0, self._set_total_steps, len(commands))
            for idx, cmd in enumerate(commands, start=1):
                self.after(0, self._set_current_step, idx, self._command_label(cmd))
                self.log_line(f"\n$ {' '.join(cmd)}\n")
                self._run_process(cmd, env)
            self.log_line("\nDone.\n")
            self.after(0, self.set_running, False, "Done")
        except Exception as exc:
            self.log_line(f"\nERROR: {exc}\n")
            self.after(0, self.set_running, False, "Error")

    def _set_total_steps(self, total: int) -> None:
        self.total_steps = max(1, total)
        self.step_progress.configure(maximum=self.total_steps, value=0)
        self.step_text.set(f"0/{self.total_steps} steps complete")

    def _set_current_step(self, step: int, label: str) -> None:
        self.step_text.set(f"Step {step}/{self.total_steps}: {label}")
        self.step_progress.configure(value=step)

    def _command_label(self, cmd: list[str]) -> str:
        try:
            idx = cmd.index("ai_quant_research_system.cli")
            return cmd[idx + 1]
        except Exception:
            return Path(cmd[0]).name

    def _run_process(self, cmd: list[str], env: dict[str, str]) -> None:
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, startupinfo=startupinfo)
        self.active_process = process
        assert process.stdout is not None
        try:
            for line in process.stdout:
                self.log_line(line)
            code = process.wait()
            if code != 0:
                raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")
        finally:
            if self.active_process is process:
                self.active_process = None

    def force_stop_all_tasks(self) -> None:
        self.log_line("\nForce stop requested. Killing AI Quant task processes...\n")
        process = self.active_process
        if process is not None and process.poll() is None:
            try:
                process.kill()
                self.log_line(f"Killed active process PID {process.pid}.\n")
            except Exception as exc:
                self.log_line(f"Failed to kill active process: {exc}\n")
        if os.name == "nt":
            script = (
                "$current=$PID; "
                "$matches=Get-CimInstance Win32_Process | Where-Object { "
                "($_.CommandLine -like '*ai_quant_research_system.cli*' -or "
                "($_.CommandLine -like '*AI Quant Research System*' -and $_.Name -like 'python*')) "
                "-and $_.CommandLine -notlike '*run_ui.py*' "
                "-and $_.ProcessId -ne $current }; "
                "$matches | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; "
                "$matches | Select-Object ProcessId,CommandLine"
            )
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                    cwd=PROJECT_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=15,
                    startupinfo=self._hidden_startupinfo(),
                )
                output = result.stdout.strip()
                if output:
                    self.log_line(output + "\n")
            except Exception as exc:
                self.log_line(f"Failed to scan/kill project processes: {exc}\n")
        self.running = False
        self.busy_progress.stop()
        self.status_text.set("Stopped")
        self.step_text.set("Force stopped. You can start a new task.")
        self.step_progress.configure(value=0)
        for child in self.winfo_children():
            self._set_button_state(child, "normal")
        self.log_line("Force stop complete.\n")

    def _hidden_startupinfo(self) -> subprocess.STARTUPINFO | None:
        if os.name != "nt":
            return None
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return startupinfo

    def _apply_ui_config(self) -> None:
        if not CONFIG.exists():
            return
        text = CONFIG.read_text(encoding="utf-8-sig")
        text = self._replace_or_add(text, "provider", f'"{self.scorer_provider.get()}"', section="llm")
        text = self._replace_or_add(text, "model", f'"{self.llm_model.get().strip() or "local-heuristic-v1"}"', section="llm")
        base_url = self.llm_base_url.get().strip()
        if base_url and base_url != "local":
            text = self._replace_or_add(text, "base_url", f'"{base_url.rstrip("/")}"', section="llm")
        api_key_env = self.llm_api_key_env.get().strip()
        if api_key_env:
            text = self._replace_or_add(text, "api_key_env", f'"{api_key_env}"', section="llm")
        text = self._replace_or_add(text, "rate_limit_per_minute", self.llm_rate_limit.get().strip() or "60", section="llm")
        if self.scorer_provider.get() == "heuristic":
            text = self._replace_or_add(text, "rate_limit_per_minute", "3000", section="llm")
        CONFIG.write_text(text, encoding="utf-8")

    def _replace_or_add(self, text: str, key: str, value: str, section: str) -> str:
        lines = text.splitlines()
        in_section = False
        inserted = False
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped == f"[{section}]":
                in_section = True
                continue
            if in_section and stripped.startswith("[") and stripped.endswith("]"):
                lines.insert(idx, f"{key} = {value}")
                inserted = True
                break
            if in_section and stripped.startswith(f"{key} "):
                lines[idx] = f"{key} = {value}"
                inserted = True
                break
        if not inserted:
            if not in_section:
                lines.extend(["", f"[{section}]"])
            lines.append(f"{key} = {value}")
        return "\n".join(lines) + "\n"

    def _start_date(self) -> str:
        return (date.today() - timedelta(days=int(self.history_days.get().strip() or "730"))).isoformat()

    def _end_date(self) -> str:
        return (date.today() + timedelta(days=1)).isoformat()

    def _force_flag(self) -> list[str]:
        return ["--force"] if self.force_refresh.get() else []

    def _model_args(self) -> list[str]:
        models: list[str] = []
        if self.use_lgbm.get():
            models.append("lightgbm")
        if self.use_xgb.get():
            models.append("xgboost")
        if self.use_rf.get():
            models.append("rf")
        return ["--models", *(models or ["rf"])]

    def _factor_table_command(self) -> list[str]:
        args = ["build-factor-table", "--config", "config.toml"]
        if self.require_pit_universe.get():
            args.append("--require-pit-universe")
        return self.command(*args)

    def data_commands(self) -> list[list[str]]:
        return [
            self.command("init-db", "--config", "config.toml"),
            self.command("ingest-universe", "--config", "config.toml", "--csv", "examples\\sp500_constituents.csv"),
            self.command("ingest-prices", "--config", "config.toml", "--csv", "examples\\daily_prices.csv"),
            self.command("ingest-sp500-yfinance-prices", "--config", "config.toml", "--days", self.history_days.get().strip() or "730", *self._force_flag()),
            self.command("ingest-yfinance-prices", "--config", "config.toml", "--tickers", "SPY", "QQQ", "--start", self._start_date(), "--end", self._end_date(), *self._force_flag()),
            self.command("ingest-benchmark", "--config", "config.toml", "--csv", "examples\\benchmark_daily.csv"),
            self.command("ingest-news", "--config", "config.toml", "--jsonl", "examples\\raw_news.jsonl"),
            self.command("ingest-sp500-news-builtins", "--config", "config.toml", "--yahoo-limit-per-ticker", self.yahoo_news_limit.get().strip() or "3", "--sec-days", self.sec_days.get().strip() or "90", "--sec-limit-per-ticker", self.sec_limit.get().strip() or "2", *self._force_flag()),
        ]

    def modeling_commands(self) -> list[list[str]]:
        return [
            self.command("build-clean-views", "--config", "config.toml"),
            self.command("run-news-factor-pipeline", "--config", "config.toml", "--limit", self.score_limit.get().strip() or "5000"),
            self._factor_table_command(),
            self.command("train-models", "--config", "config.toml", *self._model_args()),
            self.command("optimize-portfolio", "--config", "config.toml", "--max-drawdown-limit", self.max_dd_limit.get().strip() or "0.10"),
            self.command("run-trust-audit", "--config", "config.toml"),
            self.command("count-rows", "--config", "config.toml"),
        ]

    def latest_report_path(self) -> Path | None:
        models_dir = PROJECT_ROOT / "models"
        if not models_dir.exists():
            return None
        candidates = sorted(
            [*models_dir.glob("model_*/backtest_report.png"), *models_dir.glob("model_*/out_of_sample_report.png")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def latest_backtest_results_path(self) -> Path | None:
        models_dir = PROJECT_ROOT / "models"
        if not models_dir.exists():
            return None
        candidates = sorted(
            models_dir.glob("model_*/backtest_results.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def latest_model_dir(self) -> Path | None:
        results_path = self.latest_backtest_results_path()
        if results_path is not None:
            return results_path.parent
        report_path = self.latest_report_path()
        if report_path is not None:
            return report_path.parent
        models_dir = PROJECT_ROOT / "models"
        if not models_dir.exists():
            return None
        candidates = sorted(
            [path for path in models_dir.glob("model_*") if path.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def export_holdings_excel(self) -> None:
        model_dir = self.latest_model_dir()
        if model_dir is None:
            messagebox.showinfo("Export holdings", "No model output folder found. Run the pipeline first.")
            return
        holdings_path = model_dir / "latest_holdings.csv"
        if not holdings_path.exists():
            messagebox.showinfo("Export holdings", "No latest_holdings.csv found. Run a backtest or optimization first.")
            return
        sheets: list[tuple[str, list[dict[str, object]]]] = [("Holdings", self._read_csv_rows(holdings_path))]
        current_positions_path = model_dir / "latest_current_positions.csv"
        if current_positions_path.exists():
            sheets.append(("Current Positions", self._read_csv_rows(current_positions_path)))
        optimized_path = model_dir / "optimized_weights.csv"
        if optimized_path.exists():
            sheets.append(("Optimized Weights", self._read_csv_rows(optimized_path)))
        output_path = model_dir / "latest_holdings_export.xlsx"
        try:
            self._write_xlsx(output_path, sheets)
        except Exception as exc:
            messagebox.showerror("Export holdings", f"Failed to export holdings Excel:\n{exc}")
            return
        messagebox.showinfo("Export holdings", f"Exported holdings Excel:\n{output_path}")
        try:
            os.startfile(output_path)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _read_csv_rows(self, path: Path) -> list[dict[str, object]]:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]

    def _write_xlsx(self, path: Path, sheets: list[tuple[str, list[dict[str, object]]]]) -> None:
        valid_sheets = [(self._safe_sheet_name(name), rows) for name, rows in sheets if rows]
        if not valid_sheets:
            raise ValueError("No holdings rows to export.")
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", self._xlsx_content_types(len(valid_sheets)))
            archive.writestr("_rels/.rels", self._xlsx_root_rels())
            archive.writestr("xl/workbook.xml", self._xlsx_workbook([name for name, _rows in valid_sheets]))
            archive.writestr("xl/_rels/workbook.xml.rels", self._xlsx_workbook_rels(len(valid_sheets)))
            for idx, (_name, rows) in enumerate(valid_sheets, start=1):
                archive.writestr(f"xl/worksheets/sheet{idx}.xml", self._xlsx_sheet(rows))

    def _safe_sheet_name(self, name: str) -> str:
        cleaned = "".join("_" if ch in "[]:*?/\\'" else ch for ch in name).strip()
        return (cleaned or "Sheet")[:31]

    def _xlsx_column_name(self, index: int) -> str:
        name = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            name = chr(65 + remainder) + name
        return name

    def _xlsx_sheet(self, rows: list[dict[str, object]]) -> str:
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        xml_rows = [self._xlsx_row(1, columns)]
        for row_idx, row in enumerate(rows, start=2):
            xml_rows.append(self._xlsx_row(row_idx, [row.get(column, "") for column in columns]))
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData>"
            + "".join(xml_rows)
            + "</sheetData></worksheet>"
        )

    def _xlsx_row(self, row_idx: int, values: list[object]) -> str:
        cells = []
        for col_idx, value in enumerate(values, start=1):
            ref = f"{self._xlsx_column_name(col_idx)}{row_idx}"
            text = escape("" if value is None else str(value))
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        return f'<row r="{row_idx}">' + "".join(cells) + "</row>"

    def _xlsx_content_types(self, sheet_count: int) -> str:
        sheets = "".join(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for idx in range(1, sheet_count + 1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            f"{sheets}</Types>"
        )

    def _xlsx_root_rels(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>"
        )

    def _xlsx_workbook(self, sheet_names: list[str]) -> str:
        sheets = "".join(
            f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
            for idx, name in enumerate(sheet_names, start=1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{sheets}</sheets></workbook>"
        )

    def _xlsx_workbook_rels(self, sheet_count: int) -> str:
        relationships = "".join(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
            for idx in range(1, sheet_count + 1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{relationships}</Relationships>"
        )

    def load_latest_report_image(self) -> None:
        results_path = self.latest_backtest_results_path()
        if results_path is not None:
            self._render_interactive_backtest_chart(results_path)
            self.load_latest_selection_summary(results_path.parent)
            return

        image_path = self.latest_report_path()
        if image_path is None:
            self._render_visual_placeholder()
            return
        self.report_image_path = image_path
        self._render_report_image(image_path)
        self.load_latest_selection_summary(image_path.parent)

    def _clear_visualization(self) -> None:
        if self.mpl_canvas is not None:
            try:
                self.mpl_canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.mpl_canvas = None
        if self.mpl_toolbar is not None:
            try:
                self.mpl_toolbar.destroy()
            except Exception:
                pass
            self.mpl_toolbar = None
        for child in self.viz_holder.winfo_children():
            child.destroy()
        self.hover_annotation = None
        self.hover_vlines = []

    def _render_visual_placeholder(self) -> None:
        self._clear_visualization()
        placeholder = tk.Canvas(self.viz_holder, highlightthickness=0, bg="#f4f6f8")
        placeholder.grid(row=0, column=0, sticky="nsew")
        placeholder.create_text(
            420,
            180,
            text="Run the modeling pipeline or backtest to generate an interactive out-of-sample chart.",
            anchor="center",
            fill="#5f6b7a",
            width=620,
        )

    def _read_backtest_results(self, path: Path) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    rows.append(
                        {
                            "date": datetime.fromisoformat(str(row["date"])[:10]),
                            "portfolio_value": float(row["portfolio_value"]),
                            "benchmark_value": float(row["benchmark_value"]),
                            "drawdown": float(row.get("drawdown") or 0.0),
                            "turnover": float(row.get("turnover") or 0.0),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        return rows

    def _render_interactive_backtest_chart(self, path: Path) -> None:
        rows = self._read_backtest_results(path)
        if len(rows) < 2:
            self._render_report_image(self.latest_report_path() or path)
            return
        self._clear_visualization()
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure
            import matplotlib.dates as mdates
        except Exception:
            self._render_report_image(self.latest_report_path() or path)
            return

        dates = [row["date"] for row in rows]
        strategy = [float(row["portfolio_value"]) for row in rows]
        benchmark = [float(row["benchmark_value"]) for row in rows]
        drawdown = [float(row["drawdown"]) for row in rows]
        turnover = [float(row["turnover"]) for row in rows]
        x_values = mdates.date2num(dates)

        figure = Figure(figsize=(8.5, 7.6), dpi=100, facecolor="#ffffff")
        price_ax = figure.add_subplot(211)
        drawdown_ax = figure.add_subplot(212, sharex=price_ax)
        figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.13, hspace=0.12)

        price_ax.plot(dates, strategy, label="Strategy", color="#1f77b4", linewidth=2.0)
        price_ax.plot(dates, benchmark, label="S&P 500 / SPY", color="#2ca02c", linewidth=1.7)
        price_ax.set_title("Interactive Out-of-Sample Equity Curve")
        price_ax.set_ylabel("Portfolio Value")
        price_ax.grid(True, alpha=0.25)
        price_ax.legend(loc="upper left")

        drawdown_ax.fill_between(dates, drawdown, 0, color="#d55e00", alpha=0.32, label="Drawdown")
        drawdown_ax.plot(dates, turnover, color="#7f7f7f", alpha=0.55, linewidth=1.0, label="Turnover")
        drawdown_ax.set_ylabel("DD / Turnover")
        drawdown_ax.grid(True, alpha=0.25)
        drawdown_ax.legend(loc="lower left")
        drawdown_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        figure.autofmt_xdate(rotation=0, ha="center")

        self.hover_vlines = [
            price_ax.axvline(dates[0], color="#333333", alpha=0.2, linewidth=1, visible=False),
            drawdown_ax.axvline(dates[0], color="#333333", alpha=0.2, linewidth=1, visible=False),
        ]
        def make_hover_annotation(axis: object) -> object:
            annotation = axis.annotate(
                "",
                xy=(0, 0),
                xytext=(8, 10),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#ffffff", "ec": "#667085", "alpha": 0.96},
                arrowprops={"arrowstyle": "->", "color": "#667085"},
            )
            annotation.set_visible(False)
            return annotation

        price_annotation = make_hover_annotation(price_ax)
        drawdown_annotation = make_hover_annotation(drawdown_ax)
        hover_annotations = {
            price_ax: price_annotation,
            drawdown_ax: drawdown_annotation,
        }
        self.hover_annotation = price_annotation

        canvas = FigureCanvasTkAgg(figure, master=self.viz_holder)
        canvas.draw()
        chart_widget = canvas.get_tk_widget()
        chart_widget.configure(width=760, height=620)
        chart_widget.grid(row=0, column=0, sticky="nsew")
        self._register_inner_scroll_target(chart_widget, "chart")
        toolbar_frame = ttk.Frame(self.viz_holder)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left")
        self.mpl_canvas = canvas
        self.mpl_toolbar = toolbar_frame

        def place_hover_box(annotation: object, axis: object, point_x_num: float, point_y: float) -> None:
            x0, y0, width, height = axis.bbox.bounds
            point_display_x, point_display_y = axis.transData.transform((point_x_num, point_y))
            on_right_half = point_display_x > x0 + width * 0.52
            on_upper_half = point_display_y > y0 + height * 0.55
            x_offset = -8 if on_right_half else 8
            y_offset = -34 if on_upper_half else 10
            annotation.set_position((x_offset, y_offset))
            annotation.set_ha("right" if on_right_half else "left")
            annotation.set_va("top" if on_upper_half else "bottom")

        def on_motion(event: object) -> None:
            active_axes = getattr(event, "inaxes", None)
            if getattr(event, "xdata", None) is None or active_axes not in hover_annotations:
                for annotation in hover_annotations.values():
                    annotation.set_visible(False)
                for line in self.hover_vlines:
                    line.set_visible(False)
                canvas.draw_idle()
                return
            xdata = float(getattr(event, "xdata"))
            idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - xdata))
            current_x_num = float(x_values[idx])
            current_date = dates[idx]
            current_strategy = strategy[idx]
            current_benchmark = benchmark[idx]
            current_drawdown = drawdown[idx]
            current_turnover = turnover[idx]
            for line in self.hover_vlines:
                line.set_xdata([current_date, current_date])
                line.set_visible(True)
            for axis, annotation in hover_annotations.items():
                annotation.set_visible(False)
            active_annotation = hover_annotations[active_axes]
            active_annotation.xy = (
                current_date,
                current_drawdown if active_axes is drawdown_ax else current_strategy,
            )
            active_point_y = current_drawdown if active_axes is drawdown_ax else current_strategy
            place_hover_box(active_annotation, active_axes, current_x_num, active_point_y)
            active_annotation.set_text(
                f"{current_date:%Y-%m-%d}\n"
                f"Strategy: {current_strategy:,.0f}\n"
                f"S&P 500: {current_benchmark:,.0f}\n"
                f"Drawdown: {current_drawdown:.2%}\n"
                f"Turnover: {current_turnover:.2%}"
            )
            active_annotation.set_visible(True)
            self.hover_annotation = active_annotation
            canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_motion)

    def _render_report_image(self, path: Path) -> None:
        self._clear_visualization()
        try:
            from PIL import Image, ImageTk

            image = Image.open(path)
            fallback_canvas = tk.Canvas(self.viz_holder, highlightthickness=0, bg="#f4f6f8")
            fallback_canvas.grid(row=0, column=0, sticky="nsew")
            canvas_width = max(520, self.viz_holder.winfo_width() - 24)
            max_height = 620
            image.thumbnail((canvas_width, max_height))
            self.report_image = ImageTk.PhotoImage(image)
            x = max(12, (self.viz_holder.winfo_width() - image.width) // 2)
            self.report_canvas_image_id = fallback_canvas.create_image(x, 12, image=self.report_image, anchor="nw")
        except Exception:
            fallback_canvas = tk.Canvas(self.viz_holder, highlightthickness=0, bg="#f4f6f8")
            fallback_canvas.grid(row=0, column=0, sticky="nsew")
            fallback_canvas.create_text(20, 20, text=f"Report image generated:\n{path}", anchor="nw", fill="#5f6b7a", width=760)

    def load_latest_selection_summary(self, model_dir: Path) -> None:
        holdings_path = model_dir / "latest_holdings.json"
        metrics_path = model_dir / "backtest_metrics.json"
        if holdings_path.exists():
            try:
                holdings = json.loads(holdings_path.read_text(encoding="utf-8"))
                metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
                self._set_table_rows(
                    self.metrics_table,
                    [
                        ("Strategy Return", self._fmt_pct(metrics.get("total_return"))),
                        ("S&P 500 Return", self._fmt_pct(metrics.get("benchmark_return"))),
                        ("Excess Return", self._fmt_pct(metrics.get("excess_return"))),
                        ("CAGR", self._fmt_pct(metrics.get("cagr"))),
                        ("Sharpe", self._fmt_num(metrics.get("sharpe"))),
                        ("Max Drawdown", self._fmt_pct(metrics.get("max_drawdown"))),
                        ("Avg Turnover", self._fmt_pct(metrics.get("avg_turnover"))),
                    ],
                )
                holding_rows = [
                    (item.get("ticker", ""), self._fmt_pct(item.get("target_weight")))
                    for item in holdings.get("holdings", [])[:12]
                ]
                if holding_rows:
                    holding_rows.insert(0, ("Rebalance Date", holdings.get("latest_rebalance_date", "")))
                    holding_rows.insert(1, ("Invested", self._fmt_pct(holdings.get("invested_weight"))))
                    holding_rows.insert(2, ("Cash", self._fmt_pct(holdings.get("cash_weight"))))
                self._set_table_rows(self.holdings_table, holding_rows)
                return
            except Exception:
                self._set_table_rows(self.metrics_table, [("Backtest guide", str(holdings_path))])
                self._set_table_rows(self.holdings_table, [])
                return

        path = model_dir / "selection_summary.json"
        if not path.exists():
            return
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            self._set_table_rows(
                self.metrics_table,
                [
                    ("Buy Ratio", self._fmt_pct(summary.get("buy_ratio"))),
                    ("Holdings", summary.get("selected_count", "")),
                    ("Equal Weight", self._fmt_pct(summary.get("equal_weight_per_holding"))),
                    ("Latest Date", summary.get("latest_date", "")),
                ],
            )
            self._set_table_rows(self.holdings_table, [(ticker, "") for ticker in summary.get("top_tickers", [])[:12]])
        except Exception:
            self._set_table_rows(self.metrics_table, [("Selection guide", str(path))])
            self._set_table_rows(self.holdings_table, [])

    def _set_table_rows(self, table: ttk.Treeview, rows: list[tuple[object, object]]) -> None:
        for item in table.get_children():
            table.delete(item)
        for left, right in rows:
            table.insert("", "end", values=(left, right))

    def refresh_universe_status(self) -> None:
        try:
            if not CONFIG.exists():
                self.universe_status_text.set("Config missing")
                self._set_table_rows(self.universe_table, [("Action", "Run Initialize DB")])
                return
            from ai_quant_research_system.config import load_config
            from ai_quant_research_system.factor_engine import universe_coverage

            cfg = load_config(CONFIG)
            db_path = Path(cfg.database.path)
            if not db_path.is_absolute():
                db_path = PROJECT_ROOT / db_path
            if not db_path.exists():
                self.universe_status_text.set("Database missing")
                self._set_table_rows(self.universe_table, [("Action", "Run Fetch / Update Data")])
                return
            coverage = universe_coverage(db_path)
            status = "PIT ready" if coverage.has_serious_pit_history else "Fallback: current/partial universe"
            self.universe_status_text.set(status)
            self._set_table_rows(
                self.universe_table,
                [
                    ("Rows / tickers", f"{coverage.total_rows} / {coverage.distinct_tickers}"),
                    ("Snapshots", coverage.snapshot_dates),
                    ("Entry / exit rows", f"{coverage.rows_with_entry_date} / {coverage.rows_with_exit_date}"),
                    ("Active start / end", f"{coverage.active_at_price_start} / {coverage.active_at_price_end}"),
                    ("Dated ratio", self._fmt_pct(coverage.dated_ratio)),
                    ("Rejected because", "; ".join(coverage.rejection_reasons[:2])),
                ],
            )
        except Exception as exc:
            self.universe_status_text.set("Universe check failed")
            self._set_table_rows(self.universe_table, [("Error", str(exc)[:120])])

    def _fmt_pct(self, value: object) -> str:
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return ""

    def _fmt_num(self, value: object) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return ""

    def run_data_pipeline(self) -> None:
        self.run_commands("Fetch / Update Data", self.data_commands(), install_deps=True)

    def run_modeling_pipeline(self) -> None:
        self.run_commands("Run Modeling Pipeline", self.modeling_commands())

    def run_full_pipeline(self) -> None:
        self.run_commands("Full Pipeline", [*self.data_commands(), *self.modeling_commands()], install_deps=True)

    def run_init_db(self) -> None:
        self.run_commands("Initialize DB", [self.command("init-db", "--config", "config.toml")])

    def run_prices(self) -> None:
        self.run_commands("Fetch Prices", self.data_commands()[3:5])

    def run_news(self) -> None:
        self.run_commands("Fetch S&P 500 News", [self.data_commands()[-1]])

    def run_news_factors(self) -> None:
        self.run_commands("Build News Factors", self.modeling_commands()[:2])

    def run_factor_table(self) -> None:
        self.run_commands("Build Factor Table", [self._factor_table_command()])

    def run_models(self) -> None:
        self.run_commands("Train Models", [self.command("train-models", "--config", "config.toml", *self._model_args())])

    def run_walk_forward_models(self) -> None:
        self.run_commands("Train Walk-Forward", [self.command("train-walk-forward-models", "--config", "config.toml", "--models", "rf")])

    def _backtest_args(self) -> list[str]:
        return [
            "--top-pct", self.top_pct.get().strip() or "0.10",
            "--max-weight", self.max_weight.get().strip() or "0.05",
            "--min-weight", self.min_weight.get().strip() or "0.005",
            "--rebalance-days", self.rebalance_days.get().strip() or "5",
            "--transaction-bps", self.transaction_bps.get().strip() or "10",
            "--slippage-bps", self.slippage_bps.get().strip() or "5",
            "--initial-capital", self.initial_capital.get().strip() or "1000000",
            "--execution-price-model", self.execution_price_model.get().strip() or "open",
        ]

    def run_backtest(self) -> None:
        self.run_commands("Run Backtest", [self.command("run-backtest", "--config", "config.toml", *self._backtest_args())])

    def run_optimize_portfolio(self) -> None:
        self.run_commands(
            "Optimize Portfolio",
            [
                self.command(
                    "optimize-portfolio",
                    "--config", "config.toml",
                    "--max-drawdown-limit", self.max_dd_limit.get().strip() or "0.10",
                    "--execution-price-model", self.execution_price_model.get().strip() or "open",
                )
            ],
        )

    def run_trust_audit(self) -> None:
        self.run_commands("Run Trust Audit", [self.command("run-trust-audit", "--config", "config.toml")])

    def import_historical_universe(self) -> None:
        path = filedialog.askopenfilename(
            title="Import historical S&P 500 universe CSV",
            initialdir=str(PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.run_commands(
            "Import Historical Universe",
            [
                self.command("ingest-universe", "--config", "config.toml", "--csv", path),
                self.command("check-universe", "--config", "config.toml"),
            ],
        )

    def run_counts(self) -> None:
        self.run_commands("Refresh Row Counts", [self.command("count-rows", "--config", "config.toml")])


if __name__ == "__main__":
    configure_launch_environment()
    relaunch_with_project_venv()
    app = PipelineUI()
    app.mainloop()
