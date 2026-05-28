"""
Project Challenger — Interactive Controller (TUI)
Primary entry point for normal use.

Copyright (c) 2026 Rev. J. Money.
Non-commercial learning/research use only. See LICENSE and NOTICE.

Run:  python controller.py
      python controller.py --no-autostart

Tabs:
  Live          — paper-trading metrics table, model on/off toggles, worker dots
  Backtest      — configurable replay settings + results table
  Models        — top-3 archive per engine (P&L ranked), CV training scores
  Activity      — real-time per-component status + rolling event log
  Exchange&Keys — exchange selector, API key import, Coinbase account balance
  Settings      — data windows, cache config, live trading arm/disarm
"""
import argparse
import multiprocessing
import os
import sys
from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button, DataTable, Footer, Header, Input,
    Label, RichLog, Static, Switch, TabbedContent, TabPane,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from startup import check_install, tui_auth
from config import (
    CONFIG, is_demo_mode, get_active_symbol,
    save_user_settings, save_api_keys, clear_api_keys,
)
from bot_manager import BotManager, models_are_trained
from backtester import BacktestConfig
from activity import get_tracker
from database import (
    get_model_stats, get_latest_portfolio, get_training_summary,
    get_backtest_results, get_live_trade_stats, get_latest_prices,
)
from bulk_downloader import BulkDownloader

_INITIAL_CAP = CONFIG["PAPER_TRADING"]["INITIAL_CAPITAL"]

# Live-bot worker models (processes actually run for paper/live trading)
_MODEL_KEYS  = ["SKLEARN_LINEAR", "XGBOOST_TREE", "PYTORCH_LSTM"]

# All 12 trainable/archivable engines (shown in Models tab)
_ALL_MODEL_KEYS = [
    "SKLEARN_LINEAR", "XGBOOST_TREE", "PYTORCH_LSTM",
    "LGBM_TREE", "CATBOOST_TREE", "RF_TREE", "ET_TREE",
    "ELASTIC_LINEAR", "SVR_KERNEL", "MLP_NN", "ARIMA_STATS", "PROPHET_FB",
]
_MODEL_LABEL = {
    "SKLEARN_LINEAR": "Ridge (sklearn)",
    "XGBOOST_TREE":   "XGBoost",
    "PYTORCH_LSTM":   "PyTorch LSTM",
    "LGBM_TREE":      "LightGBM",
    "CATBOOST_TREE":  "CatBoost",
    "RF_TREE":        "Random Forest",
    "ET_TREE":        "Extra Trees",
    "ELASTIC_LINEAR": "Elastic Net",
    "SVR_KERNEL":     "SVR",
    "MLP_NN":         "MLP Neural Net",
    "ARIMA_STATS":    "ARIMA",
    "PROPHET_FB":     "Prophet (FB)",
}
_EXCHANGES = ["COINBASE", "BINANCE", "KRAKEN"]


# ─────────────────────────────────────────────────── helpers ──────────────────

def _fmt_pnl(val: float) -> str:
    c = "green" if val >= 0 else "red"
    return f"[{c}]{val:+.2f}[/{c}]"

def _fmt_pf(gp, gl) -> str:
    gp, gl = gp or 0, gl or 0
    if gl == 0:
        return "[green]inf[/green]" if gp > 0 else "—"
    pf = gp / gl
    c  = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
    return f"[{c}]{pf:.2f}[/{c}]"

def _fmt_wr(wins, losses) -> str:
    w, l = wins or 0, losses or 0
    t = w + l
    if t == 0:
        return "—"
    pct = w / t * 100
    c   = "green" if pct >= 55 else ("yellow" if pct >= 45 else "red")
    return f"[{c}]{pct:.0f}%[/{c}]"


# ─────────────────────────────────────────────────── key-file parser ──────────

def _parse_key_content(filename: str, content: str) -> "tuple[str,str] | None":
    """
    Extract (api_key, api_secret) from file content.
    `filename` is used to choose JSON vs text parsing.
    Returns None if nothing usable is found.
    """
    import json as _json

    fname = filename.lower()
    if fname.endswith(".json"):
        try:
            data   = _json.loads(content)
            key    = (data.get("api_key")    or data.get("key")        or
                      data.get("name")       or "").strip()
            secret = (data.get("api_secret") or data.get("secret")     or
                      data.get("privateKey") or "").strip()
            # Coinbase CDP: name is "organizations/.../apiKeys/xyz" — take last segment
            if "/" in key:
                key = key.rsplit("/", 1)[-1]
            if key and secret:
                return key, secret
        except Exception:
            pass
        return None

    # Plain-text: try KEY=VALUE pairs first
    lines = [
        ln.strip() for ln in content.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    kv: dict[str, str] = {}
    for ln in lines:
        if "=" in ln:
            k, _, v = ln.partition("=")
            kv[k.strip().upper()] = v.strip()

    key    = kv.get("API_KEY") or kv.get("KEY") or kv.get("APIKEY") or ""
    secret = (kv.get("API_SECRET") or kv.get("SECRET") or
              kv.get("APISECRET") or "")
    if key and secret:
        return key, secret

    # Two-bare-line format: line 1 = key, line 2 = secret
    bare = [ln for ln in lines if "=" not in ln]
    if len(bare) >= 2:
        return bare[0], bare[1]

    return None


def _parse_key_file(path: str) -> "tuple[str,str] | None":
    """
    Parse a Coinbase API key file and return (api_key, api_secret) or None.

    Supported formats
    -----------------
    JSON   {"api_key": "KEY", "api_secret": "SECRET"}
           {"key": "KEY", "secret": "SECRET"}
           {"name": "KEY", "privateKey": "SECRET"}   (Coinbase CDP export)
    TXT    Line 1 = key, Line 2 = secret
           OR lines containing  KEY=value / API_SECRET=value  (any case)
    ZIP    Archive containing exactly one .json or .txt file in the above formats
    """
    import zipfile

    clean = path.strip().strip('"').strip("'")
    if not os.path.exists(clean):
        return None

    if clean.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(clean) as z:
                for name in z.namelist():
                    if name.lower().endswith((".json", ".txt")):
                        data = z.read(name).decode("utf-8", errors="replace")
                        res  = _parse_key_content(name, data)
                        if res:
                            return res
        except Exception:
            pass
        return None

    try:
        with open(clean, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return _parse_key_content(clean, content)
    except Exception:
        return None


# ─────────────────────────────────────────────────── modals ───────────────────

class ConfirmModal(ModalScreen):
    CSS = """
    ConfirmModal { align: center middle; }
    #dialog { background: $surface; border: tall $primary; padding: 2 4; width: 54; height: auto; }
    #msg    { margin-bottom: 2; text-align: center; }
    #btns   { align: center middle; height: 3; }
    #btns Button { margin: 0 2; }
    """
    def __init__(self, msg: str, on_confirm):
        super().__init__()
        self._msg = msg
        self._ok  = on_confirm
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self._msg, id="msg")
            with Horizontal(id="btns"):
                yield Button("Confirm", id="yes", variant="error")
                yield Button("Cancel",  id="no",  variant="primary")
    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "yes":
            self._ok()
        self.dismiss()


class ReportModal(ModalScreen):
    CSS = """
    ReportModal { align: center middle; }
    #rbox { background: $surface; border: tall $primary; padding: 1 2; width: 90%; height: 80%; }
    #rlog { height: 1fr; }
    #close-row { height: 3; align: center middle; }
    """
    def __init__(self, content: str):
        super().__init__()
        self._content = content
    def compose(self) -> ComposeResult:
        with Vertical(id="rbox"):
            yield RichLog(id="rlog", markup=True, highlight=False)
            with Horizontal(id="close-row"):
                yield Button("Close", id="close-btn", variant="primary")
    def on_mount(self) -> None:
        log = self.query_one("#rlog", RichLog)
        for line in self._content.splitlines():
            log.write(line)
    def on_button_pressed(self, ev: Button.Pressed) -> None:
        self.dismiss()


# ─────────────────────────────────────────────────── main app ─────────────────

class ChallengerApp(App):
    TITLE = "Project Challenger"

    CSS = """
    Screen { background: $surface-darken-1; }

    /* ── global control bar ── */
    #ctrl-bar {
        height: 3; dock: top; margin-top: 3;
        background: $panel; padding: 0 1; align: left middle;
    }
    #ctrl-bar Button { min-width: 13; margin: 0 1; }
    #status { dock: right; width: auto; padding: 0 2; content-align: right middle; }

    /* ── tabs ── */
    TabbedContent { height: 1fr; }
    TabPane { padding: 0 1; height: 1fr; }

    /* ── live tab ── */
    #live-body { height: 1fr; }
    #metrics-box { width: 3fr; border: solid $primary-darken-2; padding: 0 1; }
    #metrics-title { color: $primary; text-style: bold; padding: 0 0 1 0; }
    DataTable { height: 1fr; }
    #sidebar { width: 20; border: solid $primary-darken-2; padding: 1; }
    .sidebar-head { color: $accent; text-style: bold; margin-bottom: 1; }
    .model-row { height: 3; align: left middle; }
    .model-row Label { width: 1fr; }
    .worker-dot { margin: 0 0 1 0; }

    /* ── backtest tab ── */
    #bt-body { height: 1fr; }
    #bt-config { width: 26; border: solid $primary-darken-2; padding: 1; }
    #bt-config Label { margin-bottom: 0; color: $text-muted; }
    #bt-config Input { margin-bottom: 1; width: 100%; }
    #btn-run-bt { width: 100%; margin-top: 1; }
    #bt-results-box { width: 1fr; border: solid $primary-darken-2; padding: 0 1; }
    #bt-results-title { color: $primary; text-style: bold; padding: 0 0 1 0; }
    #bt-status { color: $text-muted; margin: 1 0; }
    #bt-table { height: 1fr; }

    /* ── exchange & keys tab ── */
    #exchange-tab-body { height: 1fr; }
    #exch-selector-box { border: solid $primary-darken-2; padding: 1; margin-bottom: 1; }
    .exch-btn { min-width: 12; margin: 0 1; }
    .exch-btn.-active { background: $primary; color: $text; }
    #exch-status { color: $text-muted; margin-top: 1; }
    #api-keys-box { border: solid $primary-darken-2; padding: 1; margin-bottom: 1; }
    #api-keys-head { color: $accent; text-style: bold; margin-bottom: 1; }
    #api-key-status { margin: 1 0; }
    #key-file-path { width: 100%; margin-bottom: 1; }
    #key-btns { height: 3; margin-top: 0; }
    #key-btns Button { min-width: 10; margin: 0 1 0 0; }
    #key-hint { color: $text-muted; }
    #balance-box { border: solid $primary-darken-2; padding: 1; }
    #balance-info { color: $text; margin: 1 0; }
    #btn-refresh-balance { width: 100%; margin-top: 1; }
    #balance-last-update { color: $text-muted; margin-top: 1; }

    /* ── market data / ticker ── */
    #market-data-box { border: solid $primary-darken-2; padding: 1; margin-bottom: 1; }
    #market-data-head { color: $accent; text-style: bold; margin-bottom: 1; }
    #ticker-table { height: 16; }
    #bulk-download-box { border: solid $primary-darken-2; padding: 1; margin-bottom: 1; }
    #bulk-download-head { color: $accent; text-style: bold; margin-bottom: 1; }
    #bulk-status { margin: 1 0; color: $text; }
    #btn-bulk-start  { min-width: 22; margin: 0 1 0 0; }
    #btn-bulk-stop   { min-width: 18; margin: 0 1 0 0; }
    #btn-bulk-redownload { min-width: 22; margin: 0 1 0 0; }

    /* ── settings tab ── */
    #settings-body { height: 1fr; }
    #settings-left { width: 3fr; margin-right: 1; }
    #settings-right { width: 2fr; }
    .settings-head { color: $accent; text-style: bold; margin-bottom: 1; }
    #windows-box { border: solid $primary-darken-2; padding: 1; margin-bottom: 1; }
    #windows-box Label { color: $text-muted; margin-bottom: 0; }
    #windows-box Input { margin-bottom: 1; width: 100%; }
    #btn-apply-settings { width: 100%; margin-top: 1; }
    #settings-saved-msg { color: $success; margin-top: 1; }
    #cache-box { border: solid $primary-darken-2; padding: 1; }
    #cache-info { color: $text-muted; margin: 1 0; }
    #live-trading-box { border: solid $warning; padding: 1; margin-bottom: 1; }
    #live-trading-head { color: $warning; text-style: bold; margin-bottom: 1; }
    #arm-btn { margin-top: 1; width: 100%; }
    #arm-btn.-armed { background: $error; }
    #live-warning { color: $warning; margin-top: 1; }
    #armed-rival-notice { color: $warning; }
    #armed-rival-notice.-blink-on  { color: $warning; text-style: bold; }
    #armed-rival-notice.-blink-off { color: $warning; text-style: dim; }
    #paper-stats-box { border: solid $primary-darken-2; padding: 1; }
    #paper-stats-head { color: $accent; text-style: bold; margin-bottom: 1; }

    /* ── activity tab ── */
    #activity-body { height: 1fr; }
    #activity-table-box { width: 2fr; border: solid $primary-darken-2; padding: 0 1; }
    #activity-title { color: $primary; text-style: bold; padding: 0 0 1 0; }
    #activity-log-box { width: 3fr; border: solid $primary-darken-2; padding: 0 1; }
    #activity-log-title { color: $primary; text-style: bold; }
    #activity-log { height: 1fr; }

    /* ── models tab ── */
    #models-outer { height: 1fr; }
    #models-armed-summary {
        height: auto; padding: 1 2;
        background: $panel; border: solid $primary-darken-2; margin-bottom: 1;
    }
    #models-cv-section {
        border: solid $primary-darken-2; padding: 1; margin-bottom: 1;
        height: auto;
    }
    #cv-summary { color: $text-muted; }
    #models-scroll { height: 1fr; }
    .engine-box-stacked {
        border: solid $primary-darken-2; padding: 0 1 1 1;
        margin-bottom: 1; height: auto;
    }
    .engine-title { color: $primary; text-style: bold; padding: 0 0 0 0; height: 1; }
    .engine-archive-table { height: 6; }
    .arch-activate-row { height: 3; margin-top: 0; }
    .arch-activate-row Button { min-width: 14; margin: 0 1 0 0; }
    #btn-refresh-archive { width: 22; margin: 1 0; }

    /* ── log box ── */
    #log-box { height: 9; border: solid $primary-darken-2; padding: 0 1; }
    #log-title { color: $primary; text-style: bold; }
    RichLog { height: 1fr; }

    /* ── sub-tabs (nested TabbedContent in Exchange & Keys, Settings) ── */
    #exch-tabs { height: 1fr; }
    #settings-tabs { height: 1fr; }

    /* ── coins tab ── */
    #coins-tier-box { border: solid $primary-darken-2; padding: 1; margin-bottom: 1; }
    #coins-tier-head { color: $accent; text-style: bold; margin-bottom: 1; }
    #coins-tier-btns { height: 3; margin-bottom: 1; }
    .tier-btn { min-width: 12; margin: 0 1 0 0; }
    .tier-btn.-active { background: $primary; color: $text; }
    #coins-tier-status { color: $text-muted; }
    #coins-list-box { border: solid $primary-darken-2; padding: 1; }
    #coins-list-head { color: $accent; text-style: bold; margin-bottom: 1; }
    #coins-list-info { margin: 1 0; }
    """

    BINDINGS = [
        Binding("s", "do_start",    "Start",    show=True),
        Binding("x", "do_stop",     "Stop",     show=True),
        Binding("t", "do_retrain",  "Retrain",  show=True),
        Binding("b", "do_backtest", "Backtest", show=True),
        Binding("c", "do_compare",  "Compare",  show=True),
        Binding("a", "show_activity","Activity", show=True),
        Binding("q", "quit",        "Quit",     show=True),
    ]

    _running    = reactive(False)
    _retraining = reactive(False)
    _bt_running = reactive(False)
    _live_armed = reactive(False)

    # ── compose ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="ctrl-bar"):
            yield Button("▶  Start",      id="btn-start",   variant="success")
            yield Button("■  Stop",       id="btn-stop",    variant="error")
            yield Button("⟳  Retrain",   id="btn-retrain", variant="warning")
            yield Button("↺  Fresh Data", id="btn-fresh",   variant="default")
            yield Button("✕  Reset DB",   id="btn-reset",   variant="error")
            yield Button("≡  Compare",    id="btn-compare", variant="primary")
            yield Static("", id="status")

        with TabbedContent(initial="tab-live", id="tabs"):

            # ── Live tab ─────────────────────────────────────────────────────
            with TabPane("Live Trading", id="tab-live"):
                with Horizontal(id="live-body"):
                    with Vertical(id="metrics-box"):
                        yield Static("  LIVE PAPER-TRADING METRICS",
                                     id="metrics-title")
                        tbl = DataTable(id="metrics-table", zebra_stripes=True)
                        tbl.add_columns(
                            "Model", "Active", "Trades", "Win %",
                            "Prof. Factor", "Net PnL", "Return %", "Worker",
                        )
                        yield tbl
                    with Vertical(id="sidebar"):
                        yield Static("MODEL TOGGLES", classes="sidebar-head")
                        for key in _MODEL_KEYS:
                            on = CONFIG["ACTIVE_MODELS"][key]
                            with Horizontal(classes="model-row"):
                                yield Label(_MODEL_LABEL[key])
                                yield Switch(value=on, id=f"sw-{key}")
                        yield Static("")
                        yield Static("WORKER STATUS", classes="sidebar-head")
                        for key in _MODEL_KEYS:
                            yield Static("", id=f"dot-{key}",
                                         classes="worker-dot")

            # ── Backtest tab ──────────────────────────────────────────────────
            with TabPane("Backtest", id="tab-backtest"):
                with Horizontal(id="bt-body"):
                    with Vertical(id="bt-config"):
                        yield Static("BACKTEST SETTINGS",
                                     classes="sidebar-head")
                        yield Label("Lookback (hours):")
                        yield Input(
                            value=str(CONFIG["BACKTESTING"]["LOOKBACK_HOURS"]),
                            id="bt-hours")
                        yield Label("Initial capital ($):")
                        yield Input(
                            value=str(
                                int(CONFIG["BACKTESTING"]["INITIAL_CAPITAL"])),
                            id="bt-capital")
                        yield Label("Signal threshold:")
                        yield Input(
                            value=str(
                                CONFIG["PAPER_TRADING"]["SIGNAL_THRESHOLD"]),
                            id="bt-threshold")
                        yield Label("Position size %:")
                        yield Input(
                            value=str(
                                CONFIG["BACKTESTING"]["POSITION_SIZE_PCT"]),
                            id="bt-possize")
                        yield Label("Slippage %:")
                        yield Input(
                            value=str(CONFIG["BACKTESTING"]["SLIPPAGE_PCT"]),
                            id="bt-slippage")
                        yield Label("Fee %:")
                        yield Input(
                            value=str(CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"]),
                            id="bt-fee")
                        yield Button("▶  Run Backtest", id="btn-run-bt",
                                     variant="success")
                        yield Static("", id="bt-status")
                    with Vertical(id="bt-results-box"):
                        yield Static("  BACKTEST RESULTS",
                                     id="bt-results-title")
                        bt_tbl = DataTable(id="bt-table", zebra_stripes=True)
                        bt_tbl.add_columns(
                            "Model", "Candles", "Trades", "Win %",
                            "Prof. Factor", "Net PnL", "Return %", "Max DD",
                        )
                        yield bt_tbl
                        yield Static("", id="bt-last-run")

            # ── Models tab ────────────────────────────────────────────────────
            with TabPane("Models", id="tab-models"):
                with Vertical(id="models-outer"):
                    yield Static("", id="models-armed-summary")
                    with Vertical(id="models-cv-section"):
                        yield Static("CV TRAINING SCORES",
                                     classes="sidebar-head")
                        yield Static("[dim]not trained yet[/dim]",
                                     id="cv-summary")
                    with ScrollableContainer(id="models-scroll"):
                        for _mkey in _ALL_MODEL_KEYS:
                            with Vertical(classes="engine-box-stacked"):
                                yield Static(
                                    f"  {_MODEL_LABEL[_mkey].upper()}",
                                    id=f"arch-title-{_mkey}",
                                    classes="engine-title",
                                )
                                _at = DataTable(
                                    id=f"arch-table-{_mkey}",
                                    zebra_stripes=True,
                                    classes="engine-archive-table",
                                )
                                _at.add_columns(
                                    "#", "Timestamp", "P&L",
                                    "Trades", "Win%", "Max DD", "Status",
                                )
                                yield _at
                                with Horizontal(classes="arch-activate-row"):
                                    for _i in range(3):
                                        yield Button(
                                            f"Activate #{_i+1}",
                                            id=f"arch-act-{_mkey}-{_i}",
                                            variant="success" if _i == 0 else "default",
                                            disabled=True,
                                        )
                    yield Button("↺  Refresh Archive",
                                 id="btn-refresh-archive", variant="default")

            # ── Activity tab ──────────────────────────────────────────────────
            with TabPane("Activity", id="tab-activity"):
                with Horizontal(id="activity-body"):
                    with Vertical(id="activity-table-box"):
                        yield Static("  COMPONENT STATUS",
                                     id="activity-title")
                        act_tbl = DataTable(id="activity-table",
                                            zebra_stripes=True)
                        act_tbl.add_columns(
                            "Component", "Status", "Detail", "Age (s)",
                        )
                        yield act_tbl
                    with Vertical(id="activity-log-box"):
                        yield Static("  ACTIVITY LOG",
                                     id="activity-log-title")
                        yield RichLog(id="activity-log", markup=True,
                                      highlight=False, max_lines=500)

            # ── Coins tab ─────────────────────────────────────────────────────
            with TabPane("Coins", id="tab-coins"):
                with ScrollableContainer():
                    with Vertical(id="coins-tier-box"):
                        yield Static("TRAINING TIER", id="coins-tier-head")
                        with Horizontal(id="coins-tier-btns"):
                            for _tier in ["single", "quick", "standard",
                                          "extended", "insane"]:
                                _cur = CONFIG["COINS"]["TRAINING_TIER"]
                                yield Button(
                                    _tier.capitalize(),
                                    id=f"tier-btn-{_tier}",
                                    classes="tier-btn"
                                          + (" -active" if _tier == _cur else ""),
                                    variant="primary" if _tier == _cur else "default",
                                )
                        yield Static("", id="coins-tier-status")
                    with Vertical(id="coins-list-box"):
                        yield Static("TRAINING COINS", id="coins-list-head")
                        yield Static("", id="coins-list-info")

            # ── Exchange & Keys tab ───────────────────────────────────────────
            with TabPane("Exchange & Keys", id="tab-exchange"):
                with TabbedContent(id="exch-tabs"):

                    # Exchange selector sub-tab
                    with TabPane("Exchange", id="exch-tab-exchange"):
                        with ScrollableContainer():
                            with Vertical(id="exch-selector-box"):
                                yield Static("EXCHANGE", classes="settings-head")
                                with Horizontal(id="exch-btns"):
                                    for ex in _EXCHANGES:
                                        _act = " -active" if ex == CONFIG["EXCHANGE"] else ""
                                        yield Button(
                                            ex, id=f"exch-{ex}",
                                            classes=f"exch-btn{_act}",
                                            variant="primary"
                                            if ex == CONFIG["EXCHANGE"] else "default",
                                        )
                                yield Static("", id="exch-status")

                    # API Keys + Balance sub-tab
                    with TabPane("API Keys", id="exch-tab-keys"):
                        with ScrollableContainer():
                            with Vertical(id="api-keys-box"):
                                yield Static("API KEYS", id="api-keys-head")
                                yield Static("", id="api-key-status")
                                yield Label("Key file path (.json / .txt / .zip):")
                                yield Input(
                                    placeholder=r"C:\path\to\coinbase_keys.json",
                                    id="key-file-path",
                                )
                                with Horizontal(id="key-btns"):
                                    yield Button("Load from File",
                                                 id="btn-load-keys", variant="success")
                                    yield Button("Clear Keys",
                                                 id="btn-clear-keys", variant="error")
                                yield Static(
                                    "[dim]JSON export, plain text (2-line or "
                                    "KEY=VAL), or ZIP[/dim]",
                                    id="key-hint",
                                )
                            with Vertical(id="balance-box"):
                                yield Static("COINBASE ACCOUNT",
                                             classes="settings-head")
                                yield Static("[dim]Loading...[/dim]",
                                             id="balance-info")
                                yield Button("↻  Refresh Balance",
                                             id="btn-refresh-balance",
                                             variant="default")
                                yield Static("", id="balance-last-update")

                    # Market Prices + Bulk Download sub-tab
                    with TabPane("Market Data", id="exch-tab-market"):
                        with ScrollableContainer():
                            with Vertical(id="market-data-box"):
                                yield Static("MARKET PRICES  [dim](last cached close)[/dim]",
                                             id="market-data-head")
                                ticker_tbl = DataTable(
                                    id="ticker-table", zebra_stripes=True,
                                )
                                ticker_tbl.add_columns(
                                    "Symbol", "Price", "Change", "Candles",
                                )
                                yield ticker_tbl
                            with Vertical(id="bulk-download-box"):
                                yield Static("BULK COIN DATA DOWNLOAD",
                                             id="bulk-download-head")
                                yield Static(
                                    "[dim]Download 365 days × 15-min candles for all "
                                    "Coinbase USD products.  One-time initial load; "
                                    "coins already cached are skipped automatically.[/dim]",
                                )
                                yield Static("", id="bulk-status")
                                with Horizontal():
                                    yield Button("⬇  Download All Coins",
                                                 id="btn-bulk-start",
                                                 variant="success")
                                    yield Button("⬇  Re-download All",
                                                 id="btn-bulk-redownload",
                                                 variant="warning")
                                    yield Button("■  Stop",
                                                 id="btn-bulk-stop",
                                                 variant="error",
                                                 disabled=True)

            # ── Settings tab ──────────────────────────────────────────────────
            with TabPane("Settings", id="tab-settings"):
                with TabbedContent(id="settings-tabs"):

                    # Data & Training sub-tab
                    with TabPane("Data & Training", id="settings-tab-data"):
                        with ScrollableContainer():
                            with Vertical(id="windows-box"):
                                yield Static("DATA WINDOWS",
                                             classes="settings-head")
                                yield Label("Training lookback (days, 1–1095):")
                                yield Input(
                                    value=str(CONFIG["TRAINING"]["LOOKBACK_DAYS"]),
                                    id="set-train-days")
                                yield Label("Data cache window (days, 1–1095):")
                                yield Input(
                                    value=str(
                                        max(1, CONFIG["DATA_CACHE"]["MAX_HOURS"]
                                            // 24)),
                                    id="set-cache-days")
                                yield Label(
                                    "Cache refresh interval (min, 5–1440):")
                                yield Input(
                                    value=str(
                                        CONFIG["DATA_CACHE"][
                                            "REFRESH_INTERVAL_MIN"]),
                                    id="set-refresh-min")
                                yield Label(
                                    "Default backtest window (days, 1–1095):")
                                yield Input(
                                    value=str(
                                        max(1,
                                            CONFIG["BACKTESTING"]["LOOKBACK_HOURS"]
                                            // 24)),
                                    id="set-bt-days")
                                yield Label("Signal threshold (paper & live):")
                                yield Input(
                                    value=str(
                                        CONFIG["PAPER_TRADING"]["SIGNAL_THRESHOLD"]),
                                    id="set-threshold")
                                yield Label("Coinbase taker fee % (auto-fetched if API keys set):")
                                yield Input(
                                    value=str(
                                        CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"]),
                                    id="set-fee")
                                yield Label("Zero OBI in inference (recommended — matches training):")
                                yield Switch(
                                    value=CONFIG["PAPER_TRADING"].get("ZERO_OBI_IN_INFERENCE", True),
                                    id="sw-zero-obi")
                                yield Button("Apply & Save Settings",
                                             id="btn-apply-settings",
                                             variant="success")
                                yield Static("", id="settings-saved-msg")
                            with Vertical(id="cache-box"):
                                yield Static("DATA CACHE",
                                             classes="settings-head")
                                yield Static("", id="cache-info")
                                yield Button("↺  Force Refresh",
                                             id="btn-force-refresh",
                                             variant="default")

                    # Live Trading sub-tab
                    with TabPane("Live Trading ⚠", id="settings-tab-live"):
                        with ScrollableContainer():
                            with Vertical(id="live-trading-box"):
                                yield Static("LIVE TRADING ⚠",
                                             id="live-trading-head")
                                yield Static("", id="gate-requirements")
                                yield Label("Min paper trading hours before arming:")
                                yield Input(
                                    value=str(CONFIG["LIVE_TRADING"]["MIN_PAPER_HOURS"]),
                                    id="set-min-paper-hours")
                                yield Label("Min paper P&L % required before arming:")
                                yield Input(
                                    value=str(CONFIG["LIVE_TRADING"]["MIN_PAPER_PNL_PCT"]),
                                    id="set-min-paper-pnl")
                                yield Label("Preferred model (used when within tolerance):")
                                yield Input(
                                    value=str(CONFIG["LIVE_TRADING"]["PREFERRED_MODEL"]),
                                    id="set-preferred-model")
                                yield Label("Preferred model tolerance % (max gap vs best):")
                                yield Input(
                                    value=str(CONFIG["LIVE_TRADING"]["PREFERRED_MODEL_TOLERANCE_PCT"]),
                                    id="set-preferred-tolerance")
                                yield Static(
                                    "Armed: "
                                    + CONFIG["LIVE_TRADING"]["ARMED_MODEL"],
                                    id="armed-model-label")
                                yield Static("", id="armed-rival-notice")
                                yield Button("ARM LIVE TRADING",
                                             id="arm-btn", variant="error")
                                yield Static(
                                    "WARNING: Places REAL orders with REAL funds.",
                                    id="live-warning")
                            with Vertical(id="paper-stats-box"):
                                yield Static("PAPER TRADING STATS",
                                             id="paper-stats-head")
                                yield Static("", id="live-trade-stats")

        with Vertical(id="log-box"):
            yield Static("  LOG", id="log-title")
            yield RichLog(id="bot-log", markup=True, highlight=False, max_lines=200)

        yield Footer()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.manager = BotManager(log_fn=self._post_log)
        self._bulk   = BulkDownloader(log_fn=self._post_log)
        self._activity_log_seen: set[tuple] = set()
        self._blink_state: bool = False   # toggled each tick for rival-notice animation
        self.set_interval(2.0,  self._tick)
        self.set_interval(1.0,  self._tick_activity)
        self.set_interval(30.0, self._tick_balances)    # balance panel refresh
        self.set_interval(5.0,  self._refresh_archive_tables)  # models tab
        self.set_interval(10.0, self._refresh_ticker_table)    # market prices

        mode = ("[bold cyan]PAPER[/bold cyan]" if is_demo_mode()
                else f"[bold green]{get_active_symbol()}[/bold green]")
        self._post_log(
            f"Project Challenger ready — {CONFIG['EXCHANGE']} / {mode}"
        )

        if not getattr(self, "_no_autostart", False):
            if models_are_trained():
                self._post_log("Models found — auto-starting bot...")
                self.call_after_refresh(self._do_start_now)
            else:
                self._post_log(
                    "[yellow]No trained models found. Press T or click Retrain.[/yellow]"
                )

        self._sync_buttons()
        self._refresh_cv_summary()
        self._refresh_exch_status()
        self._refresh_cache_info()
        self._refresh_gate_display()
        self._refresh_key_status()
        self.call_after_refresh(self._tick_balances)          # initial balance fetch
        self.call_after_refresh(self._refresh_archive_tables) # initial archive display
        self.call_after_refresh(self._refresh_ticker_table)   # initial price display
        self.call_after_refresh(self._refresh_coins_display)  # initial coins display

    # ── button handler ────────────────────────────────────────────────────────

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        bid = ev.button.id
        {
            "btn-start":        self.action_do_start,
            "btn-stop":         self.action_do_stop,
            "btn-retrain":      self.action_do_retrain,
            "btn-fresh":        self._do_fresh_data,
            "btn-reset":        self._do_reset_db,
            "btn-compare":      self.action_do_compare,
            "btn-run-bt":       self.action_do_backtest,
            "btn-go-live":      self._toggle_live_trading,
            "arm-btn":          self._toggle_live_trading,
            "btn-force-refresh":     self._do_force_refresh,
            "btn-apply-settings":    self._apply_data_settings,
            "btn-refresh-balance":   self._do_refresh_balance,
            "btn-load-keys":         self._do_load_keys,
            "btn-clear-keys":        self._do_clear_keys,
            "btn-refresh-archive":   self._refresh_archive_tables,
            "btn-bulk-start":        lambda: self._do_bulk_download(force=False),
            "btn-bulk-redownload":   lambda: self._do_bulk_download(force=True),
            "btn-bulk-stop":         self._do_bulk_stop,
        }.get(bid, lambda: None)()

        # Exchange selector buttons
        if bid and bid.startswith("exch-"):
            ex = bid.replace("exch-", "")
            if ex in _EXCHANGES:
                self._switch_exchange(ex)

        # Training tier buttons
        if bid and bid.startswith("tier-btn-"):
            self._switch_training_tier(bid.replace("tier-btn-", ""))

        # Archive activate buttons: arch-act-{MODEL_KEY}-{rank_idx}
        if bid and bid.startswith("arch-act-"):
            parts = bid.split("-")
            # Format: arch-act-ENGINE_PART1_PART2-idx  (engine key may contain underscores)
            # Find the last segment as idx, everything between arch-act- and -idx is the key
            try:
                idx      = int(parts[-1])
                model_key = "-".join(parts[2:-1]).upper().replace("-", "_")
                self._activate_archive_model(model_key, idx)
            except (ValueError, IndexError):
                pass

    # ── switch handler ────────────────────────────────────────────────────────

    def on_switch_changed(self, ev: Switch.Changed) -> None:
        switch_id = ev.switch.id

        if switch_id == "sw-zero-obi":
            CONFIG["PAPER_TRADING"]["ZERO_OBI_IN_INFERENCE"] = ev.value
            state = "ON (zeroed — matches training)" if ev.value else "OFF (live OBI used)"
            self._post_log(f"Zero OBI in inference: {state}")
            from config import save_user_settings
            save_user_settings({"ZERO_OBI_IN_INFERENCE": ev.value})
            return

        key = switch_id.replace("sw-", "")
        CONFIG["ACTIVE_MODELS"][key] = ev.value
        state = "enabled" if ev.value else "disabled"
        self._post_log(f"Model {key} {state}.")
        if self.manager.running:
            if ev.value:
                self.manager.start_worker(key)
            else:
                self.manager.stop_worker(key)

    # ── key actions ───────────────────────────────────────────────────────────

    def action_do_start(self) -> None:
        if self._retraining:
            return
        if not models_are_trained():
            self._post_log("[yellow]No models — retraining first.[/yellow]")
            self.action_do_retrain()
            return
        self._do_start_now()

    def action_do_stop(self) -> None:
        if not self.manager.running:
            return
        self.manager.stop()
        self._running = False
        self._sync_buttons()

    def action_do_retrain(self) -> None:
        if self._retraining:
            self._post_log("[yellow]Already retraining.[/yellow]")
            return
        self._retraining = True
        self._sync_buttons()
        self.manager.retrain(done_cb=self._on_retrain_done)

    def action_do_backtest(self) -> None:
        if self._bt_running:
            self._post_log("[yellow]Backtest already running.[/yellow]")
            return
        if not models_are_trained():
            self._post_log("[yellow]Train models first before running a backtest.[/yellow]")
            return
        cfg = self._read_bt_config()
        if cfg is None:
            return
        self._bt_running = True
        self._set_bt_status("[yellow]Running...[/yellow]")
        self.manager.run_backtest(cfg=cfg, done_cb=self._on_backtest_done)

    def action_do_compare(self) -> None:
        self.push_screen(ReportModal(self._build_compare_text()))

    def action_show_activity(self) -> None:
        """Switch to the Activity tab."""
        try:
            self.query_one("#tabs", TabbedContent).active = "tab-activity"
        except NoMatches:
            pass

    def _do_fresh_data(self) -> None:
        self.manager.fetch_fresh_data()

    def _do_force_refresh(self) -> None:
        self.manager.fetch_fresh_data()
        self._post_log("Data cache refresh triggered.")

    def _apply_data_settings(self) -> None:
        """Read the Data Windows inputs, validate, apply to CONFIG, and persist."""
        try:
            train_days   = int(self.query_one("#set-train-days",  Input).value or 7)
            cache_days   = int(self.query_one("#set-cache-days",  Input).value or 1)
            refresh_min  = int(self.query_one("#set-refresh-min", Input).value or 60)
            bt_days      = int(self.query_one("#set-bt-days",     Input).value or 1)
            threshold    = float(self.query_one("#set-threshold", Input).value or CONFIG["PAPER_TRADING"]["SIGNAL_THRESHOLD"])
            fee_pct          = float(self.query_one("#set-fee",             Input).value or CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"])
            zero_obi         = self.query_one("#sw-zero-obi", Switch).value
            min_paper_hours    = int(self.query_one("#set-min-paper-hours",    Input).value or CONFIG["LIVE_TRADING"]["MIN_PAPER_HOURS"])
            min_paper_pnl      = float(self.query_one("#set-min-paper-pnl",   Input).value or CONFIG["LIVE_TRADING"]["MIN_PAPER_PNL_PCT"])
            preferred_model    = (self.query_one("#set-preferred-model",       Input).value or CONFIG["LIVE_TRADING"]["PREFERRED_MODEL"]).strip().upper()
            preferred_tol      = float(self.query_one("#set-preferred-tolerance", Input).value or CONFIG["LIVE_TRADING"]["PREFERRED_MODEL_TOLERANCE_PCT"])
        except (ValueError, Exception) as e:
            self._post_log(f"[red]Invalid setting: {e}[/red]")
            return

        # Clamp to sensible ranges — up to 365×3 = 1095 days
        train_days  = max(1,   min(1095, train_days))
        cache_days  = max(1,   min(1095, cache_days))
        refresh_min = max(5,   min(1440, refresh_min))   # 5 min – 24h
        bt_days     = max(1,   min(cache_days, bt_days))
        threshold   = max(0.00001, min(0.05,  threshold))
        fee_pct         = max(0.0,  min(0.10,   fee_pct))
        min_paper_hours = max(1,    min(720,    min_paper_hours))
        min_paper_pnl   = max(0.0,  min(100.0,  min_paper_pnl))
        preferred_tol   = max(0.0,  min(100.0,  preferred_tol))

        cache_hours = cache_days * 24
        bt_hours    = bt_days   * 24

        # Apply to live CONFIG immediately
        CONFIG["TRAINING"]["LOOKBACK_DAYS"]          = train_days
        CONFIG["DATA_CACHE"]["MAX_HOURS"]            = cache_hours
        CONFIG["DATA_CACHE"]["REFRESH_INTERVAL_MIN"] = refresh_min
        CONFIG["BACKTESTING"]["LOOKBACK_HOURS"]      = bt_hours
        CONFIG["PAPER_TRADING"]["SIGNAL_THRESHOLD"]      = threshold
        CONFIG["PAPER_TRADING"]["COINBASE_FEE_PCT"]      = fee_pct
        CONFIG["PAPER_TRADING"]["ZERO_OBI_IN_INFERENCE"] = zero_obi
        CONFIG["LIVE_TRADING"]["MIN_PAPER_HOURS"]                = min_paper_hours
        CONFIG["LIVE_TRADING"]["MIN_PAPER_PNL_PCT"]              = min_paper_pnl
        CONFIG["LIVE_TRADING"]["PREFERRED_MODEL"]                = preferred_model
        CONFIG["LIVE_TRADING"]["PREFERRED_MODEL_TOLERANCE_PCT"]  = preferred_tol

        # Also sync the Backtest tab default inputs
        try:
            self.query_one("#bt-hours",     Input).value = str(bt_hours)
            self.query_one("#bt-threshold", Input).value = str(threshold)
            self.query_one("#bt-fee",       Input).value = str(fee_pct)
        except NoMatches:
            pass

        # Persist to data/user_settings.json
        save_user_settings({
            "TRAINING_LOOKBACK_DAYS": train_days,
            "DATA_CACHE_MAX_DAYS":    cache_days,
            "DATA_CACHE_REFRESH_MIN": refresh_min,
            "BACKTEST_LOOKBACK_DAYS": bt_days,
            "SIGNAL_THRESHOLD":       threshold,
            "COINBASE_FEE_PCT":       fee_pct,
            "ZERO_OBI_IN_INFERENCE":  zero_obi,
            "MIN_PAPER_HOURS":                min_paper_hours,
            "MIN_PAPER_PNL_PCT":              min_paper_pnl,
            "PREFERRED_MODEL":                preferred_model,
            "PREFERRED_MODEL_TOLERANCE_PCT":  preferred_tol,
        })

        self._post_log(
            f"Settings saved — train={train_days}d  cache={cache_days}d ({cache_hours}h)  "
            f"refresh={refresh_min}min  bt={bt_days}d  threshold={threshold}  fee={fee_pct*100:.4f}%"
        )
        try:
            self.query_one("#settings-saved-msg", Static).update(
                "[green]Saved. Retrain to apply training-day changes.[/green]"
            )
        except NoMatches:
            pass

    def _do_reset_db(self) -> None:
        self.push_screen(ConfirmModal(
            "Clear ALL paper-trading data?\nThis cannot be undone.",
            self.manager.reset_db,
        ))

    def _toggle_live_trading(self) -> None:
        if self.manager.live_trader.is_armed:
            self.manager.disarm_live_trading()
            self._live_armed = False
            self._refresh_gate_display()
            self._sync_arm_buttons()
        else:
            ok, msg = self.manager.arm_live_trading()
            if ok:
                self._live_armed = True
                self._post_log(f"[bold green]{msg}[/bold green]")
            else:
                self._post_log(f"[red]Cannot arm: {msg}[/red]")
            self._refresh_gate_display()
            self._sync_arm_buttons()

    def _switch_exchange(self, ex: str) -> None:
        if self.manager.running:
            self._post_log("[yellow]Stop the bot before switching exchanges.[/yellow]")
            return
        CONFIG["EXCHANGE"] = ex
        self._post_log(f"Exchange switched to {ex}. Restart bot to apply.")
        self._refresh_exch_status()
        self._refresh_key_status()
        # Update button styles
        for e in _EXCHANGES:
            try:
                btn = self.query_one(f"#exch-{e}", Button)
                if e == ex:
                    btn.variant = "primary"
                    btn.add_class("-active")
                else:
                    btn.variant = "default"
                    btn.remove_class("-active")
            except NoMatches:
                pass

    # ── internals ─────────────────────────────────────────────────────────────

    def _do_start_now(self) -> None:
        self.manager.start()
        self._running = True
        self._sync_buttons()

    def _on_retrain_done(self) -> None:
        self.call_from_thread(self._after_retrain)

    def _after_retrain(self) -> None:
        self._retraining = False
        self._refresh_cv_summary()
        self._sync_buttons()
        self._refresh_archive_tables()   # show newly archived models
        if not self.manager.running:
            self._do_start_now()

    def _on_backtest_done(self, results: dict) -> None:
        self.call_from_thread(self._after_backtest, results)

    def _after_backtest(self, results: dict) -> None:
        self._bt_running = False
        self._set_bt_status(
            f"[green]Done — {datetime.now().strftime('%H:%M:%S')}[/green]"
        )
        self._refresh_bt_table(results)

    def _post_log(self, msg: str) -> None:
        try:
            self.call_from_thread(self._write_log, msg)
        except Exception:
            try:
                self.query_one("#bot-log", RichLog).write(msg)
            except Exception:
                pass

    def _write_log(self, msg: str) -> None:
        try:
            self.query_one("#bot-log", RichLog).write(msg)
        except NoMatches:
            pass

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#status", Static).update(text)
        except NoMatches:
            pass

    def _set_bt_status(self, text: str) -> None:
        try:
            self.query_one("#bt-status", Static).update(text)
        except NoMatches:
            pass

    def _sync_buttons(self) -> None:
        running    = self.manager.running
        retraining = self._retraining
        try:
            self.query_one("#btn-start",   Button).disabled = running or retraining
            self.query_one("#btn-stop",    Button).disabled = not running or retraining
            self.query_one("#btn-retrain", Button).disabled = retraining
            self.query_one("#btn-reset",   Button).disabled = retraining
            self.query_one("#btn-run-bt",  Button).disabled = self._bt_running or retraining
        except NoMatches:
            pass

        if retraining:
            self._set_status("[bold yellow]⟳  RETRAINING[/bold yellow]")
        elif running:
            sym = get_active_symbol()
            tag = "PAPER" if is_demo_mode() else sym
            ex  = CONFIG["EXCHANGE"]
            self._set_status(f"[bold green]●  RUNNING  {ex}/{tag}[/bold green]")
        else:
            self._set_status("[bold red]■  STOPPED[/bold red]")

    def _sync_arm_buttons(self) -> None:
        armed = self.manager.live_trader.is_armed
        label = "DISARM LIVE TRADING" if armed else "ARM LIVE TRADING"
        for bid in ("#btn-go-live", "#arm-btn"):
            try:
                btn = self.query_one(bid, Button)
                btn.label = label
                if armed:
                    btn.add_class("-armed")
                else:
                    btn.remove_class("-armed")
            except NoMatches:
                pass

    def _read_bt_config(self) -> BacktestConfig | None:
        """Parse the backtest Input fields into a BacktestConfig."""
        try:
            hours     = int(self.query_one("#bt-hours",     Input).value or 24)
            capital   = float(self.query_one("#bt-capital",   Input).value or 10000)
            threshold = float(self.query_one("#bt-threshold", Input).value or CONFIG["PAPER_TRADING"]["SIGNAL_THRESHOLD"])
            possize   = float(self.query_one("#bt-possize",   Input).value or 0.10)
            slippage  = float(self.query_one("#bt-slippage",  Input).value or 0.0005)
            fee       = float(self.query_one("#bt-fee",       Input).value or 0.006)
            max_hours = CONFIG["DATA_CACHE"]["MAX_HOURS"]
            return BacktestConfig(
                lookback_hours    = max(1, min(max_hours, hours)),
                initial_capital   = max(100, capital),
                signal_threshold  = max(0.00001, threshold),
                position_size_pct = max(0.01, min(1.0, possize)),
                slippage_pct      = max(0, slippage),
                fee_pct           = max(0, fee),
            )
        except ValueError as e:
            self._post_log(f"[red]Invalid backtest settings: {e}[/red]")
            return None

    # ── periodic refresh ──────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._refresh_live_table()
        self._refresh_worker_dots()
        self._sync_buttons()
        self._refresh_cache_info()
        self._refresh_gate_display()
        self._refresh_armed_model()

    def _refresh_armed_model(self) -> None:
        """Re-run model selection and update the armed label + rival blink notice."""
        try:
            from model_archive import select_armed_model, list_archive_for
            chosen_key, chosen_pnl, rival_key = select_armed_model()

            # Update armed model label
            try:
                self.query_one("#armed-model-label", Static).update(
                    f"Armed: [bold cyan]{chosen_key}[/bold cyan]  "
                    f"(archive P&L {chosen_pnl:+.2f})"
                )
            except Exception:
                pass

            # Rival notice — blink on/off each tick
            rival_widget = self.query_one("#armed-rival-notice", Static)
            if rival_key:
                rival_models = list_archive_for(rival_key)
                rival_pnl    = rival_models[0].pnl if rival_models else 0.0
                self._blink_state = not self._blink_state
                star  = "★" if self._blink_state else "☆"
                style = "bold yellow" if self._blink_state else "dim yellow"
                rival_widget.update(
                    f"[{style}]{star} NOTE: {rival_key} has higher archive P&L "
                    f"({rival_pnl:+.2f}) — preferred model {chosen_key} selected "
                    f"(within {CONFIG['LIVE_TRADING']['PREFERRED_MODEL_TOLERANCE_PCT']:.0f}% tolerance)[/{style}]"
                )
            else:
                rival_widget.update("")
        except Exception:
            pass

    def _tick_activity(self) -> None:
        """1-second Activity tab refresh: component status table + rolling log."""
        import time as _time
        tracker = get_tracker()

        # ── component status table ─────────────────────────────────────────────
        try:
            tbl = self.query_one("#activity-table", DataTable)
            tbl.clear()
            now = _time.time()
            for comp in tracker.get_components():
                name    = comp["name"]
                status  = comp["status"]
                msg     = comp["message"] or "—"
                age     = f"{now - comp['ts']:.1f}" if comp["ts"] else "—"
                color   = tracker.color_for(status)
                tbl.add_row(
                    f"[bold]{name}[/bold]",
                    f"[{color}]{status}[/{color}]",
                    msg,
                    age,
                    key=name,
                )
        except (NoMatches, Exception):
            pass

        # ── rolling activity log ───────────────────────────────────────────────
        try:
            log_widget = self.query_one("#activity-log", RichLog)
            entries    = tracker.get_log(n=200)
            for entry in entries:
                key = (entry["ts"], entry["component"], entry["status"])
                if key in self._activity_log_seen:
                    continue
                self._activity_log_seen.add(key)
                # Trim cache to prevent unbounded growth
                if len(self._activity_log_seen) > 1000:
                    oldest = sorted(self._activity_log_seen)[:200]
                    for k in oldest:
                        self._activity_log_seen.discard(k)

                from datetime import datetime as _dt
                ts_str  = _dt.fromtimestamp(entry["ts"]).strftime("%H:%M:%S")
                comp    = entry["component"]
                status  = entry["status"]
                msg     = entry["message"] or ""
                color   = tracker.color_for(status)
                log_widget.write(
                    f"[dim]{ts_str}[/dim]  "
                    f"[bold]{comp:<12}[/bold]  "
                    f"[{color}]{status:<13}[/{color}]  "
                    f"{msg}"
                )
        except (NoMatches, Exception):
            pass

    def _refresh_live_table(self) -> None:
        try:
            tbl = self.query_one("#metrics-table", DataTable)
        except NoMatches:
            return
        tbl.clear()
        for key in _MODEL_KEYS:
            active = CONFIG["ACTIVE_MODELS"][key]
            s = get_model_stats(key) if active else None

            if s and s[0]:
                count, wins, losses, pnl, fees, slip, gp, gl = s
                pnl   = pnl or 0.0
                ret   = (pnl / _INITIAL_CAP) * 100
                pnl_s = _fmt_pnl(pnl)
                ret_s = _fmt_pnl(ret) + "%"
                wr_s  = _fmt_wr(wins, losses)
                pf_s  = _fmt_pf(gp, gl)
                cnt_s = str(count or 0)
            else:
                pnl_s = ret_s = wr_s = pf_s = cnt_s = "—"

            alive = self.manager._procs.get(key)
            alive = alive.is_alive() if alive else False
            w_dot = ("[green]●[/green]" if alive
                     else ("[dim]○[/dim]" if not active else "[red]✗[/red]"))
            act_s = "[green]ON[/green]" if active else "[dim]off[/dim]"

            tbl.add_row(
                _MODEL_LABEL[key], act_s, cnt_s, wr_s, pf_s, pnl_s, ret_s, w_dot,
                key=key,
            )

    def _refresh_worker_dots(self) -> None:
        status = self.manager.worker_status()
        for key in _MODEL_KEYS:
            alive  = status.get(key, False)
            active = CONFIG["ACTIVE_MODELS"][key]
            if not active:
                txt = "[dim]○  disabled[/dim]"
            elif alive:
                txt = f"[green]●  {key} running[/green]"
            else:
                txt = f"[red]✗  {key} dead[/red]"
            try:
                self.query_one(f"#dot-{key}", Static).update(txt)
            except NoMatches:
                pass

    def _refresh_cv_summary(self) -> None:
        rows = get_training_summary()
        txt  = (
            "\n".join(
                f"[dim]{r[0].split('_')[0]}[/dim]  "
                f"DA=[cyan]{r[4]:.3f}[/cyan]"
                for r in rows
            )
            if rows else "[dim]not trained yet[/dim]"
        )
        try:
            self.query_one("#cv-summary", Static).update(txt)
        except NoMatches:
            pass

    def _refresh_bt_table(self, results: dict) -> None:
        try:
            tbl = self.query_one("#bt-table", DataTable)
        except NoMatches:
            return
        tbl.clear()
        for key, r in results.items():
            if r.error:
                tbl.add_row(
                    _MODEL_LABEL[key], "—", "—", "—", "—",
                    f"[red]{r.error[:30]}[/red]", "—", "—",
                    key=key,
                )
                continue
            pf  = (f"{r.profit_factor:.2f}"
                   if r.profit_factor != float("inf") else "inf")
            c   = "green" if r.net_pnl >= 0 else "red"
            ret = r.net_return_pct
            tbl.add_row(
                _MODEL_LABEL[key],
                str(r.total_candles),
                str(r.total_trades),
                f"{r.win_rate:.0%}",
                pf,
                f"[{c}]{r.net_pnl:+.2f}[/{c}]",
                f"[{c}]{ret:+.2f}%[/{c}]",
                f"{r.max_drawdown:.1%}",
                key=key,
            )

    def _refresh_exch_status(self) -> None:
        ex   = CONFIG["EXCHANGE"]
        demo = is_demo_mode()
        sym  = get_active_symbol()
        if ex == "COINBASE":
            txt = f"[dim]COINBASE / {sym} / {'PAPER' if demo else 'LIVE DATA'}[/dim]"
        else:
            txt = f"[dim]{ex} / {sym} / Public data (no keys needed)[/dim]"
        try:
            self.query_one("#exch-status", Static).update(txt)
        except NoMatches:
            pass

    def _refresh_cache_info(self) -> None:
        try:
            dm    = self.manager.data_mgr
            count = dm.candle_count
            last  = dm.last_refresh
            lr    = last.strftime("%H:%M:%S") if last else "never"
            txt   = (
                f"Exchange: [cyan]{dm.exchange_name}[/cyan]  "
                f"Symbol: [cyan]{dm.symbol}[/cyan]\n"
                f"Cached: [cyan]{count}[/cyan] candles\n"
                f"Last refresh: [cyan]{lr}[/cyan]"
            )
            self.query_one("#cache-info", Static).update(txt)
        except Exception:
            pass

    def _refresh_gate_display(self) -> None:
        try:
            trader = self.manager.live_trader
            lines  = trader.gate_status()
            # Compact format for sidebar
            compact = "\n".join(
                ("[green]✓[/green]" if ok else "[red]✗[/red]") + f" {desc}"
                for ok, desc in lines
            )
            try:
                self.query_one("#live-gate-display", Static).update(compact)
            except NoMatches:
                pass
            # Full format for settings tab
            full = "\n".join(
                ("[green]✓[/green]" if ok else "[red]✗[/red]") + f"  {desc}"
                for ok, desc in lines
            )
            try:
                self.query_one("#gate-requirements", Static).update(full)
            except NoMatches:
                pass
        except AttributeError:
            pass

        self._sync_arm_buttons()

        # Live trade stats in settings tab
        try:
            model  = CONFIG["LIVE_TRADING"]["ARMED_MODEL"]
            stats  = get_live_trade_stats(model)
            if stats and stats[0]:
                cnt, wins, losses, pnl, fees = stats
                pnl   = pnl or 0.0
                c     = "green" if pnl >= 0 else "red"
                txt   = (
                    f"Model: [cyan]{model}[/cyan]\n"
                    f"Live trades: [cyan]{cnt or 0}[/cyan]  "
                    f"Wins: [cyan]{wins or 0}[/cyan]\n"
                    f"Net PnL: [{c}]{pnl:+.2f}[/{c}]  "
                    f"Fees: [dim]{fees or 0:.2f}[/dim]"
                )
            else:
                txt = f"[dim]No live trades yet for {model}[/dim]"
            self.query_one("#live-trade-stats", Static).update(txt)
        except (NoMatches, AttributeError):
            pass

    # ── account balance panel ─────────────────────────────────────────────────

    def _do_refresh_balance(self) -> None:
        """Manual refresh button — immediately re-fetch account balances."""
        self._tick_balances()

    def _tick_balances(self) -> None:
        """
        Refresh the COINBASE ACCOUNT balance panel.
        - In paper mode or non-Coinbase exchange: show an info message, no API call.
        - Otherwise: spawn a daemon thread to fetch balances, then update the UI
          via call_from_thread() so we never block the Textual event loop.
        """
        import threading

        if is_demo_mode() or CONFIG["EXCHANGE"] != "COINBASE":
            try:
                self.query_one("#balance-info", Static).update(
                    "[dim]Add real Coinbase API keys\nto see live balances.[/dim]"
                )
                self.query_one("#balance-last-update", Static).update("")
            except NoMatches:
                pass
            return

        # Mark as fetching so the user sees feedback immediately
        try:
            self.query_one("#balance-info", Static).update(
                "[dim]Fetching...[/dim]"
            )
        except NoMatches:
            pass

        def _fetch():
            try:
                from exchanges.coinbase import CoinbaseExchange
                accounts = CoinbaseExchange().get_accounts()
            except Exception:
                accounts = []
            self.call_from_thread(self._update_balance_display, accounts)

        threading.Thread(target=_fetch, daemon=True).start()

    def _update_balance_display(self, accounts: list) -> None:
        """
        Update the balance Static widget (must be called from the main thread).
        accounts: list of {currency, available, hold, total} dicts.
        """
        try:
            widget = self.query_one("#balance-info", Static)
        except NoMatches:
            return

        if not accounts:
            widget.update(
                "[red]Could not fetch balances.[/red]\n"
                "[dim]Check API keys and network.[/dim]"
            )
        else:
            lines = []
            for acct in accounts[:12]:   # cap display at 12 currencies
                curr  = acct["currency"]
                avail = acct["available"]
                hold  = acct["hold"]
                if curr == "USD":
                    lines.append(f"[cyan]USD[/cyan]   ${avail:>12,.2f}")
                    if hold > 0:
                        lines.append(f"       [dim]hold  ${hold:,.2f}[/dim]")
                else:
                    lines.append(f"[cyan]{curr:<5}[/cyan]  {avail:>14.8f}")
                    if hold > 0:
                        lines.append(f"       [dim]hold  {hold:.8f}[/dim]")
            widget.update("\n".join(lines))

        try:
            ts = datetime.now().strftime("%H:%M:%S")
            self.query_one("#balance-last-update", Static).update(
                f"[dim]Updated {ts}[/dim]"
            )
        except NoMatches:
            pass

    # ── Bulk coin downloader ──────────────────────────────────────────────────

    def _do_bulk_download(self, force: bool = False) -> None:
        if self._bulk.running:
            self._post_log("[yellow]Bulk download already in progress.[/yellow]")
            return
        self._post_log(
            f"[cyan]Starting bulk download ({'force re-download' if force else 'skip cached'})...[/cyan]"
        )
        self._bulk.start(force=force)
        self._sync_bulk_buttons()
        # Poll for progress every 5 seconds
        self.set_interval(5.0, self._refresh_bulk_status)

    def _do_bulk_stop(self) -> None:
        self._bulk.stop()
        self._post_log("[yellow]Bulk download stop requested.[/yellow]")

    def _sync_bulk_buttons(self) -> None:
        running = self._bulk.running
        for bid in ("#btn-bulk-start", "#btn-bulk-redownload"):
            try:
                self.query_one(bid, Button).disabled = running
            except NoMatches:
                pass
        try:
            self.query_one("#btn-bulk-stop", Button).disabled = not running
        except NoMatches:
            pass

    def _refresh_bulk_status(self) -> None:
        try:
            self.query_one("#bulk-status", Static).update(self._bulk.status_line)
        except NoMatches:
            pass
        self._sync_bulk_buttons()
        if not self._bulk.running:
            self._refresh_ticker_table()   # update prices after download

    # ── Market price ticker ───────────────────────────────────────────────────

    def _refresh_ticker_table(self) -> None:
        """
        Populate the Market Prices DataTable from the latest cached candle closes.
        Shows symbol, last price, % change vs previous candle, and candle count.
        """
        try:
            tbl = self.query_one("#ticker-table", DataTable)
        except NoMatches:
            return

        try:
            rows = get_latest_prices(CONFIG["EXCHANGE"])
        except Exception:
            return

        tbl.clear()
        if not rows:
            tbl.add_row("—", "—", "—", "—", key="empty")
            return

        for symbol, last_close, prev_close in rows:
            # Price formatting
            if last_close >= 1000:
                price_s = f"${last_close:,.2f}"
            elif last_close >= 1:
                price_s = f"${last_close:.4f}"
            else:
                price_s = f"${last_close:.6f}"

            # Change %
            if prev_close and prev_close > 0:
                chg = (last_close - prev_close) / prev_close * 100
                chg_color = "green" if chg >= 0 else "red"
                chg_s = f"[{chg_color}]{chg:+.2f}%[/{chg_color}]"
            else:
                chg_s = "—"

            # Candle count
            from database import get_candle_cache_count
            count = get_candle_cache_count(CONFIG["EXCHANGE"], symbol)
            count_s = f"{count:,}" if count else "—"

            tbl.add_row(symbol, price_s, chg_s, count_s, key=symbol)

    # ── API key import ────────────────────────────────────────────────────────

    def _refresh_key_status(self) -> None:
        """Update the API key status label in the Settings tab."""
        try:
            widget = self.query_one("#api-key-status", Static)
        except NoMatches:
            return
        if is_demo_mode():
            widget.update("[yellow]PAPER — no API keys loaded[/yellow]")
        else:
            key = CONFIG["COINBASE"]["API_KEY"]
            # Show a masked preview: first 4 + last 4 chars
            if len(key) >= 8:
                masked = key[:4] + "•" * (len(key) - 8) + key[-4:]
            else:
                masked = key[:2] + "•" * max(0, len(key) - 2)
            widget.update(f"[green]● Keys active[/green]\n[dim]{masked}[/dim]")

    def _do_load_keys(self) -> None:
        """Read the file path input, parse the key file, save if valid."""
        try:
            path = self.query_one("#key-file-path", Input).value.strip()
        except NoMatches:
            return

        if not path:
            self._post_log("[yellow]Enter a file path first.[/yellow]")
            return

        result = _parse_key_file(path)
        if result is None:
            self._post_log(
                f"[red]Could not parse keys from:[/red] {path}\n"
                "[red]Expected: JSON with api_key+api_secret, "
                "TXT with two lines or KEY=VAL pairs, or ZIP containing either.[/red]"
            )
            return

        api_key, api_secret = result
        save_api_keys(api_key, api_secret)

        self._post_log(
            f"[green]API keys loaded and saved.[/green]  "
            f"Key: {api_key[:4]}{'•' * max(0, len(api_key) - 8)}{api_key[-4:] if len(api_key) >= 8 else ''}"
        )
        self._refresh_key_status()
        self._refresh_exch_status()
        # Immediately fetch balances now that keys are live
        self.call_after_refresh(self._tick_balances)

    def _do_clear_keys(self) -> None:
        """Confirm then delete saved API keys, reverting to paper mode."""
        def _confirmed():
            clear_api_keys()
            self._post_log("[yellow]API keys cleared. Reverting to PAPER mode.[/yellow]")
            self._refresh_key_status()
            self._refresh_exch_status()
            self.call_after_refresh(self._tick_balances)

        self.push_screen(ConfirmModal(
            "Delete saved API keys?\nBot will revert to PAPER mode.",
            _confirmed,
        ))

    # ── models / archive tab ─────────────────────────────────────────────────

    def _activate_archive_model(self, model_key: str, rank: int) -> None:
        """
        Manually promote an archived model (by rank 0=best) to the active slot,
        pin it as the preferred model, and refresh the display.
        """
        try:
            from model_archive import list_archive_for, activate_archive_model
            models = list_archive_for(model_key)
            if rank >= len(models):
                self._post_log(
                    f"[yellow]No model at rank {rank+1} for {model_key}[/yellow]"
                )
                return
            m = models[rank]
            ok = activate_archive_model(model_key, m.path)
            if ok:
                self._post_log(
                    f"[bold green]Activated {model_key} (rank {rank+1}, "
                    f"P&L {m.pnl:+.2f}) — now armed as preferred model.[/bold green]"
                )
                self._refresh_archive_tables()
                self._refresh_gate_display()
            else:
                self._post_log(
                    f"[red]Could not activate {model_key} rank {rank+1} "
                    f"— archive file missing?[/red]"
                )
        except Exception as e:
            self._post_log(f"[red]Activate error: {e}[/red]")

    def _refresh_archive_tables(self) -> None:
        """
        Populate the Models tab DataTables from the on-disk model archive.
        Called every 5 s and on retrain-complete.  Safe to call at any time
        (no-ops gracefully when the widgets aren't mounted yet).
        """
        try:
            from model_archive import list_all_archives
        except ImportError:
            return

        try:
            archives = list_all_archives()
        except Exception:
            return

        # ── summary bar ───────────────────────────────────────────────────────
        armed_key  = CONFIG["LIVE_TRADING"].get("ARMED_MODEL", "—")
        preferred_key = CONFIG["LIVE_TRADING"].get("PREFERRED_MODEL", armed_key)
        best_entry = None
        for models in archives.values():
            if models and models[0].is_armed:
                best_entry = models[0]

        if best_entry:
            summary = (
                f"Armed: [bold cyan]{armed_key}[/bold cyan]  "
                f"Preferred: [yellow]{preferred_key}[/yellow]  "
                f"Best archive P&L: {_fmt_pnl(best_entry.pnl)}  "
                f"[dim]Click Activate to manually override auto-selection[/dim]"
            )
        else:
            any_models = any(v for v in archives.values())
            summary = (
                f"Armed: [bold cyan]{armed_key}[/bold cyan]  "
                f"Preferred: [yellow]{preferred_key}[/yellow]  "
                + ("[dim]Archive present — retrain to score[/dim]"
                   if any_models else
                   "[dim]No archive yet — retrain to populate[/dim]")
            )

        try:
            self.query_one("#models-armed-summary", Static).update(summary)
        except NoMatches:
            return   # tab not yet mounted

        # ── per-engine tables ─────────────────────────────────────────────────
        for key in _ALL_MODEL_KEYS:
            try:
                tbl = self.query_one(f"#arch-table-{key}", DataTable)
            except NoMatches:
                continue

            tbl.clear()
            models = archives.get(key, [])

            # Update engine title to flag armed/preferred/active state
            title_suffix = ""
            if key == armed_key:
                title_suffix = "  [bold green]◀ ARMED[/bold green]"
            elif key == preferred_key:
                title_suffix = "  [yellow]◀ preferred[/yellow]"
            elif models and models[0].is_active:
                title_suffix = "  [cyan]◀ active[/cyan]"
            try:
                self.query_one(f"#arch-title-{key}", Static).update(
                    f"  {_MODEL_LABEL.get(key, key).upper()}{title_suffix}"
                )
            except NoMatches:
                pass

            if not models:
                tbl.add_row("—", "—", "—", "—", "—", "—",
                            "[dim]no archive[/dim]", key="empty")
                # Disable all activate buttons for this engine
                for i in range(3):
                    try:
                        self.query_one(f"#arch-act-{key}-{i}", Button).disabled = True
                    except NoMatches:
                        pass
                continue

            for rank, m in enumerate(models):
                raw = m.timestamp
                try:
                    ts_fmt = (
                        f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]} "
                        f"{raw[9:11]}:{raw[11:13]}"
                    )
                except Exception:
                    ts_fmt = raw

                # Rank indicator with P&L color
                if m.pnl > 20:
                    pnl_color = "bold green"
                elif m.pnl > 0:
                    pnl_color = "green"
                elif m.pnl == 0:
                    pnl_color = "yellow"
                else:
                    pnl_color = "red"

                rank_s  = f"[{pnl_color}]#{rank+1}[/{pnl_color}]"
                pnl_s   = f"[{pnl_color}]{m.pnl:+.2f}[/{pnl_color}]"
                trades_s = str(m.trades) if m.trades else "—"
                wr_s    = (f"[{'green' if m.win_rate >= 0.55 else 'yellow' if m.win_rate >= 0.45 else 'red'}]"
                           f"{m.win_rate*100:.0f}%"
                           f"[/{'green' if m.win_rate >= 0.55 else 'yellow' if m.win_rate >= 0.45 else 'red'}]"
                           if m.trades else "—")
                dd_s    = (f"[{'green' if m.max_dd <= 0.05 else 'yellow' if m.max_dd <= 0.15 else 'red'}]"
                           f"{m.max_dd*100:.1f}%"
                           f"[/{'green' if m.max_dd <= 0.05 else 'yellow' if m.max_dd <= 0.15 else 'red'}]"
                           if m.trades else "—")

                badges: list[str] = []
                if m.is_armed:
                    badges.append("[bold green]ARMED[/bold green]")
                if m.is_active:
                    badges.append("[cyan]active[/cyan]")
                if key == preferred_key and rank == 0:
                    badges.append("[yellow]preferred[/yellow]")
                status = "  ".join(badges) if badges else "[dim]archived[/dim]"

                tbl.add_row(rank_s, ts_fmt, pnl_s, trades_s, wr_s, dd_s, status,
                            key=m.path)

                # Enable the corresponding activate button
                try:
                    self.query_one(f"#arch-act-{key}-{rank}", Button).disabled = False
                except NoMatches:
                    pass

            # Disable buttons for slots beyond archive length
            for i in range(len(models), 3):
                try:
                    self.query_one(f"#arch-act-{key}-{i}", Button).disabled = True
                except NoMatches:
                    pass

    # ── training tier / coins ─────────────────────────────────────────────────

    def _switch_training_tier(self, tier: str) -> None:
        """Switch the active training tier, persist, and update the Coins tab."""
        from coin_manager import TIER_COINS
        if tier not in TIER_COINS:
            return
        CONFIG["COINS"]["TRAINING_TIER"] = tier
        from config import save_user_settings
        save_user_settings({"TRAINING_TIER": tier})
        self._post_log(f"Training tier → [cyan]{tier}[/cyan]")
        self._refresh_coins_display()
        # Update button highlight states
        for t in ["single", "quick", "standard", "extended", "insane"]:
            try:
                btn = self.query_one(f"#tier-btn-{t}", Button)
                if t == tier:
                    btn.variant = "primary"
                    btn.add_class("-active")
                else:
                    btn.variant = "default"
                    btn.remove_class("-active")
            except NoMatches:
                pass

    def _refresh_coins_display(self) -> None:
        """Populate the Coins tab status and coin list widgets."""
        try:
            from coin_manager import get_training_bases, get_training_symbols
            tier  = CONFIG["COINS"]["TRAINING_TIER"]
            bases = get_training_bases()
            syms  = get_training_symbols()
            if tier == "insane":
                coins_txt = (
                    f"[dim]All available USD-quoted coins on "
                    f"{CONFIG['EXCHANGE']}[/dim]\n"
                    f"[cyan]{len(syms)}[/cyan] symbols in cache"
                )
            else:
                coins_txt = "  ".join(
                    f"[cyan]{b}[/cyan]" for b in bases
                )
            status_txt = (
                f"Active tier: [bold cyan]{tier.upper()}[/bold cyan]  "
                f"· {len(syms)} symbol{'s' if len(syms) != 1 else ''}"
            )
            try:
                self.query_one("#coins-tier-status", Static).update(status_txt)
            except NoMatches:
                pass
            try:
                self.query_one("#coins-list-info", Static).update(coins_txt)
            except NoMatches:
                pass
        except Exception as e:
            self._post_log(f"[red]Coins display error: {e}[/red]")

    # ── compare report ────────────────────────────────────────────────────────

    def _build_compare_text(self) -> str:
        from io import StringIO
        buf = StringIO()

        buf.write("[bold cyan]── CV Training Results ──[/bold cyan]\n\n")
        rows = get_training_summary()
        if rows:
            buf.write(f"  {'Model':<20} {'Folds':>5} {'Avg MSE':>12} {'DirAcc':>8}\n")
            buf.write("  " + "─" * 48 + "\n")
            best_da = max(r[4] for r in rows)
            for model, folds, mse, mae, da, _ in rows:
                c = "green" if da == best_da else "white"
                buf.write(
                    f"  {model:<20} {folds:>5} {mse:>12.8f} "
                    f"[{c}]{da:>8.3f}[/{c}]\n"
                )
        else:
            buf.write("  [yellow]No training results. Run Retrain first.[/yellow]\n")

        buf.write("\n[bold cyan]── Live Paper-Trading ──[/bold cyan]\n\n")
        buf.write(
            f"  {'Model':<20} {'Trades':>7} {'Win%':>7} "
            f"{'PF':>7} {'Net PnL':>12} {'Return%':>9}\n"
        )
        buf.write("  " + "─" * 66 + "\n")
        any_live = False
        for key in _MODEL_KEYS:
            s = get_model_stats(key)
            if not s or not s[0]:
                continue
            any_live = True
            count, wins, losses, pnl, fees, slip, gp, gl = s
            pnl = pnl or 0.0
            ret = pnl / _INITIAL_CAP * 100
            w   = wins   or 0
            l   = losses or 0
            wr  = f"{w/(w+l)*100:.0f}%" if (w+l) > 0 else "—"
            pf  = f"{gp/gl:.2f}" if (gl or 0) > 0 else "—"
            c   = "green" if pnl >= 0 else "red"
            buf.write(
                f"  {key:<20} {count:>7} {wr:>7} {pf:>7} "
                f"[{c}]{pnl:>+11.2f}[/{c}] [{c}]{ret:>+8.2f}%[/{c}]\n"
            )
        if not any_live:
            buf.write("  [yellow]No closed trades yet.[/yellow]\n")

        buf.write("\n[bold cyan]── Most Recent Backtests ──[/bold cyan]\n\n")
        bt_rows = get_backtest_results(limit=6)
        if bt_rows:
            buf.write(
                f"  {'Model':<20} {'Trades':>7} {'Win%':>6} "
                f"{'PF':>6} {'PnL':>10} {'DD':>7}\n"
            )
            buf.write("  " + "─" * 60 + "\n")
            for r in bt_rows:
                mn, ran_at, ex, sym, lbh, trades, wins, losses, pnl, wr, pf, dd = r
                c    = "green" if pnl >= 0 else "red"
                pf_s = f"{pf:.2f}" if pf < 9999 else "inf"
                buf.write(
                    f"  {mn:<20} {trades:>7} {wr*100:>5.0f}% "
                    f"{pf_s:>6} [{c}]{pnl:>+9.2f}[/{c}] {dd*100:>6.1f}%\n"
                )
        else:
            buf.write("  [yellow]No backtest results yet. Run a backtest first.[/yellow]\n")

        return buf.getvalue()


# ─────────────────────────────────────────────────── entry point ──────────────

def main():
    parser = argparse.ArgumentParser(description="Project Challenger Controller")
    parser.add_argument("--no-autostart", action="store_true",
                        help="Don't auto-start the bot on launch")
    args = parser.parse_args()

    check_install()
    tui_auth()
    app = ChallengerApp()
    app._no_autostart = args.no_autostart
    app.run()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
