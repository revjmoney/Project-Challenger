"""
Project Challenger — Model Comparison Report
Run: python compare_models.py

Prints a side-by-side comparison of:
  1. Cross-validation metrics from training (all models, same folds)
  2. Live paper-trading performance from the database

Because all models share identical CV fold indices (data/cv_folds.pkl),
every metric here is a true apples-to-apples comparison.
"""
import sys
import os
import time

try:
    from rich.console import Console
    from rich.table   import Table
    from rich         import box
    RICH = True
except ImportError:
    RICH = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database  import get_training_results, get_training_summary, get_model_stats, get_all_model_names
from config    import CONFIG

INITIAL_CAP = CONFIG["PAPER_TRADING"]["INITIAL_CAPITAL"]
MODELS      = ["SKLEARN_LINEAR", "XGBOOST_TREE", "PYTORCH_LSTM"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _pf(gross_profit, gross_loss):
    gp = gross_profit or 0.0
    gl = gross_loss   or 0.0
    if gl == 0:
        return "inf" if gp > 0 else "—"
    return f"{gp / gl:.3f}"


def _wr(wins, losses):
    w, l = wins or 0, losses or 0
    t = w + l
    return f"{w / t * 100:.1f}%" if t > 0 else "—"


def _sign(val):
    if val is None:
        return "—"
    return f"+{val:.4f}" if val >= 0 else f"{val:.4f}"


# ---------------------------------------------------------------------------
# Rich report
# ---------------------------------------------------------------------------

def _cv_table(console):
    summary = get_training_summary()
    details = get_training_results()

    if not summary:
        console.print("[yellow]No training results found. Run 'python main.py' first.[/yellow]")
        return

    # ---- Summary across folds ----
    tbl = Table(
        title="[bold cyan]CV Training Results  (shared fold indices — all models evaluated on same timestamps)[/bold cyan]",
        box=box.ROUNDED, show_header=True, header_style="bold magenta",
    )
    tbl.add_column("Model",         style="bold white", no_wrap=True)
    tbl.add_column("Folds",         justify="center")
    tbl.add_column("Avg MSE",       justify="right")
    tbl.add_column("Avg MAE",       justify="right")
    tbl.add_column("Dir Accuracy",  justify="right")
    tbl.add_column("Last Trained",  justify="right", style="dim")

    best_mse     = min(r[2] for r in summary)
    best_dir_acc = max(r[4] for r in summary)

    for row in summary:
        model, folds, mse, mae, dir_acc, last = row
        mse_str  = f"[bold green]{mse:.8f}[/bold green]"   if mse == best_mse     else f"{mse:.8f}"
        dir_str  = f"[bold green]{dir_acc:.3f}[/bold green]" if dir_acc == best_dir_acc else f"{dir_acc:.3f}"
        last_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(last)) if last else "—"
        tbl.add_row(model, str(folds), mse_str, f"{mae:.8f}", dir_str, last_str)

    console.print(tbl)

    # ---- Per-fold breakdown ----
    tbl2 = Table(
        title="[bold]Per-Fold Breakdown[/bold]",
        box=box.SIMPLE, header_style="bold blue",
    )
    tbl2.add_column("Model",     style="dim white", no_wrap=True)
    tbl2.add_column("Fold",      justify="center")
    tbl2.add_column("Train",     justify="right")
    tbl2.add_column("Test",      justify="right")
    tbl2.add_column("MSE",       justify="right")
    tbl2.add_column("MAE",       justify="right")
    tbl2.add_column("DirAcc",    justify="right")

    for row in details:
        model, fold, tr, te, mse, mae, da, _ = row
        tbl2.add_row(model, str(fold), f"{tr:,}", f"{te:,}",
                     f"{mse:.8f}", f"{mae:.6f}", f"{da:.3f}")

    console.print(tbl2)


def _live_table(console):
    models = get_all_model_names()
    if not models:
        console.print("[yellow]No paper trading data yet. Run 'python main.py' to start live trading.[/yellow]\n")
        return

    tbl = Table(
        title="[bold cyan]Live Paper-Trading Performance[/bold cyan]",
        box=box.ROUNDED, show_header=True, header_style="bold magenta",
    )
    tbl.add_column("Model",          style="bold white", no_wrap=True)
    tbl.add_column("Trades",         justify="right")
    tbl.add_column("Win Rate",       justify="right")
    tbl.add_column("Profit Factor",  justify="right")
    tbl.add_column("Net PnL (USD)",  justify="right")
    tbl.add_column("Return %",       justify="right")
    tbl.add_column("Fees Paid",      justify="right")

    rows_data = []
    for m in models:
        s = get_model_stats(m)
        if not s or not s[0]:
            continue
        count, wins, losses, pnl, fees, slip, gp, gl = s
        pnl    = pnl   or 0.0
        fees   = fees  or 0.0
        cap    = INITIAL_CAP + pnl
        ret    = (cap - INITIAL_CAP) / INITIAL_CAP * 100
        rows_data.append((m, count or 0, wins or 0, losses or 0, pnl, fees, gp or 0, gl or 0, ret))

    if not rows_data:
        console.print("[yellow]No closed trades yet.[/yellow]\n")
        return

    best_pnl = max(r[4] for r in rows_data)
    for m, count, wins, losses, pnl, fees, gp, gl, ret in rows_data:
        pnl_col = "green" if pnl >= 0 else "red"
        ret_col = "green" if ret >= 0 else "red"
        pnl_str = f"[{pnl_col}]{_sign(pnl)}[/{pnl_col}]"
        if pnl == best_pnl:
            pnl_str = f"[bold {pnl_col}]{_sign(pnl)} *[/bold {pnl_col}]"
        ret_str = f"[{ret_col}]{ret:+.2f}%[/{ret_col}]"
        tbl.add_row(
            m, str(count), _wr(wins, losses), _pf(gp, gl),
            pnl_str, ret_str, f"${fees:.4f}",
        )
    console.print(tbl)
    console.print("[dim]  * = best performing model[/dim]\n")


def _verdict(console):
    summary = get_training_summary()
    models  = get_all_model_names()
    if not summary and not models:
        return

    console.print("[bold underline]Scientific Verdict[/bold underline]\n")

    if summary:
        best_cv = min(summary, key=lambda r: r[2])   # lowest avg MSE
        console.print(f"  Best CV accuracy (lowest MSE) : [bold green]{best_cv[0]}[/bold green]  "
                      f"(MSE={best_cv[2]:.8f}, DirAcc={best_cv[4]:.3f})")

    live_rows = [(m, get_model_stats(m)) for m in models]
    live_pnl  = [(m, (s[3] or 0)) for m, s in live_rows if s and s[0]]
    if live_pnl:
        best_live = max(live_pnl, key=lambda x: x[1])
        console.print(f"  Best live paper-trade PnL     : [bold green]{best_live[0]}[/bold green]  "
                      f"(PnL=${best_live[1]:+.4f})")

    console.print()


# ---------------------------------------------------------------------------
# Plain-text fallback
# ---------------------------------------------------------------------------

def _plain_report():
    print("=" * 70)
    print("  PROJECT CHALLENGER -- MODEL COMPARISON REPORT")
    print("=" * 70)

    summary = get_training_summary()
    if summary:
        print("\n  CV TRAINING RESULTS (shared folds)\n")
        print(f"  {'Model':<20} {'Folds':>5} {'Avg MSE':>12} {'Avg MAE':>12} {'DirAcc':>9}")
        print("  " + "-" * 60)
        for row in summary:
            model, folds, mse, mae, da, _ = row
            print(f"  {model:<20} {folds:>5} {mse:>12.8f} {mae:>12.8f} {da:>9.3f}")

    print("\n  LIVE PAPER-TRADING RESULTS\n")
    for m in get_all_model_names():
        s = get_model_stats(m)
        if not s or not s[0]:
            continue
        count, wins, losses, pnl, fees, slip, gp, gl = s
        pnl = pnl or 0.0
        ret = pnl / INITIAL_CAP * 100
        print(f"  {m}")
        print(f"    Trades: {count or 0}  Win Rate: {_wr(wins or 0, losses or 0)}")
        print(f"    PF: {_pf(gp or 0, gl or 0)}  Net PnL: ${pnl:+.4f}  Return: {ret:+.2f}%")
        print()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if RICH:
        c = Console()
        c.print()
        c.print("[bold]PROJECT CHALLENGER[/bold] — Model Comparison Report", style="bold cyan")
        c.print(f"[dim]Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
        _cv_table(c)
        c.print()
        _live_table(c)
        _verdict(c)
    else:
        _plain_report()
