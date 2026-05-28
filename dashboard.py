"""
Project Challenger — Live Performance Dashboard
Run in a separate terminal: python dashboard.py
Refreshes every 2 seconds. Press Ctrl+C to exit.
"""
import time
import sys
import os

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box
    RICH = True
except ImportError:
    RICH = False

sys.path.insert(0, os.path.dirname(__file__))
from database import get_model_stats, get_latest_portfolio, get_all_model_names
from config import CONFIG

REFRESH_RATE = 2.0   # seconds
INITIAL_CAP  = CONFIG["PAPER_TRADING"]["INITIAL_CAPITAL"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profit_factor(gross_profit, gross_loss) -> str:
    gp = gross_profit or 0.0
    gl = gross_loss   or 0.0
    if gl == 0:
        return "∞" if gp > 0 else "—"
    return f"{gp / gl:.3f}"


def _win_rate(wins, losses) -> str:
    w = wins   or 0
    l = losses or 0
    total = w + l
    return f"{w / total * 100:.1f}%" if total > 0 else "—"


def _pnl_color(val: float) -> str:
    """Rich markup color for a PnL value."""
    if val > 0:  return "green"
    if val < 0:  return "red"
    return "white"


# ---------------------------------------------------------------------------
# Rich dashboard
# ---------------------------------------------------------------------------

def build_table(console: "Console") -> "Table":
    models = get_all_model_names()

    table = Table(
        title="[bold cyan]PROJECT CHALLENGER — Paper Trade Monitor[/bold cyan]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )

    table.add_column("Metric",         style="bold white",  no_wrap=True)
    for m in models or ["(no trades yet)"]:
        table.add_column(m, justify="right")

    rows_data = {
        "Trades":          [],
        "Win Rate":        [],
        "Profit Factor":   [],
        "Net PnL (USD)":   [],
        "Total Fees":      [],
        "Total Slippage":  [],
        "Capital":         [],
        "Position":        [],
        "Unrealized PnL":  [],
        "Return %":        [],
    }

    for m in (models or []):
        stats    = get_model_stats(m)
        port     = get_latest_portfolio(m)

        if stats and stats[0]:
            count, wins, losses, total_pnl, fees, slip, gp, gl = stats
            count     = count     or 0
            wins      = wins      or 0
            losses    = losses    or 0
            total_pnl = total_pnl or 0.0
            fees      = fees      or 0.0
            slip      = slip      or 0.0
            gp        = gp        or 0.0
            gl        = gl        or 0.0

            pnl_col = _pnl_color(total_pnl)

            rows_data["Trades"].append(str(count))
            rows_data["Win Rate"].append(_win_rate(wins, losses))
            rows_data["Profit Factor"].append(_profit_factor(gp, gl))
            rows_data["Net PnL (USD)"].append(
                f"[{pnl_col}]${total_pnl:+.4f}[/{pnl_col}]"
            )
            rows_data["Total Fees"].append(f"${fees:.4f}")
            rows_data["Total Slippage"].append(f"${slip:.4f}")
        else:
            for k in ["Trades","Win Rate","Profit Factor","Net PnL (USD)","Total Fees","Total Slippage"]:
                rows_data[k].append("—")

        if port:
            capital, pos_side, unreal, total_pnl_p = port
            ret_pct = (capital - INITIAL_CAP) / INITIAL_CAP * 100
            ret_col = _pnl_color(ret_pct)
            unreal_col = _pnl_color(unreal or 0.0)

            rows_data["Capital"].append(f"${capital:,.2f}")
            rows_data["Position"].append(pos_side or "FLAT")
            rows_data["Unrealized PnL"].append(
                f"[{unreal_col}]${(unreal or 0):+.4f}[/{unreal_col}]"
            )
            rows_data["Return %"].append(
                f"[{ret_col}]{ret_pct:+.2f}%[/{ret_col}]"
            )
        else:
            for k in ["Capital","Position","Unrealized PnL","Return %"]:
                rows_data[k].append("—")

    for metric, values in rows_data.items():
        table.add_row(metric, *values)

    return table


def run_rich_dashboard():
    console = Console()
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            try:
                tbl = build_table(console)
                ts  = time.strftime("%Y-%m-%d %H:%M:%S")
                live.update(Panel(tbl, subtitle=f"[dim]Updated: {ts}  |  Ctrl+C to exit[/dim]"))
            except Exception as e:
                live.update(f"[red]Dashboard error: {e}[/red]")
            time.sleep(REFRESH_RATE)


# ---------------------------------------------------------------------------
# Plain-text fallback (no rich installed)
# ---------------------------------------------------------------------------

def run_plain_dashboard():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 65)
        print("   PROJECT CHALLENGER — Live Monitor")
        print(f"   {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 65)

        models = get_all_model_names()
        if not models:
            print("  No trades logged yet. Waiting for data...")
        else:
            for m in models:
                stats = get_model_stats(m)
                port  = get_latest_portfolio(m)
                print(f"\n  [{m}]")
                if stats and stats[0]:
                    count, wins, losses, total_pnl, fees, slip, gp, gl = stats
                    pf = _profit_factor(gp or 0, gl or 0)
                    wr = _win_rate(wins or 0, losses or 0)
                    print(f"    Trades        : {count or 0}")
                    print(f"    Win Rate      : {wr}")
                    print(f"    Profit Factor : {pf}")
                    print(f"    Net PnL       : ${(total_pnl or 0):+.4f}")
                    print(f"    Fees          : ${(fees or 0):.4f}")
                if port:
                    capital, pos_side, unreal, _ = port
                    ret = (capital - INITIAL_CAP) / INITIAL_CAP * 100
                    print(f"    Capital       : ${capital:,.2f}  ({ret:+.2f}%)")
                    print(f"    Position      : {pos_side or 'FLAT'}")
                    print(f"    Unrealized    : ${(unreal or 0):+.4f}")
                print("  " + "-" * 50)

        print("\n  Ctrl+C to exit")
        time.sleep(REFRESH_RATE)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        if RICH:
            run_rich_dashboard()
        else:
            run_plain_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard closed.")
