# Project Challenger — Cyber HUD Facelift

A complete visual rebuild of the trading-bot dashboard.
Same FastAPI backend, same `/api/*` contract, same element IDs the JS
targets — everything you'd want to keep working still works. Just dressed
up in a cyberpunk-terminal HUD with tunable animations.

---

## 1. Files in this project

| Path                          | What it is                                                            |
|-------------------------------|-----------------------------------------------------------------------|
| `web/index.html`              | **Production drop-in** — full wired replacement for your dashboard.   |
| `Project Challenger.html`     | **Standalone mockup** — pure visual demo with hard-coded data. Useful for screenshots / showing people; not wired to the backend. |
| `CHANGES.md`                  | This document.                                                        |

The production file is at `web/index.html` so it mirrors the path in your
repo (`project-challenger-web/web/index.html`). Copy it into the same
location to deploy.

---

## 2. Deploying

```bash
# from your repo root
cp web/index.html web/index.html.bak                 # back up the original
cp /path/to/this/project/web/index.html web/        # drop the new one in
# restart your bot / reload the dashboard
```

That's it — no backend changes, no new dependencies, no build step.
FastAPI was already serving `web/index.html` as the SPA; it'll just serve
the new HTML instead.

To roll back: `mv web/index.html.bak web/index.html`.

External fetches the file makes:
- Google Fonts: `JetBrains Mono`, `Space Grotesk`, `Major Mono Display`.
  If your deploy is air-gapped, download the WOFF2 files and self-host
  them, or remove the `<link href="...fonts...">` block and the
  `var(--mono)` / `var(--ui)` / `var(--display)` references will fall back
  to system monospace and the page still works (just plainer typography).

---

## 3. What was changed

### 3.1 Visual layer (entirely new)

- **Backdrop**
  - Animated **matrix rain** on a `<canvas id="fx-rain">` (1’s, 0’s, and
    glyphs falling in your accent color, picks up palette changes).
  - **Faint grid** with slow drift (`bg-grid`).
  - **Radial bloom** behind it (`bg-bloom`) using the accent colors.
  - **CRT scanlines** overlay (`scanlines`).
  - **Vignette** to ground the corners.

- **Ticker bar** at the very top — scrolling coin prices in monospace,
  with red/green up/down chevrons. Loops seamlessly. Decorative
  (synthetic data — see §6 if you want to wire it to a real feed).

- **Header**
  - Three-ring **orbital wordmark** spinning at different speeds.
  - **Glitch overlay** on the wordmark (offset magenta clip-path slicing
    every few seconds — toggle off in FX panel).
  - **LED price tile** with a pulsing diode indicator.
  - Mode (`LIVE`/`PAPER`) and exchange badges keep their original meaning
    but get neon styling.
  - **Bot toggle button** — the existing `#ss-btn` keeps its `.btn pri` /
    `.btn danger` classes (toggled by the JS), it just looks fancy now.
  - **Logout button** is now a single glyph `⎋` to save space.

- **Tabs**
  - HUD-numbered (`01·LIVE` through `07·SETTINGS`).
  - Active tab has a neon underline with a pulsing dot indicator.
  - A meta strip on the right shows live UTC time and uptime.

- **Live tab**
  - **Command strip** with a 3-arc orbital "ARMED/IDLE" indicator + a
    rotating status-line typer + an E-STOP button that pulses red.
  - **Model P&L cards** (Ridge / XGB / LSTM) — each in its own accent
    color (mint / cyan / magenta), with:
    - rotating conic-gradient "halo" in the background
    - orbital ring around the model letter badge
    - giant LCD-style P&L number that turns red on negative
    - 3-stat footer (TRADES / WIN% / RET)
  - **Worker swarm** — same `live-workers` container the JS injects into,
    but the rows are now glowing rows with colored diodes.
  - **Recent log** — terminal panel with traffic-light dots, fake path
    `~/challenger/logs/live`, tail-style header. Same `live-log` element
    your JS appends to, just styled as a CRT terminal.
  - **Open Positions** and **Recent Trades** tables get the
    monospaced/cyber styling.

- **Backtest tab**
  - Preset buttons + form, same as before.
  - Results table with cyber styling.
  - The "Results by Coin" rendering (`renderBTByCoin`) was updated to
    match the new look but keeps the same DOM shape.

- **Training tab**
  - **Big orbital training rig** (the JS-driven `.orb` element) at the
    top of the progress section.
  - Three per-model rigs (`mring` + spinning `ms` ring + colored letter)
    — these are exactly the elements your `updateTrainingProgress` JS
    expects (`tr-spin-SKLEARN`, `tr-badge-SKLEARN`, `tr-bar-SKLEARN`,
    `tr-detail-SKLEARN`, etc.). The behavior is identical; the visual is
    new.
  - **CPU/RAM/ETA chips** under the big progress bar — also driven by
    your existing JS (`tr-cpu`, `tr-ram`, `tr-elapsed`, `tr-eta`).
  - Coin tier matrix, hyperparameter grids, status panel, training log
    — all preserved, all restyled.

- **Models tab** — CV summary + 3 archive tables, restyled.

- **Activity tab** — Component matrix grid + full activity terminal,
  same JS hooks.

- **Exchange & Keys tab** — Exchange selector, API key form, connection
  status, balance table, all restyled with the corner-bracket HUD frame.

- **Settings tab** — All inputs preserved, restyled.

- **Auth overlay** — Setup + login panels restyled in a glowing
  HUD-frame card, with HUD `▸` markers.

- **Footer** — Your original credits/links are preserved (Rev. J. Money
  credit, the SoundCloud + revjmoney.com links), just restyled.

### 3.2 Visual chrome — design system

Set of CSS custom properties on `:root` you can tweak per-environment if
you ever want to change the look manually:

| Token        | Meaning                                                  |
|--------------|----------------------------------------------------------|
| `--mint`     | Primary accent (default: `#00ffa3`)                      |
| `--cyan`     | Secondary accent                                         |
| `--mag`      | Tertiary accent                                          |
| `--amber`    | Warning color                                            |
| `--red`      | Danger / negative color                                  |
| `--void`     | Page background `#03060a`                                |
| `--surface`  | Card surface                                             |
| `--hair`     | 1px hairline border color                                |
| `--text`     | Foreground text                                          |
| `--mono`     | Monospace font stack                                     |
| `--display`  | Display font (Major Mono Display)                        |
| `--spin-mult`| Animation speed multiplier (1 = normal, 0 = paused)      |

Legacy aliases (so the original JS classnames keep working without
changes):

| Alias        | Maps to       |
|--------------|---------------|
| `--bg`       | `--void`      |
| `--bg2`      | `--surface`   |
| `--bg3`      | `--surface-2` |
| `--border`   | `--hair`      |
| `--accent`   | `--mint`      |
| `--green`    | `--mint`      |
| `--blue`     | `--cyan`      |

### 3.3 Animation/effects layer (new)

All animations are controlled by:
1. The CSS custom property `--spin-mult` (applied to every `animation`
   duration), and
2. A set of `body.fx-off-*` classes that toggle individual effects.

Specifically:

| Body class       | Effect when present                            |
|------------------|------------------------------------------------|
| `fx-off-scan`    | Hides the CRT scanline overlay                 |
| `fx-off-grid`    | Hides the grid backdrop                        |
| `fx-off-glitch`  | Hides the magenta glitch slice on the wordmark |
| `fx-off-ticker`  | Stops the top ticker animation                 |
| `fx-off-spin`    | Pauses all animations + hides matrix rain      |

The matrix rain is its own `<canvas>` and reads its opacity from inline
style (set by the FX layer).

---

## 4. The FX / Tweaks panel

Floating **⚙ button** in the bottom-right corner — click it to open the
FX panel.

Settings, persisted to `localStorage` under the key `challenger.fx`:

| Setting          | Values                                                      |
|------------------|-------------------------------------------------------------|
| `palette`        | `0`–`5` (Greenroom / Bubblegum / Citrus / Iceblue / Inferno / Mono) |
| `rain`           | `0`, `0.15`, `0.35`, `0.7` (opacity)                        |
| `spinSpeed`      | `0` (paused), `0.5`, `1`, `2`                               |
| `scanlines`      | `"on"` / `"off"`                                            |
| `grid`           | `"on"` / `"off"`                                            |
| `glitch`         | `"on"` / `"off"`                                            |
| `ticker`         | `"on"` / `"off"`                                            |

The **Reset to Defaults** button restores `palette=0, rain=0.35,
spinSpeed=1, all toggles on`.

Each user/browser gets their own settings. Nothing is sent to the server
— purely a client-side preference.

If you ever want to **force a default palette for new users**, edit the
`FX_DEFAULTS` object near the top of the FX script block in
`web/index.html`.

---

## 5. JS contract preserved (no API changes)

The new `web/index.html` calls every original endpoint with the exact
same payloads:

- `GET /api/auth/status`, `POST /api/auth/setup`, `POST /api/auth/login`,
  `POST /api/auth/logout`
- `GET /api/status`
- `GET /api/stream` (Server-Sent Events) — same fields read:
  `bot_running, price, components, new_log, retrain (running/done),
  cpu_pct, ram_pct, train_elapsed_s`
- `GET /api/metrics`
- `GET /api/trades?limit=50&model=...`
- `GET /api/activity` (boot history)
- `GET /api/archive`, `GET /api/cv`, `GET /api/training`
- `GET /api/settings`, `POST /api/settings`
- `GET /api/coins`, `POST /api/coins`, `POST /api/coins/refresh`
- `GET /api/accounts`
- `POST /api/keys`, `DELETE /api/keys`
- `POST /api/bot/start`, `POST /api/bot/stop`, `POST /api/bot/estop`,
  `POST /api/bot/retrain`, `POST /api/bot/refresh_data`,
  `POST /api/bot/reset_db`
- `POST /api/backtest/run`, `GET /api/backtest/status`

Every DOM element ID the original JS reads or writes is intact:

- Header / state: `bot-dot`, `ss-btn`, `price-display`, `candle-count`,
  `exch-badge`, `mode-badge`, `health-dot`, `train-banner`
- Model cards: `mc-SKLEARN_LINEAR`, `mc-XGBOOST_TREE`,
  `mc-PYTORCH_LSTM`; for each model: `pnl-*`, `tr-*`, `wr-*`, `rt-*`
- Live: `live-workers`, `live-log`, `live-positions`, `live-trades`,
  `trades-filter`
- Training: `tr-progress-section`, `tr-stage`, `tr-pct`, `tr-fill`,
  `tr-cpu`, `tr-ram`, `tr-elapsed`, `tr-eta`, `tr-spin-SKLEARN/
  XGBOOST/PYTORCH`, `tr-badge-*`, `tr-bar-*`, `tr-detail-*`,
  `tr-status-grid`, `tr-cv-container`, `tr-log`, `tr-status-lbl`,
  hyperparam fields `t-lookback`, `t-splits`, `t-sk/xg/pt`,
  `t-ridge-alpha`, `t-xgb-n/depth/lr/sub/col`, `t-seq`, `t-hidden`,
  `t-layers`, `t-dropout`, `t-epochs`, `t-batch`, `t-lr`
- Coins: `coin-tier`, `coin-trading`, `coin-tier-display`,
  `coin-avail-info`, `bt-coin-checks`, `bt-coins-custom`
- Backtest: `bt-lkbk`, `bt-thr`, `bt-cap`, `bt-coins`, `bt-btn`,
  `bt-spin`, `bt-results`, `bt-bycoin`
- Models: `cv-container`, `arch-SKLEARN_LINEAR`, `arch-XGBOOST_TREE`,
  `arch-PYTORCH_LSTM`, `retrain-lbl`
- Activity: `act-components`, `act-log`
- Exchange: `exch-sel`, `key-in`, `sec-in`, `key-status`, `acct-card`,
  `acct-body`
- Settings: `s-sig`, `s-pos`, `s-ch`, `s-crf`, `s-tld`, `s-spl`, `s-btl`
- Auth: `auth-overlay`, `auth-setup-panel`, `auth-login-panel`,
  `setup-user`/`-pass`/`-pass2`/`-err`, `login-user`/`-pass`/`-err`
- Toast: `toast`

All `onclick="..."` handlers from your original HTML still reference the
same global functions (`tab`, `doSetup`, `doLogin`, `doLogout`,
`toggleBot`, `confirmEstop`, `applyBtPreset`, `runBacktest`,
`saveTrainingSettings`, `saveAndRetrain`, `refreshCoinList`,
`saveCoinSettings`, `updateTierDisplay`, `loadTrades`, `loadAccounts`,
`clearTrainLog`, `loadArchive`, `loadCV`, `confirmRetrain`, `clearLog`,
`saveExchange`, `saveKeys`, `clearKeys`, `saveSettings`).

The JS bodies of all those functions were copied **verbatim** from your
original file — no logic changes. The only diffs in the JS section are:

1. Tiny tweak to `applyBotState` to also update the orbital "ARMED/IDLE"
   label (`#orb-label`).
2. Tiny tweak to `bootApp` to populate the fake session ID `#fx-session`
   on the command strip.
3. Toast styling uses class names `show ok|err|info` instead of inline
   styles. No behavior change.

---

## 6. Optional next steps (NOT done — would need backend tweaks)

These would be nice-to-haves but are out of scope of a pure facelift:

1. **Wire the top ticker to a real feed.** Currently the ticker is
   decorative (synthetic random deltas). If you expose a `GET
   /api/ticker` returning a list of `{symbol, price, change_pct}`
   objects, swap the `(function ticker() {...})()` IIFE at the bottom of
   `web/index.html` to fetch that and render it. ~15 lines of code.

2. **Wire the BTC price flicker animation.** The header price already
   updates via `applyBotState(running, price)` from your SSE stream —
   so this is already real. The only "fake" bit is that it shows `$—`
   until the first SSE message arrives.

3. **Sound effects** — keystroke clicks, alarm beeps on E-Stop, etc.
   Easy to add with a small `<audio>` pool. Say the word.

4. **Real glitch on errors** — currently the glitch text effect is
   purely cosmetic. Could trigger it whenever a worker reports an
   `error` status, as a visual alarm.

5. **Sparklines on model cards** — small inline SVG of last N P&L
   points per model. Would need `/api/metrics` to return a history
   array.

---

## 7. Troubleshooting

**Q: Everything looks plain — no neon, no animations.**
A: Google Fonts didn't load (offline / blocked / first-load slow).
Either wait for the fonts to fetch, self-host them, or remove the
`<link href="...fonts...">` block to fall back to system fonts.

**Q: Matrix rain is using a ton of CPU on a low-end machine.**
A: Open the FX panel (⚙ bottom-right), set **Matrix Rain → OFF** or
**LOW**, and/or set **Spin Speed → PAUSED**. Saved to localStorage.

**Q: I want to permanently kill the rain for all users.**
A: In `web/index.html`, find `const FX_DEFAULTS = {...}` and change
`rain: 0.35` to `rain: 0`.

**Q: Where are the original `tag-active` / `tag-armed` styles?**
A: Preserved — `loadArchive()` injects them via JS, the CSS for both is
near the bottom of the style block. Mint glow for active, red for armed.

**Q: The Tweaks panel doesn't open.**
A: Make sure you clicked the ⚙ button in the bottom-right of the page,
not the keyboard cog. The button is positioned `fixed` at
`right:18px;bottom:20px;z-index:51`. If you have a browser extension
covering that corner, it might be the culprit.

**Q: I want to change the wordmark from `PROJECT·CHALLENGER` to
something else.**
A: Search for `PROJECT·CHALLENGER` in `web/index.html` — it appears
twice in the header (the visible text and the glitch overlay). Change
both. The HUD style works best with short, all-caps, dot-separated
strings.

**Q: How do I tell which version of the file is running?**
A: Look at the page title in the browser tab — the new one says
`Project Challenger` (same as before, but if you ever inspect element on
the wordmark, the structure is `.wordmark > .l1 + .l2` with the version
string `v3.0 ▸ ORBITAL·TRADING·MESH`). To version-bump: search the file
for `v3.0` and update.

---

## 8. File anatomy (`web/index.html`, top to bottom)

```
<head>
  <style>                 ← CSS theme (~700 lines)
    :root { ... }         ← Color tokens, fonts, spacing
    backdrop layers       ← #fx-rain, .bg-grid, .bg-bloom, .scanlines
    .ticker, header.hdr   ← Header + ticker styling
    nav.tabs              ← Tab nav
    .card, .corner-*      ← HUD corner-bracketed card frames
    .cmdstrip, .orb-*     ← Command strip + orbital rings
    .model-card           ← Model P&L cards
    .wgrid, .wrow         ← Worker swarm rows
    .terminal, .log       ← CRT terminal styling
    .dtbl                 ← Data tables
    .progress-track       ← Progress bars
    .orb, .mring          ← Training rig orbitals (legacy JS classes)
    .stat-chip, .pbadge   ← Stat chips + progress badges
    .fg, .sw, .sl         ← Form inputs + toggle switches
    #toast                ← Toast notifications
    #auth-overlay         ← Auth modal
    #tweaks, .fx-cog      ← FX/Tweaks panel
    @media (max-width...) ← Mobile-ish breakpoints
</head>

<body>
  <canvas id="fx-rain">   ← Matrix rain canvas
  <div class="bg-*">      ← Backdrop layers (grid, bloom, scanlines, vignette)
  <div id="toast">        ← Toast container
  <div id="auth-overlay"> ← Setup + login panels

  <div class="app">
    <div class="ticker">       ← Top ticker
    <header class="hdr">       ← Wordmark, price LED, mode/exch, controls
    <nav class="tabs">         ← Tab buttons
    <main>
      <div id="tp-live">       ← Live tab content
      <div id="tp-backtest">   ← Backtest tab content
      <div id="tp-training">   ← Training tab content
      <div id="tp-models">     ← Models tab content
      <div id="tp-activity">   ← Activity tab content
      <div id="tp-exchange">   ← Exchange & Keys tab content
      <div id="tp-settings">   ← Settings tab content
    </main>
    <footer>                   ← Credits
  </div>

  <div class="fx-cog">       ← Floating settings cog (bottom-right)
  <div id="tweaks">          ← FX panel (hidden by default)

  <script>                   ← ORIGINAL APP JS (preserved verbatim)
    initAuth/doSetup/doLogin/doLogout
    tab, call, toast, fmt, fmtP, fmtPx, sColor, esc, scrollLogs
    startSSE, applyBotState, toggleBot
    loadMetrics, loadTrades
    applyComponents
    appendLog, renderLog, clearLog
    BT_PRESETS, applyBtPreset, runBacktest, renderBT
    loadArchive, loadCV, confirmRetrain
    loadExchangeStatus, loadAccounts, saveExchange, saveKeys, clearKeys
    loadSettings, saveSettings
    Training tab: trainLogLines, isTrainLine, appendTrainLog,
      renderTrainLog, clearTrainLog, loadTrainingStatus,
      renderCVTable, loadTrainingSettings, _collectTrainingBody,
      saveTrainingSettings, saveAndRetrain, _showTrainingComplete,
      fmtDuration, updateTrainingProgress
    Coin Management: TIER_DEFAULTS, loadCoins, updateTierDisplay,
      _renderBtCoinChecks, saveCoinSettings, refreshCoinList,
      renderBTByCoin
    bootApp, init
    confirmEstop
  </script>

  <script>                   ← FX LAYER (new, ~150 lines)
    FX_DEFAULTS, FX_PALETTES, FX (loaded from localStorage)
    applyPalette, applyFX
    Tweaks panel wiring (swatch + segment buttons)
    Matrix rain IIFE
    Ticker IIFE
    Status typer IIFE
    Clock/uptime IIFE
    applyFX() boot call
  </script>
</body>
```

---

## 9. Credits preserved

The original footer still shows:

> **PROJECT·CHALLENGER** · Orchestrated by **Rev. J. Money**
> [revjmoney.com](https://revjmoney.com) · [♫ SoundCloud](https://soundcloud.com/revjmoney)
> © 2026 Rev. J. Money. All rights reserved.

— styled to match the new look but otherwise unchanged.

---

If you want me to extend any of this — wire the ticker to a real feed,
add sound effects, build alternate themes, or split the FX script into a
separate `web/static/fx.js` so it caches separately — just ask.
