# -*- coding: utf-8 -*-
"""
Backtest Day (5m/15m) — MySQL streaming + EMA/ATR + Risk Mgmt
Entrées : cross / pullback VWAP / breakout
Filtres : ATR% / Volume / VWAP / Fenêtre RTH
Sizing : risk_frac * capital en $R, commission fixe par ordre
Sécurité : perte max journalière (en R), 1 trade/symbole/jour
WF : fenêtres glissantes (optionnel)
Telegram : rapport compact (PF, Sharpe, DD, TP/STOP/Trend, Fiabilité)
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
from sqlalchemy import create_engine, text

PRINT_PREFIX = "[INFO]"


# ------------------------------- Utils --------------------------------- #

def _log(msg: str):
    print(f"{PRINT_PREFIX} {msg}")

def _setup_console_encoding():
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

def load_config(config_path: str) -> dict:
    _log(f"Chargement de la configuration à partir de {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _mysql_engine_from_config(cfg: dict):
    # Priorité au bloc db.uri sinon database.mysql
    uri = None
    if "db" in cfg and isinstance(cfg["db"], dict):
        uri = cfg["db"].get("uri")
    if uri:
        _log("Connexion via URI SQLAlchemy (pool optimisé).")
        eng = create_engine(uri, pool_pre_ping=True, pool_recycle=1800)
        _log("Connexion DB (URI) OK.")
        return eng

    db = cfg.get("database", {}).get("mysql", {})
    host = db.get("host", "127.0.0.1")
    port = db.get("port", 3306)
    user = db.get("user", "root")
    pwd  = db.get("password", "")
    name = db.get("db", "trading_bot")
    charset = db.get("charset", "utf8mb4")
    uri = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{name}?charset={charset}"
    _log("Connexion via paramètres MySQL...")
    eng = create_engine(uri, pool_pre_ping=True, pool_recycle=1800)
    _log("Connexion DB (paramètres) OK.")
    return eng

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def _load_symbols(path: Optional[str], max_symbols: Optional[int], offset: int = 0) -> List[str]:
    syms: List[str] = []
    if path:
        p = Path(path)
        syms = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if offset < 0:
        offset = 0
    if max_symbols is not None and max_symbols > 0:
        syms = syms[offset: offset + max_symbols]
    else:
        syms = syms[offset:]
    _log(f"Fichier symboles: {path} — {len(syms)} symboles")
    return syms


# ----------------------------- SQL Helpers ----------------------------- #

def _fetch_bars_5m_stream(
    eng,
    table_5m: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    batch_syms: int = 300,
) -> pd.DataFrame:
    """Stream par paquets de symboles pour limiter mémoire & accélérer l’IN()."""
    if not symbols:
        return pd.DataFrame(columns=["symbol","ts","open","high","low","close","volume"])
    rows_total = 0
    out = []
    with eng.begin() as conn:
        for i in range(0, len(symbols), batch_syms):
            chunk = symbols[i:i+batch_syms]
            placeholders = ",".join([f":s{k}" for k in range(len(chunk))])
            params = {f"s{k}": s for k, s in enumerate(chunk)}
            params.update({
                "start": start.strftime("%Y-%m-%d %H:%M:%S"),
                "end": (end + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            })
            sql = text(f"""
                SELECT symbol, ts, open, high, low, close, volume
                FROM {table_5m}
                WHERE symbol IN ({placeholders})
                  AND ts >= :start AND ts < :end
                ORDER BY symbol, ts
            """)
            df_part = pd.read_sql(sql, conn, params=params)
            out.append(df_part)
            rows_total += len(df_part)
            print(f"[RUN] batch {min(i+batch_syms, len(symbols))}/{len(symbols)} — rows={rows_total:,}")
    if not out:
        return pd.DataFrame(columns=["symbol","ts","open","high","low","close","volume"])
    df = pd.concat(out, ignore_index=True)
    # types
    df["ts"] = pd.to_datetime(df["ts"])
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)

def _query_regime_daily_ma(
    eng,
    table_5m: str,
    symbol: str,
    start: datetime,
    end: datetime,
    ma_len: int,
    mode: str = "close_above_ma",
) -> pd.DataFrame:
    """Construit un masque daily True/False selon un MA sur close 5m → daily."""
    with eng.begin() as conn:
        sql = text(f"""
            SELECT ts, close FROM {table_5m}
            WHERE symbol = :sym
              AND ts >= :start AND ts < :end
            ORDER BY ts
        """)
        df = pd.read_sql(
            sql, conn,
            params={"sym": symbol,
                    "start": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": (end + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")}
        )
    if df.empty:
        return pd.DataFrame(columns=["date","regime"])
    df["ts"] = pd.to_datetime(df["ts"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    # daily agg
    df["date"] = df["ts"].dt.date
    d = df.groupby("date", observed=False)["close"].last().to_frame()
    if d.empty:
        return pd.DataFrame(columns=["date","regime"])
    # moving average / ema slope
    if mode == "ema_slope_up":
        ema = d["close"].ewm(span=ma_len, adjust=False, min_periods=ma_len).mean()
        ema_lag = ema.shift(1)
        regime = (ema >= ema_lag)
    else:
        ma = d["close"].rolling(ma_len, min_periods=ma_len).mean()
        regime = (d["close"] > ma)
    out = pd.DataFrame({"date": d.index, "regime": regime.fillna(False).astype(bool)})
    return out


# -------------------------- Indicators & Filters ----------------------- #

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _vwap_day(df: pd.DataFrame) -> pd.Series:
    """VWAP intraday (reset par jour)."""
    g = df.copy()
    g["date"] = g["ts"].dt.date
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    g["tpv"] = tp * g["volume"]
    g["cum_vol"] = g.groupby("date", observed=False)["volume"].cumsum()
    g["cum_tpv"] = g.groupby("date", observed=False)["tpv"].cumsum()
    vwap = g["cum_tpv"] / g["cum_vol"].replace(0, np.nan)
    return vwap

def _time_filters(df: pd.DataFrame,
                  rth_only: bool,
                  skip_first_min: int,
                  skip_last_min: int,
                  midday_start: Optional[str],
                  midday_end: Optional[str]) -> pd.Series:
    """Mask True = tradable."""
    t = df["ts"].dt.tz_localize(None).dt.time
    # RTH 9:30-16:00 New York
    if rth_only:
        in_rth = (t >= time(9,30)) & (t < time(16,0))
    else:
        in_rth = pd.Series(True, index=df.index)

    df2 = df.copy()
    df2["date"] = df2["ts"].dt.date
    minutes_from_open = (df2["ts"].dt.hour * 60 + df2["ts"].dt.minute) - (9*60 + 30)
    minutes_to_close = (16*60) - (df2["ts"].dt.hour * 60 + df2["ts"].dt.minute)

    ok_first = minutes_from_open >= skip_first_min
    ok_last  = minutes_to_close >= skip_last_min

    mask = in_rth & ok_first & ok_last

    if midday_start and midday_end:
        try:
            hh1, mm1 = map(int, midday_start.split(":"))
            hh2, mm2 = map(int, midday_end.split(":"))
            in_midday = (t >= time(hh1, mm1)) & (t < time(hh2, mm2))
            mask = mask & (~in_midday)
        except Exception:
            pass
    return mask


# --------------------------- Strategy Runner --------------------------- #

@dataclass
class StratParams:
    ema_fast: int
    ema_slow: int
    atr_len: int
    stop_mode: str
    atr_mult: float
    stop_pct: float
    tp_mult: float
    cooldown: int
    max_hold: int
    slope_lag: int
    fee_bps: int
    commission_fixed: float
    side_mode: str
    entry_mode: str
    breakout_n: int
    pullback_tol: float
    require_vwap_reclaim: bool
    min_atr_pct: float
    max_atr_pct: float
    vol_window: int
    vol_spike_mult: float
    min_vol_5m: int
    rth_only: bool
    skip_first_min: int
    skip_last_min: int
    midday_start: Optional[str]
    midday_end: Optional[str]
    breakeven_after_r: float
    trail_after_r: float
    trail_mult_r: float
    max_trades_per_symbol_day: int
    max_loss_r_per_symbol_day: float
    max_loss_r_total_day: float
    tf: str
    mom_lkb: int
    mom_th_long: float
    mom_th_short: float
    mom_ref: str
    cap: float
    risk_frac: float
    qty_fixed: int
    min_shares: int
    max_shares: int

def _smart_runner_ema_atr(
    bars: Dict[str, pd.DataFrame],
    regime_daily: Optional[pd.DataFrame],
    P: StratParams,
) -> Tuple[Dict[str, float], List[dict]]:
    """Core backtest par symbole. Retourne metrics + trade log."""
    ENTRY_TOL = 0.001  # 0.1% tolérance sur EMA_f / prix

    # Regime map (daily bool)
    regime_map = {}
    if regime_daily is not None and not regime_daily.empty:
        rd = regime_daily.copy()
        rd["date"] = pd.to_datetime(rd["date"]).dt.date
        regime_map = dict(zip(rd["date"], rd["regime"].astype(bool)))

    loss_r_total_by_day: Dict[object, float] = {}  # en R (dollars de R)
    pnl = wins = trades = 0
    tp_count = stop_count = trend_count = timeout_count = 0
    log: List[dict] = []

    for sym, df in bars.items():
        if df is None or df.empty:
            continue
        g = df.copy().sort_values("ts").reset_index(drop=True)

        # indicateurs
        g["ema_f"] = _ema(g["close"], P.ema_fast)
        g["ema_s"] = _ema(g["close"], P.ema_slow)
        g["ema_s_lag"] = g["ema_s"].shift(P.slope_lag)
        g["atr"] = _atr(g["high"], g["low"], g["close"], P.atr_len)
        g["atr_pct"] = (g["atr"] / g["close"]).clip(lower=0.0).fillna(0.0)
        g["vwap"] = _vwap_day(g)
        g["date"] = g["ts"].dt.date

        # volume filters
        vw = max(P.vol_window, 1)
        g["vol_mean"] = g["volume"].rolling(vw, min_periods=vw).mean()
        g["vol_ok"] = (g["volume"] >= (P.vol_spike_mult * g["vol_mean"]).fillna(0.0)) & (g["volume"] >= P.min_vol_5m)

        # time filters
        tradable = _time_filters(g, P.rth_only, P.skip_first_min, P.skip_last_min, P.midday_start, P.midday_end)

        pos = False
        entry = 0.0
        entry_ts = None
        stop_px = 0.0
        tp_px = 0.0
        hold = 0
        cooldown = 0
        highest_close = None
        pos_qty = 0.0  # quantité d’actions en position
        trades_today: Dict[object, int] = {}
        loss_r_sym_by_day: Dict[object, float] = {}

        for i in range(len(g) - 1):
            row, nxt = g.iloc[i], g.iloc[i+1]
            if not tradable.iloc[i]:
                continue
            dte = row["date"]

            # regime OFF -> on coupe (si en position) ou on saute l'entrée
            if regime_map and not regime_map.get(dte, True):
                if pos:
                    exit_px = float(nxt["open"])
                    # PnL avec quantité + fees une seule fois
                    gross = pos_qty * (exit_px - entry)
                    fees = _fee_total(P, entry, exit_px)
                    r = gross - fees
                    pnl += r
                    trades += 1
                    wins += 1 if r > 0 else 0
                    trend_count += 1
                    log.append({"symbol": sym, "entry_ts": entry_ts, "exit_ts": nxt["ts"],
                                "entry": entry, "exit": exit_px, "qty": pos_qty, "pnl": r, "reason": "REGIME_OFF"})
                    # MAJ pertes/jour en R$
                    R_per_share = abs(entry - stop_px)
                    R_dollars = max(pos_qty * max(R_per_share, 1e-9), 1e-9)
                    r_in_R = r / R_dollars
                    if r_in_R < 0:
                        loss_r_sym_by_day[dte] = loss_r_sym_by_day.get(dte, 0.0) + r_in_R
                        loss_r_total_by_day[dte] = loss_r_total_by_day.get(dte, 0.0) + r_in_R
                    pos = False
                    pos_qty = 0.0
                    cooldown = P.cooldown
                continue

            emaf = float(row["ema_f"])
            emas = float(row["ema_s"])
            emas_lag = float(row["ema_s_lag"]) if not math.isnan(row["ema_s_lag"]) else np.nan
            close = float(row["close"])
            high = float(row["high"])
            low  = float(row["low"])
            atr  = float(row["atr"]) if not math.isnan(row["atr"]) else 0.0
            atr_pct = float(row["atr_pct"])
            vwap_val = float(row["vwap"]) if not np.isnan(row["vwap"]) else np.nan

            # filtres statiques
            if not (P.min_atr_pct <= atr_pct <= P.max_atr_pct):
                continue
            if not bool(row["vol_ok"]):
                continue
            if P.entry_mode in ("pullback", "breakout") and P.require_vwap_reclaim:
                if np.isnan(vwap_val):
                    continue

            slope_ok = (not math.isnan(emas)) and (not math.isnan(emas_lag)) and (emas >= emas_lag)

            if cooldown > 0:
                cooldown -= 1

            # ---------------- Entrées ---------------- #
            if not pos:
                # arrêt nouvelles entrées si perte journalière max atteinte (en R$ cumulés)
                if loss_r_total_by_day.get(dte, 0.0) <= -P.max_loss_r_total_day:
                    continue
                if trades_today.get(dte, 0) >= P.max_trades_per_symbol_day:
                    continue

                long_signal = False
                short_signal = False  # (short non implémenté dans ce runner)

                if P.entry_mode == "cross":
                    # cross EMA_f / EMA_s avec tolérance
                    prev = g.iloc[i-1] if i >= 1 else None
                    cross_up = False
                    if prev is not None:
                        prev_emaf = float(prev["ema_f"]); prev_emas = float(prev["ema_s"])
                        cross_up = (prev_emaf <= prev_emas) and (emaf >= emas) \
                                   and (close >= emaf * (1 - ENTRY_TOL)) and slope_ok
                    long_signal = cross_up

                elif P.entry_mode == "pullback":
                    # pullback vers EMA_f + reclaim VWAP optionnel
                    near_ema = (close >= emaf * (1 - P.pullback_tol)) and (close <= emaf * (1 + P.pullback_tol))
                    reclaim_vwap = True
                    if P.require_vwap_reclaim:
                        reclaim_vwap = (not np.isnan(vwap_val)) and (close >= vwap_val) and (emaf >= vwap_val)
                    long_signal = near_ema and slope_ok and reclaim_vwap

                elif P.entry_mode == "breakout":
                    n = max(P.breakout_n, 2)
                    if i >= n:
                        hh = g["high"].iloc[i-n+1:i+1].max()
                        long_signal = (close >= hh * (1 - ENTRY_TOL)) and slope_ok

                # side filters (on gère long-only pour ce runner)
                ok_side = (P.side_mode in ("long","both")) and long_signal

                if ok_side and cooldown == 0:
                    entry = float(nxt["open"])
                    entry_ts = nxt["ts"]

                    # calcule R par action (per-share)
                    if P.stop_mode == "atr":
                        R_per_share = P.atr_mult * max(atr, 1e-9)
                        stop_px = entry - R_per_share
                    else:
                        R_per_share = P.stop_pct * entry
                        stop_px = entry - R_per_share

                    tp_px = entry + (P.tp_mult * R_per_share)

                    # sizing (nombre d’actions)
                    if P.qty_fixed and P.qty_fixed > 0:
                        qty = float(P.qty_fixed)
                    else:
                        R_dollars = max(P.cap * P.risk_frac, 1.0)  # $R à risquer
                        qty = max(R_dollars / max(R_per_share, 1e-9), P.min_shares)
                        qty = min(qty, float(P.max_shares))

                    pos = True
                    pos_qty = qty
                    hold = 0
                    highest_close = close
                    # IMPORTANT : pas de déduction de commission à l'entrée (comptée à la sortie via _fee_total)
                    continue

            # ---------------- Gestion de position ---------------- #
            if pos:
                hold += 1
                if highest_close is None or close > highest_close:
                    highest_close = close

                R_per_share = abs(entry - stop_px)

                # break-even / trailing
                if (close - entry) >= P.breakeven_after_r * R_per_share:
                    stop_px = max(stop_px, entry)
                if (close - entry) >= P.trail_after_r * R_per_share and highest_close is not None:
                    trail_stop = highest_close - P.trail_mult_r * R_per_share
                    stop_px = max(stop_px, trail_stop)

                hit_stop = (low <= stop_px)
                hit_tp   = (high >= tp_px)

                exit_px = None
                reason = None
                if hit_stop:
                    exit_px = stop_px; reason = "STOP"
                elif hit_tp:
                    exit_px = tp_px; reason = "TP"
                else:
                    # tendance perdue (EMA_f sous EMA_s) ou hold trop long
                    trend_up = (emaf >= emas) and (close >= emaf * (1 - ENTRY_TOL))
                    if not trend_up:
                        exit_px = float(nxt["open"]); reason = "TREND_BROKE"
                    elif hold >= P.max_hold:
                        exit_px = float(nxt["open"]); reason = "TIMEOUT"

                if exit_px is not None:
                    # PnL en dollars avec quantité + fees (une seule fois ici)
                    gross = pos_qty * (exit_px - entry)
                    fees = _fee_total(P, entry, exit_px)
                    r = gross - fees
                    pnl += r
                    trades += 1
                    wins += 1 if r > 0 else 0
                    if reason == "TP": tp_count += 1
                    elif reason == "STOP": stop_count += 1
                    elif reason == "TREND_BROKE": trend_count += 1
                    elif reason == "TIMEOUT": timeout_count += 1

                    log.append({"symbol": sym, "entry_ts": entry_ts, "exit_ts": nxt["ts"],
                                "entry": entry, "exit": exit_px, "qty": pos_qty, "pnl": r, "reason": reason})

                    # MAJ pertes/jour en R$
                    R_dollars = max(pos_qty * max(R_per_share, 1e-9), 1e-9)
                    r_in_R = r / R_dollars
                    if r_in_R < 0:
                        loss_r_sym_by_day[dte] = loss_r_sym_by_day.get(dte, 0.0) + r_in_R
                        loss_r_total_by_day[dte] = loss_r_total_by_day.get(dte, 0.0) + r_in_R

                    pos = False
                    pos_qty = 0.0
                    cooldown = P.cooldown
                    trades_today[dte] = trades_today.get(dte, 0) + 1

        # clôture fin de données
        if pos:
            last = g.iloc[-1]
            exit_px = float(last["close"])
            # PnL en $ avec quantité + fees
            gross = pos_qty * (exit_px - entry)
            fees = _fee_total(P, entry, exit_px)
            r = gross - fees
            pnl += r
            trades += 1
            wins += 1 if r > 0 else 0
            trend_count += 1
            log.append({"symbol": sym, "entry_ts": entry_ts, "exit_ts": last["ts"],
                        "entry": entry, "exit": exit_px, "qty": pos_qty, "pnl": r, "reason": "EOD"})
            # MAJ pertes/jour en R$
            R_per_share = abs(entry - stop_px)
            R_dollars = max(pos_qty * max(R_per_share, 1e-9), 1e-9)
            r_in_R = r / R_dollars
            if r_in_R < 0:
                dte_last = last["ts"].date()
                loss_r_total_by_day[dte_last] = loss_r_total_by_day.get(dte_last, 0.0) + r_in_R

    wr = (wins / trades * 100.0) if trades else 0.0
    # Profit Factor
    if trades:
        pnl_pos = sum([max(0.0, t["pnl"]) for t in log])
        pnl_neg = abs(sum([min(0.0, t["pnl"]) for t in log]))
        pf = (pnl_pos / pnl_neg) if pnl_neg > 0 else (float("inf") if pnl_pos > 0 else 0.0)
    else:
        pf = 0.0
    # Sharpe approximé (par trade)
    rets = np.array([t["pnl"] for t in log], dtype=float) if log else np.array([])
    sharpe = ((rets.mean() / (rets.std()+1e-9)) * math.sqrt(max(len(rets),1))) if len(rets) else 0.0
    # drawdown approx (cumul trades)
    if len(rets):
        eq = np.cumsum(rets)
        peak = np.maximum.accumulate(eq)
        dd = float(np.max(peak - eq))
    else:
        dd = 0.0

    metrics = {
        "trades": trades,
        "winrate_pct": wr,
        "pf": pf,
        "sharpe": sharpe,
        "dd": dd,
        "pnl": float(np.sum(rets)) if len(rets) else 0.0,
        "tp": tp_count,
        "stop": stop_count,
        "trend": trend_count,
        "timeout": timeout_count,
    }
    return metrics, log


# ---------------------------- Fees & Sizing ---------------------------- #

def _fee_entry(P: StratParams, entry_px: float) -> float:
    return float(P.commission_fixed)

def _fee_exit(P: StratParams, exit_px: float) -> float:
    return float(P.commission_fixed)

def _fee_total(P: StratParams, entry_px: float, exit_px: float) -> float:
    # Option : ajouter fee_bps si besoin (ici, seulement commission fixe entrée + sortie)
    return _fee_entry(P, entry_px) + _fee_exit(P, exit_px)


# ------------------------------ Telegram ------------------------------- #

def _tg_send(token: str, chat_id: str, text: str):
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=12,
        )
        if resp.status_code != 200:
            _log(f"[TG] HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        _log(f"[TG] Exception: {e}")

def _fmt_money(x: float) -> str:
    s = f"{x:0.2f}"
    return f"${s}"

def _tg_report(
    title: str,
    start: datetime,
    end: datetime,
    n_symbols: int,
    tf: str,
    entry_mode: str,
    side_mode: str,
    m: Dict[str, float],
) -> str:
    # Fiabilité TP = TP / (TP+STOP)
    denom = (m.get("tp",0) + m.get("stop",0))
    reliab = (100.0 * m.get("tp",0) / denom) if denom else 0.0
    msg = (
        f"📊 {title}\n"
        f"🗓️ {start.date()} → {end.date()} • {n_symbols} symb • TF {tf.upper()} • {entry_mode}/{side_mode}\n"
        f"📦 Trades {m['trades']} • Win {m['winrate_pct']:.1f}% • PF {m['pf']:.2f} • Sharpe {m['sharpe']:.2f} • DD {m['dd']:.2f}\n"
        f"✅TP {m.get('tp',0)} • ❌STOP {m.get('stop',0)} • 🔁Trend {m.get('trend',0)} • ⏰Time {m.get('timeout',0)}\n"
        f"🧠 Fiabilité TP {reliab:.1f}% (sur TP+STOP)\n"
        f"💰 PnL: {_fmt_money(m['pnl'])}"
    )
    return msg


# ----------------------------- Main / CLI ------------------------------ #

def _build_args():
    p = argparse.ArgumentParser("bot_backtest")
    p.add_argument("--profile", choices=["defensif","normal","agressif"], default="defensif")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--symbols-file")
    p.add_argument("--exclude-symbols", default="")
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--symbols-offset", type=int, default=0)
    p.add_argument("--batch-syms", type=int, default=300)

    p.add_argument("--universe")         # réservé si tu veux brancher une table bot_universe
    p.add_argument("--uni-limit", type=int, default=None)

    p.add_argument("--regime-symbol", default="")
    p.add_argument("--regime-ma", type=int, default=50)
    p.add_argument("--regime-mode", choices=["close_above_ma","ema_slope_up"], default="close_above_ma")

    p.add_argument("--ema-fast", type=int, default=20)
    p.add_argument("--ema-slow", type=int, default=140)
    p.add_argument("--atr-len", type=int, default=14)
    p.add_argument("--stop-mode", choices=["pct","atr"], default="atr")
    p.add_argument("--atr-mult", type=float, default=2.6)
    p.add_argument("--stop-pct", type=float, default=0.015)
    p.add_argument("--tp-mult", type=float, default=1.9)
    p.add_argument("--cooldown", type=int, default=12)
    p.add_argument("--max-hold", type=int, default=48)
    p.add_argument("--slope-lag", type=int, default=30)
    p.add_argument("--fee-bps", type=int, default=0)

    p.add_argument("--max-trades-per-symbol-day", type=int, default=1)
    p.add_argument("--breakeven-after-r", type=float, default=1.0)
    p.add_argument("--trail-after-r", type=float, default=1.6)
    p.add_argument("--trail-mult-r", type=float, default=0.9)
    p.add_argument("--max-loss-r-per-symbol-day", type=float, default=2.0)
    p.add_argument("--max-loss-r-total-day", type=float, default=4.0)

    p.add_argument("--breakout-n", type=int, default=20)
    p.add_argument("--min-at_r-pct-deprecated", type=float, default=None)  # compat
    p.add_argument("--min-atr-pct", type=float, default=0.0)
    p.add_argument("--max-atr-pct", type=float, default=1.0)
    p.add_argument("--vol-spike-mult", type=float, default=1.0)
    p.add_argument("--vol-window", type=int, default=5)
    p.add_argument("--min-vol-5m", type=int, default=0)
    p.add_argument("--vwap-filter", action="store_true")

    p.add_argument("--entry-mode", choices=["breakout","pullback","cross"], default="cross")
    p.add_argument("--pullback-tol", type=float, default=0.005)
    p.add_argument("--require-vwap-reclaim", action="store_true")

    p.add_argument("--side-mode", choices=["long","short","both"], default="long")
    p.add_argument("--rth-only", action="store_true")
    p.add_argument("--skip-first-min", type=int, default=10)
    p.add_argument("--skip-last-min", type=int, default=240)
    p.add_argument("--midday-start", default=None)
    p.add_argument("--midday-end", default=None)
    p.add_argument("--tf", choices=["5m","15m"], default="15m")

    p.add_argument("--mom-lkb", type=int, default=0)
    p.add_argument("--mom-th-long", type=float, default=-1.0)
    p.add_argument("--mom-th-short", type=float, default=1.0)
    p.add_argument("--mom-ref", choices=["close","ema_f"], default="ema_f")

    wf = p.add_mutually_exclusive_group()
    wf.add_argument("--wf", action="store_true")
    wf.add_argument("--no-wf", action="store_true")

    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--embargo-days", type=int, default=5)

    p.add_argument("--commission-fixed", type=float, default=1.0)
    p.add_argument("--qty-fixed", type=int, default=0)
    p.add_argument("--cap", type=float, default=10000.0)
    p.add_argument("--risk-frac", type=float, default=0.006)
    p.add_argument("--min-shares", type=int, default=1)
    p.add_argument("--max-shares", type=int, default=400)

    p.add_argument("--env", choices=["PAPER","LIVE"], default="PAPER")
    p.add_argument("--notify", action="store_true")
    p.add_argument("--config", default="src/config/config.yaml")
    return p

def _build_params(args) -> StratParams:
    return StratParams(
        ema_fast=args.ema_fast, ema_slow=args.ema_slow, atr_len=args.atr_len,
        stop_mode=args.stop_mode, atr_mult=args.atr_mult, stop_pct=args.stop_pct,
        tp_mult=args.tp_mult, cooldown=args.cooldown, max_hold=args.max_hold,
        slope_lag=args.slope_lag, fee_bps=args.fee_bps, commission_fixed=args.commission_fixed,
        side_mode=args.side_mode, entry_mode=args.entry_mode,
        breakout_n=args.breakout_n, pullback_tol=args.pullback_tol,
        require_vwap_reclaim=(args.require_vwap_reclaim or args.vwap_filter),
        min_atr_pct=args.min_atr_pct, max_atr_pct=args.max_atr_pct,
        vol_window=args.vol_window, vol_spike_mult=args.vol_spike_mult, min_vol_5m=args.min_vol_5m,
        rth_only=args.rth_only, skip_first_min=args.skip_first_min, skip_last_min=args.skip_last_min,
        midday_start=args.midday_start, midday_end=args.midday_end,
        breakeven_after_r=args.breakeven_after_r, trail_after_r=args.trail_after_r, trail_mult_r=args.trail_mult_r,
        max_trades_per_symbol_day=args.max_trades_per_symbol_day,
        max_loss_r_per_symbol_day=args.max_loss_r_per_symbol_day,
        max_loss_r_total_day=args.max_loss_r_total_day,
        tf=args.tf, mom_lkb=args.mom_lkb, mom_th_long=args.mom_th_long, mom_th_short=args.mom_th_short, mom_ref=args.mom_ref,
        cap=args.cap, risk_frac=args.risk_frac, qty_fixed=args.qty_fixed, min_shares=args.min_shares, max_shares=args.max_shares
    )

def _symbols_from_args(args) -> List[str]:
    # universe non branché ici : on reste sur files (promoteur_core.txt, etc.)
    return _load_symbols(args.symbols_file, args.max_symbols, args.symbols_offset)

def _bars_dict(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return out
    for sym, g in df.groupby("symbol", observed=False):
        out[str(sym)] = g.reset_index(drop=True)
    return out

def _run_single_window(
    eng,
    cfg: dict,
    args,
    start: datetime,
    end: datetime,
    symbols: List[str],
) -> Tuple[Dict[str, float], List[dict]]:
    table_5m = cfg.get("db", {}).get("table_5m") or cfg.get("database", {}).get("mysql", {}).get("table_5m", "bars_5m_ny")
    df = _fetch_bars_5m_stream(eng, table_5m, symbols, start, end, args.batch_syms)
    bars = _bars_dict(df)

    regime = None
    if args.regime_symbol:
        regime = _query_regime_daily_ma(eng, table_5m, args.regime_symbol, start, end, args.regime_ma, args.regime_mode)

    P = _build_params(args)
    m, log = _smart_runner_ema_atr(bars, regime, P)
    return m, log

def main(argv=None):
    print("[INFO] Le script a démarré avec succès.")
    _setup_console_encoding()
    ap = _build_args()
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    start = _parse_date(args.start)
    end   = _parse_date(args.end)

    symbols = _symbols_from_args(args)
    excl = [s.strip() for s in (args.exclude_symbols or "").split(",") if s.strip()]
    if excl:
        exset = set(excl)
        symbols = [s for s in symbols if s not in exset]

    eng = _mysql_engine_from_config(cfg)

    title = f"Backtest Day ({args.tf}) [{args.env}]"

    if args.wf:
        # Walk-forward simple : fenêtres successives (on évalue seulement la fenêtre test)
        train, test, embargo = max(1,args.train_days), max(1,args.test_days), max(0,args.embargo_days)
        windows = []
        cur = start
        while cur + timedelta(days=train+test) <= end:
            tr_start = cur
            tr_end   = cur + timedelta(days=train)
            te_start = tr_end + timedelta(days=embargo)
            te_end   = te_start + timedelta(days=test)
            windows.append((te_start, te_end))
            cur = te_start  # glissement

        agg = {"trades":0,"winrate_pct":0.0,"pnl":0.0,"pf":0.0,"sharpe":0.0,"dd":0.0,"tp":0,"stop":0,"trend":0,"timeout":0}
        logs_total: List[dict] = []
        for (ws, we) in windows:
            m, log = _run_single_window(eng, cfg, args, ws, we, symbols)
            agg["trades"] += m["trades"]
            agg["pnl"] += m["pnl"]
            agg["tp"] += m.get("tp",0); agg["stop"] += m.get("stop",0)
            agg["trend"] += m.get("trend",0); agg["timeout"] += m.get("timeout",0)
            logs_total.extend(log)

        # recompute wr/pf/sharpe/dd from aggregated logs
        if logs_total:
            rets = np.array([t["pnl"] for t in logs_total], dtype=float)
            wins = int((rets > 0).sum())
            trades = len(rets)
            wr = 100.0 * wins / trades if trades else 0.0
            pnl_pos = rets[rets>0].sum()
            pnl_neg = abs(rets[rets<0].sum())
            pf = (pnl_pos / pnl_neg) if pnl_neg>0 else (float("inf") if pnl_pos>0 else 0.0)
            sharpe = (rets.mean() / (rets.std()+1e-9)) * math.sqrt(max(trades,1)) if trades else 0.0
            eq = np.cumsum(rets); peak = np.maximum.accumulate(eq); dd = float(np.max(peak - eq)) if trades else 0.0
            agg.update({"winrate_pct":wr,"pf":pf,"sharpe":sharpe,"dd":dd})

        print("📈 Walk-Forward", f"{args.tf.upper()} [{args.env}]")
        print(f"🗓️ {start.date()} → {end.date()} • {len(symbols)} symboles")
        print(f"🔁 {len(windows)} fenêtres")
        print(f"• PF {agg['pf']:.2f} • Sharpe {agg['sharpe']:.2f} • WorstDD {agg['dd']:.2f}")
        print(f"• Trades {agg['trades']} • Winrate {agg['winrate_pct']:.1f}%")
        print(f"💰 PnL cumulé: {agg['pnl']:.2f}")

        if args.notify and cfg.get("telegram", {}).get("token"):
            tok = cfg["telegram"]["token"]
            chat = cfg["telegram"].get("chat_id") or cfg["telegram"].get("chat_id_paper")
            msg = _tg_report("Walk-Forward " + args.tf, start, end, len(symbols), args.tf, args.entry_mode, args.side_mode, agg)
            _tg_send(tok, chat, msg)
        return 0

    # Single-window backtest
    m, log = _run_single_window(eng, cfg, args, start, end, symbols)
    print(f"📊 {title}")
    print(f"🗓️ {start.date()} → {end.date()} • {len(symbols)} symboles")
    print(f"• Trades {m['trades']} • Winrate {m['winrate_pct']:.1f}%")
    print(f"• PF {m['pf']:.2f} • Sharpe {m['sharpe']:.2f} • WorstDD {m['dd']:.2f}")
    print(f"✅TP {m.get('tp',0)} • ❌STOP {m.get('stop',0)} • 🔁Trend {m.get('trend',0)} • ⏰Time {m.get('timeout',0)}")
    print(f"💰 PnL: {m['pnl']:.2f}")

    if args.notify and cfg.get("telegram", {}).get("token"):
        tok = cfg["telegram"]["token"]
        chat = cfg["telegram"].get("chat_id") or cfg["telegram"].get("chat_id_paper")
        msg = _tg_report(title, start, end, len(symbols), args.tf, args.entry_mode, args.side_mode, m)
        _tg_send(tok, chat, msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
