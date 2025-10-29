# -*- coding: utf-8 -*-
# Backtest Day (5m/15m) — MySQL streaming + Breakout/Pullback/Cross
# - ORB/VWAP/VCP/RS (Breakout V2) avec options d’assouplissement
# - Sizing risk_frac ou qty_fixed, commissions fixes
# - Logs détaillés: raisons d’ignorés / signaux non tradés
# - Test unitaire 1 symbole (AAPL, BTCUSDT, etc.)
# - Telegram alert formatée

import argparse
import csv
import math
import os
import sys
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import requests
import yaml
from sqlalchemy import create_engine, text

# ============================== Utils console ==============================

def _setup_console_encoding():
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
        print("[INFO] Console encodée correctement.")
    except Exception as e:
        print(f"[WARN] Encodage console: {e}")

# ============================== Config ==============================

def load_config(config_path="src/config/config.yaml"):
    print(f"[INFO] Chargement de la configuration à partir de {config_path}...")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print("[INFO] Configuration chargée avec succès.")
        return cfg
    except Exception as e:
        print(f"[ERROR] Erreur config: {e}")
        return {}

# ============================== DB ==============================

def _make_engine(cfg: dict):
    """
    Essaie d'abord cfg['db']['uri'], sinon retombe sur database.mysql
    """
    uri = None
    if "db" in cfg and isinstance(cfg["db"], dict):
        uri = cfg["db"].get("uri")
    if uri:
        url = uri
        print("[INFO] Connexion via URI SQLAlchemy (pool optimisé).")
    else:
        m = cfg.get("database", {}).get("mysql", {})
        user = m.get("user")
        pwd = m.get("password")
        host = m.get("host", "127.0.0.1")
        port = m.get("port", 3306)
        db  = m.get("db")
        charset = m.get("charset", "utf8mb4")
        url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{db}?charset={charset}"
        print(f"[INFO] Connexion MySQL classique {host}:{port}/{db}")
    eng = create_engine(url, pool_pre_ping=True, pool_recycle=600, pool_size=6, max_overflow=4, future=True)
    with eng.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("[INFO] Connexion DB (URI) OK.")
    return eng

def _iter_query_bars_5m(engine, table: str, symbols: List[str], start: datetime, end: datetime, batch_syms: int = 150) -> Iterable[pd.DataFrame]:
    """
    Générateur qui renvoie des DataFrames par batch de symboles, pour limiter la RAM.
    """
    if not symbols:
        return
    total = len(symbols)
    with engine.begin() as c:
        for i in range(0, total, batch_syms):
            chunk = symbols[i:i + batch_syms]
            placeholders = ",".join([":s" + str(j) for j in range(len(chunk))])
            params = {f"s{k}": s for k, s in enumerate(chunk)}
            params.update({"start": start, "end": end})
            sql = text(f"""
                SELECT symbol, ts, open, high, low, close, volume
                FROM {table}
                WHERE symbol IN ({placeholders})
                  AND ts >= :start AND ts < :end
                ORDER BY symbol, ts
            """)
            df = pd.read_sql(sql, c, params=params)
            print(f"[RUN] batch {min(i+batch_syms, total)}/{total} — rows={len(df):,}")
            yield df

def _query_symbol_5m(engine, table: str, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    with engine.begin() as c:
        sql = text(f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM {table}
            WHERE symbol = :sym AND ts >= :start AND ts < :end
            ORDER BY ts
        """)
        return pd.read_sql(sql, c, params={"sym": symbol, "start": start, "end": end})

# ============================== Regime (daily) ==============================

def _daily_regime_mask(df_5m: pd.DataFrame, ma_len: int = 50, mode: str = "close_above_ma") -> pd.DataFrame:
    """
    Fabrique un masque daily pour SPY (ou autre):
      - date
      - regime (bool)
    """
    if df_5m.empty:
        return pd.DataFrame(columns=["date", "regime"])
    d = df_5m.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    dd = d.set_index("ts")["close"].resample("1D").last().dropna().to_frame("close")
    dd["ma"] = dd["close"].rolling(ma_len, min_periods=ma_len).mean()
    if mode == "close_above_ma":
        dd["regime"] = dd["close"] > dd["ma"]
    else:  # ema_slope_up équivalent simple
        dd["ma_lag"] = dd["ma"].shift(1)
        dd["regime"] = dd["ma"] > dd["ma_lag"]
    out = dd[["regime"]].copy()
    out["date"] = out.index.date
    return out.reset_index(drop=True)

# ============================== Telegram ==============================

def _tg_report(title: str, start: datetime, end: datetime, n_symbols: int, stats: dict, tf: str, entry_mode: str,
               tp_count: int, stop_count: int, trend_count: int, time_count: int, pnl_dollars: float) -> str:
    wr = stats.get("winrate_pct", 0.0)
    pf = stats.get("pf", 0.0)
    sharpe = stats.get("sharpe", 0.0)
    dd = stats.get("dd", 0.0)
    trades = stats.get("trades", 0)
    # Fiabilité TP sur (TP+STOP)
    tp_stop = max(1, tp_count + stop_count)
    fiab = (tp_count / tp_stop) * 100.0
    sign = "+" if pnl_dollars >= 0 else "-"
    return (
        f"📊 {title}\n"
        f"🗓️ {start.date()} → {end.date()} • {n_symbols} symb • TF {tf.upper()} • {entry_mode}\n"
        f"📦 Trades {trades} • Win {wr:.1f}% • PF {pf:.2f} • Sharpe {sharpe:.2f} • DD {dd:.2f}\n"
        f"✅TP {tp_count} • ❌STOP {stop_count} • 🔁Trend {trend_count} • ⏰Time {time_count}\n"
        f"🧠 Fiabilité TP {fiab:.1f}% (sur TP+STOP)\n"
        f"💰 PnL: {sign}${abs(pnl_dollars):,.2f}"
    )

def _send_tg(cfg: dict, text: str, env: str = "PAPER") -> None:
    tgc = cfg.get("telegram", {}) or {}
    if not tgc:
        return
    token = tgc.get("token") or tgc.get("bot_token")
    if not token:
        return
    chat_id = tgc.get("chat_id")
    if not chat_id:
        chat_id = tgc.get("chat_id_live") if env.upper() == "LIVE" else tgc.get("chat_id_paper")
    if not chat_id:
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": text})
        if r.status_code != 200:
            print(f"[WARN] Telegram HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[WARN] Telegram send error: {e}")

# ============================== Features & Runner ==============================

def _within_rth(ts: pd.Timestamp, tz_naive=True):
    # RTH US: 9:30 → 16:00 heure de la bourse (on suppose données alignées)
    tm = ts.time()
    return (tm >= dtime(9,30)) and (tm <= dtime(16,0))

def _prep_intraday_features(g: pd.DataFrame, ema_fast: int, ema_slow: int, slope_lag: int,
                            orb_minutes: int = 30) -> pd.DataFrame:
    g = g.copy().sort_values("ts").reset_index(drop=True)
    g["ts"] = pd.to_datetime(g["ts"])
    g["date"] = g["ts"].dt.date

    # EMAs
    g["ema_f"] = g["close"].ewm(span=ema_fast, adjust=False, min_periods=ema_fast).mean()
    g["ema_s"] = g["close"].ewm(span=ema_slow, adjust=False, min_periods=ema_slow).mean()
    g["ema_s_lag"] = g["ema_s"].shift(slope_lag)

    # VWAP intraday
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    vol = g["volume"].clip(lower=0)
    vwap_num = (tp * vol).groupby(g["date"]).cumsum()
    vwap_den = vol.groupby(g["date"]).cumsum().replace(0, np.nan)
    g["vwap"] = vwap_num / vwap_den

    # ORB
    # intervalle en minutes par barre
    if len(g) >= 3:
        step_min = int(np.median(np.diff(g["ts"].values).astype("timedelta64[m]").astype(int)))
    else:
        step_min = 5
    step_min = max(1, step_min)
    n_bars = max(1, orb_minutes // step_min)
    g["bar_idx"] = g.groupby("date").cumcount()
    orb = g[g["bar_idx"] < n_bars].groupby("date").agg(orb_hi=("high", "max"), orb_lo=("low", "min"))
    g = g.join(orb, on="date")

    # VCP (contraction): ATR(10)/ATR(20)
    tr = (g["high"] - g["low"]).clip(lower=0)
    g["atr10"] = tr.rolling(10, min_periods=10).mean()
    g["atr20"] = tr.rolling(20, min_periods=20).mean()
    g["vcp_ratio"] = (g["atr10"] / g["atr20"])
    return g

def _daily_rs_ok_map(df_5m: pd.DataFrame, lb: int = 20, th: float = 0.05) -> Dict[object, bool]:
    if df_5m.empty:
        return {}
    d = df_5m.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    dd = d.set_index("ts")["close"].resample("1D").last().dropna()
    ret20 = dd.pct_change(lb)
    mask = (ret20.shift(1) > th)  # veille
    return {idx.date(): bool(val) for idx, val in mask.items()}

def _calc_stats(all_logs: List[dict], commission_fixed: float, cap: float) -> dict:
    trades = len(all_logs)
    if trades == 0:
        return {"trades": 0, "winrate_pct": 0.0, "pf": 0.0, "sharpe": 0.0, "dd": 0.0, "pnl": 0.0}
    pnl_list = [x["pnl"] for x in all_logs]
    wins = sum(1 for x in pnl_list if x > 0)
    wr = wins / trades * 100.0
    gross_profit = sum(x for x in pnl_list if x > 0)
    gross_loss = -sum(x for x in pnl_list if x < 0)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    # Sharpe simple avec returns approximés / cap
    rets = np.array(pnl_list) / max(1.0, cap)
    if rets.std(ddof=1) > 1e-12:
        sharpe = (rets.mean() / rets.std(ddof=1)) * np.sqrt(252*6.5*4)  # très approx intraday
    else:
        sharpe = 0.0
    # DD approximé cumul
    eq = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(eq)
    dd = np.max(peak - eq) if len(eq) else 0.0
    return {"trades": trades, "winrate_pct": wr, "pf": pf, "sharpe": sharpe, "dd": dd, "pnl": sum(pnl_list)}

def _size_position(entry: float, stop_px: float, cap: float, risk_frac: float, qty_fixed: int,
                   min_shares: int, max_shares: int) -> int:
    if qty_fixed and qty_fixed > 0:
        q = qty_fixed
    else:
        R_abs = max(0.01, entry - stop_px)  # valeur $ du risque par action
        risk_dollars = max(10.0, cap * risk_frac)
        q = int(risk_dollars / R_abs)
    q = max(min_shares, min(q, max_shares))
    return q

def _smart_runner_ema_atr(
    bars: Dict[str, pd.DataFrame],
    # ===== core params =====
    ema_fast: int = 20,
    ema_slow: int = 120,
    atr_len: int = 14,
    stop_pct: float = 0.012,
    tp_mult: float = 1.25,
    cooldown_bars: int = 5,                 # A) cooldown réduit
    max_hold_bars: int = 28,
    slope_lag: int = 20,
    max_trades_per_symbol_day: int = 3,     # A) 2–3 trades / symbole / jour
    # ===== filters =====
    require_vwap_reclaim: bool = False,     # A) désactivé par défaut pour tester
    min_atr_pct: float = 0.003, max_atr_pct: float = 0.14,
    min_vol_5m: int = 40000,
    rth_only: bool = True,
    skip_first_min: int = 15, skip_last_min: int = 75,
    midday_start: str = "11:00", midday_end: str = "13:10",
    # ===== entry modes =====
    entry_mode: str = "breakout",           # breakout | pullback | cross
    breakout_n: int = 12,                   # A) 10–15 → plus de signaux
    pullback_tol: float = 0.01,             # A) 0.01 à 0.015
    side_mode: str = "long",                # long | short | both (implémenté long)
    # ===== risk controls =====
    breakeven_after_r: float = 0.7,
    trail_after_r: float = 1.1,
    trail_mult_r: float = 0.8,
    fee_bps: int = 0,
    # ===== accounting =====
    commission_fixed: float = 1.0,
    cap: float = 10000.0,
    risk_frac: float = 0.015,               # B) 1.5% par défaut
    qty_fixed: int = 0,
    min_shares: int = 1, max_shares: int = 600,
    # ===== regime =====
    regime_daily: Optional[pd.DataFrame] = None,
    # ===== logging =====
    logger=None,
) -> Tuple[Dict[str, float], List[dict], Dict[str, int]]:
    """
    Retourne (stats, trade_logs, exit_reason_counts)
    """
    def log(msg): 
        if logger: logger(msg)

    def fee(px): return px * (fee_bps / 1e4)

    # regime map
    if regime_daily is None or len(regime_daily) == 0:
        regime_map = {}
    else:
        if isinstance(regime_daily, dict):
            regime_map = regime_daily
        else:
            regime_map = {r["date"]: bool(r["regime"]) for _, r in regime_daily.iterrows()}

    pnl = 0.0
    trades = 0
    wins = 0
    trade_logs: List[dict] = []
    exit_counts = {"TP":0, "STOP":0, "TREND_BROKE":0, "TIMEOUT":0}

    for sym, df in bars.items():
        if df is None or df.empty:
            log(f"[SKIP] {sym}: dataframe vide")
            continue

        g = _prep_intraday_features(df, ema_fast, ema_slow, slope_lag, orb_minutes=30)
        rs_map = _daily_rs_ok_map(df, lb=20, th=0.05)

        pos = False
        entry = 0.0
        entry_ts = None
        stop_px = 0.0
        tp_px = 0.0
        hold = 0
        cooldown = 0
        highest_close = None
        trades_today: Dict[object, int] = {}
        date_last_tx = None

        for i in range(len(g)-1):
            row, nxt = g.iloc[i], g.iloc[i+1]
            dte = row["date"]
            ts = row["ts"]
            nxt_open = float(nxt["open"])

            # régime
            if regime_map and not regime_map.get(dte, True):
                continue

            # Filtres temporels (RTH / fenêtre)
            if rth_only and not _within_rth(ts):
                log(f"[IGN] {sym} {ts} hors RTH")
                continue
            # skip première x min et dernière y min
            minute_of_day = ts.hour*60 + ts.minute
            if minute_of_day < 9*60+30 + skip_first_min:
                log(f"[IGN] {sym} {ts} early-window")
                continue
            if minute_of_day > 16*60 - skip_last_min:
                log(f"[IGN] {sym} {ts} late-window")
                continue
            # pause midday
            try:
                md_s_h, md_s_m = map(int, midday_start.split(":"))
                md_e_h, md_e_m = map(int, midday_end.split(":"))
                if dtime(md_s_h, md_s_m) <= ts.time() <= dtime(md_e_h, md_e_m):
                    log(f"[IGN] {sym} {ts} midday")
                    continue
            except:
                pass

            # Filtres marché (ATR/Volume)
            # ATR% instantané approximé: range/close
            rng = float(row["high"] - row["low"])
            close = float(row["close"])
            atr_pct_now = rng / max(1e-6, close)
            if atr_pct_now < min_atr_pct:
                log(f"[IGN] {sym} {ts} ATR_LOW {atr_pct_now:.4f}")
                continue
            if atr_pct_now > max_atr_pct:
                log(f"[IGN] {sym} {ts} ATR_HIGH {atr_pct_now:.2f}")
                continue
            if row["volume"] < min_vol_5m:
                log(f"[IGN] {sym} {ts} VOL_LOW {int(row['volume'])}")
                continue

            # Conditions trend/slope
            emaf = float(row["ema_f"]) if not pd.isna(row["ema_f"]) else np.nan
            emas = float(row["ema_s"]) if not pd.isna(row["ema_s"]) else np.nan
            emas_lag = float(row["ema_s_lag"]) if not pd.isna(row["ema_s_lag"]) else np.nan
            trend_up = (not np.isnan(emaf) and not np.isnan(emas)) and (emaf > emas) and (close > emaf)
            slope_ok = (not np.isnan(emas) and not np.isnan(emas_lag) and emas > emas_lag)

            if cooldown > 0:
                cooldown -= 1

            # Entrée
            if not pos:
                # limites journalières
                if trades_today.get(dte, 0) >= max_trades_per_symbol_day:
                    log(f"[IGN] {sym} {ts} MAX_TRADES_DAY")
                    continue
                if cooldown > 0:
                    log(f"[IGN] {sym} {ts} COOLDOWN {cooldown}")
                    continue

                # Sélecteurs d’entrée
                signal_ok = False
                reason_ok = []

                if entry_mode == "breakout":
                    # Cassure: plus haut des breakout_n dernières bougies
                    hi_n = float(g["high"].iloc[max(0, i-breakout_n+1):i+1].max()) if i>0 else row["high"]
                    breakout_ok = close > hi_n
                    reason_ok.append(f"BO[{breakout_n}]={breakout_ok}")
                    # ORB, VWAP reclaim optionnels (assouplis)
                    orb_ok = (not pd.isna(row.get("orb_hi"))) and (close > float(row["orb_hi"]))
                    vwap_ok = True if not require_vwap_reclaim else (not pd.isna(row.get("vwap")) and close > float(row["vwap"]))
                    # Contraction
                    vcp_ok = (not pd.isna(row.get("vcp_ratio"))) and (float(row["vcp_ratio"]) < 0.95)
                    rs_ok = rs_map.get(dte, True)
                    signal_ok = (trend_up and slope_ok and breakout_ok and vcp_ok and rs_ok and (orb_ok or breakout_ok) and vwap_ok)

                elif entry_mode == "pullback":
                    # Repli vers ema_f puis reprise
                    if not np.isnan(emaf) and emaf > 0:
                        tol = pullback_tol
                        near = abs(close - emaf) / emaf <= tol
                        # confirmation: close >= prev high
                        prev_hi = float(g["high"].iloc[i-1]) if i>0 else row["high"]
                        conf = close >= prev_hi
                        signal_ok = trend_up and slope_ok and near and conf
                        reason_ok.append(f"PB tol={tol} near={near} conf={conf}")

                else:  # cross
                    # croisement ema_f > ema_s récent
                    cross_up = (i >= 1) and (g.iloc[i - 1]["ema_f"] <= g.iloc[i - 1]["ema_s"]) and (emaf > emas)
                    signal_ok = cross_up and slope_ok
                    reason_ok.append(f"CROSS={signal_ok}")

                if not signal_ok:
                    log(f"[IGN] {sym} {ts} NO_SIGNAL {','.join(reason_ok)} trend_up={trend_up} slope_ok={slope_ok}")
                    continue

                # Calcule stop/tp et sizing
                R = stop_pct * close
                entry = float(nxt_open)
                stop_px = entry - R
                tp_px = entry + (tp_mult * R)
                qty = _size_position(entry, stop_px, cap, risk_frac, qty_fixed, min_shares, max_shares)
                if qty <= 0:
                    log(f"[IGN] {sym} {ts} QTY_ZERO")
                    continue

                pos = True
                entry_ts = nxt["ts"]
                highest_close = close
                hold = 0
                cooldown = 0
                date_last_tx = dte
                continue

            # Gestion position
            hold += 1
            if highest_close is None or close > highest_close:
                highest_close = close

            R = stop_pct * entry
            # BE
            if (close - entry) >= breakeven_after_r * R:
                stop_px = max(stop_px, entry)
            # Trailing
            if (close - entry) >= trail_after_r * R and highest_close is not None:
                trail_stop = highest_close - trail_mult_r * R
                stop_px = max(stop_px, trail_stop)

            # sorties
            hit_stop = row["low"] <= stop_px
            hit_tp   = row["high"] >= tp_px

            exit_px = None
            reason = None
            if hit_stop:
                exit_px = stop_px; reason = "STOP"
            elif hit_tp:
                exit_px = tp_px; reason = "TP"
            elif not (trend_up and slope_ok and (not require_vwap_reclaim or (close > float(row.get("vwap", -1e9))))):
                exit_px = nxt_open; reason = "TREND_BROKE"
            elif hold >= max_hold_bars:
                exit_px = nxt_open; reason = "TIMEOUT"

            if exit_px is not None:
                # PnL en $: (exit-entry)*qty - commissions
                r_dollars = (exit_px - entry) * qty - commission_fixed
                pnl += r_dollars
                trades += 1
                if r_dollars > 0:
                    wins += 1
                trade_logs.append({
                    "symbol": sym, "entry_ts": entry_ts, "exit_ts": nxt["ts"],
                    "entry": entry, "exit": exit_px, "qty": qty, "pnl": r_dollars, "reason": reason
                })
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
                pos = False
                cooldown = max(cooldown_bars, 0)
                trades_today[dte] = trades_today.get(dte, 0) + 1

        # clôture EOD si encore en position
        if pos and len(g):
            last = g.iloc[-1]
            exit_px = float(last["close"])
            qty = _size_position(entry, stop_px, cap, risk_frac, qty_fixed, min_shares, max_shares)
            r_dollars = (exit_px - entry) * qty - commission_fixed
            pnl += r_dollars
            trades += 1
            if r_dollars > 0:
                wins += 1
            trade_logs.append({
                "symbol": sym, "entry_ts": entry_ts, "exit_ts": last["ts"],
                "entry": entry, "exit": exit_px, "qty": qty, "pnl": r_dollars, "reason": "EOD"
            })
            exit_counts["EOD"] = exit_counts.get("EOD", 0) + 1

    wr = (wins / trades * 100.0) if trades else 0.0
    stats = {"trades": trades, "winrate_pct": wr, "pnl": pnl, "pnl_weighted": pnl}
    return stats, trade_logs, exit_counts

# ============================== Main ==============================

def main(argv=None):
    _setup_console_encoding()
    cfg = load_config()
    if not cfg:
        return 1

    parser = argparse.ArgumentParser("bot_backtest")
    # Dates / univers
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--symbols-offset", type=int, default=0)
    parser.add_argument("--batch-syms", type=int, default=180)
    parser.add_argument("--one-symbol", default="")  # D) test unitaire

    # Profil / TF
    parser.add_argument("--profile", choices=["defensif","normal","agressif"], default=cfg.get("profile","defensif"))
    parser.add_argument("--tf", choices=["5m","15m"], default="15m")

    # Regime
    parser.add_argument("--regime-symbol", default="SPY")
    parser.add_argument("--regime-ma", type=int, default=50)
    parser.add_argument("--regime-mode", choices=["close_above_ma","ema_slope_up"], default="close_above_ma")

    # EMAs & ATR
    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-slow", type=int, default=120)
    parser.add_argument("--atr-len", type=int, default=14)

    # Stops / TP / gestion
    parser.add_argument("--stop-mode", choices=["pct","atr"], default="pct")
    parser.add_argument("--atr-mult", type=float, default=1.9)
    parser.add_argument("--stop-pct", type=float, default=0.012)
    parser.add_argument("--tp-mult", type=float, default=1.25)
    parser.add_argument("--cooldown", type=int, default=5)                         # A)
    parser.add_argument("--max-hold", type=int, default=28)
    parser.add_argument("--slope-lag", type=int, default=20)
    parser.add_argument("--max-trades-per-symbol-day", type=int, default=3)        # A)
    parser.add_argument("--breakeven-after-r", type=float, default=0.7)
    parser.add_argument("--trail-after-r", type=float, default=1.1)
    parser.add_argument("--trail-mult-r", type=float, default=0.8)

    # Filtres marché
    parser.add_argument("--min-atr-pct", type=float, default=0.003)
    parser.add_argument("--max-atr-pct", type=float, default=0.14)
    parser.add_argument("--min-vol-5m", type=int, default=40000)
    parser.add_argument("--vwap-filter", action="store_true", dest="require_vwap_reclaim")  # A) toggle
    parser.add_argument("--rth-only", action="store_true", default=True)
    parser.add_argument("--skip-first-min", type=int, default=15)
    parser.add_argument("--skip-last-min", type=int, default=75)
    parser.add_argument("--midday-start", default="11:00")
    parser.add_argument("--midday-end", default="13:10")

    # Entrées
    parser.add_argument("--entry-mode", choices=["breakout","pullback","cross"], default="breakout")
    parser.add_argument("--breakout-n", type=int, default=12)
    parser.add_argument("--pullback-tol", type=float, default=0.01)
    parser.add_argument("--side-mode", choices=["long","short","both"], default="long")

    # Sizing & commissions
    parser.add_argument("--commission-fixed", type=float, default=1.0)
    parser.add_argument("--qty-fixed", type=int, default=0)
    parser.add_argument("--cap", type=float, default=10000.0)
    parser.add_argument("--risk-frac", type=float, default=0.015)                   # B)
    parser.add_argument("--min-shares", type=int, default=1)
    parser.add_argument("--max-shares", type=int, default=600)

    # Environnement / notif
    parser.add_argument("--env", choices=["PAPER","LIVE"], default="PAPER")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--config", default="src/config/config.yaml")

    # Debug logs
    parser.add_argument("--debug-log", default="")  # C) fichier log détaillé

    args = parser.parse_args(argv)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end   = datetime.strptime(args.end, "%Y-%m-%d")
    print(f"[INFO] Période du backtest : {start} à {end}")

    # Logger
    logger = None
    log_file = None
    if args.debug_log:
        log_path = Path(args.debug_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
        def logger_fn(msg): 
            log_file.write(msg + "\n")
        logger = logger_fn
        print(f"[INFO] Debug log → {log_path}")

    # Symbols
    syms: List[str] = []
    if args.one_symbol:
        syms = [args.one_symbol.strip().upper()]
        print(f"[INFO] Test 1 symbole: {syms[0]}")
    elif args.symbols_file and Path(args.symbols_file).is_file():
        syms = [l.strip() for l in Path(args.symbols_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.symbols_offset:
            syms = syms[args.symbols_offset:]
        if args.max_symbols and args.max_symbols > 0:
            syms = syms[:args.max_symbols]
        print(f"[INFO] Fichier symboles: {args.symbols_file} — {len(syms)} symboles")
    else:
        print("[ERROR] Aucun symbole fourni (fichier ou --one-symbol).")
        if log_file: log_file.close()
        return 2

    # DB
    eng = _make_engine(cfg)
    table_5m = cfg.get("db", {}).get("table_5m") or cfg.get("database", {}).get("table_5m", "bars_5m_ny")

    # Regime
    regime = None
    if args.regime_symbol:
        try:
            r5 = _query_symbol_5m(eng, table_5m, args.regime_symbol, start, end)
            regime = _daily_regime_mask(r5, args.regime_ma, args.regime_mode)
        except Exception as e:
            print(f"[WARN] Regime indisponible: {e}")

    # Agrégation résultats
    all_logs: List[dict] = []
    exit_totals = {"TP":0,"STOP":0,"TREND_BROKE":0,"TIMEOUT":0,"EOD":0}

    # Streaming par batch
    for df in _iter_query_bars_5m(eng, table_5m, syms, start, end, batch_syms=args.batch_syms):
        if df.empty:
            continue
        # split par symbole
        bars = {s: g.copy().reset_index(drop=True) for s, g in df.groupby("symbol")}

        # stop_pct à partir de stop-mode
        stop_pct = args.stop_pct
        if args.stop_mode == "atr":
            # conversion simple: stop_pct ≈ atr_mult * ATR% moyen (approx via range/close)
            # (pour rester compatible: on continue d’utiliser stop_pct comme R)
            stop_pct = 0.008 * args.atr_mult

        stats, logs, exit_counts = _smart_runner_ema_atr(
            bars,
            ema_fast=args.ema_fast, ema_slow=args.ema_slow, atr_len=args.atr_len,
            stop_pct=stop_pct, tp_mult=args.tp_mult,
            cooldown_bars=args.cooldown, max_hold_bars=args.max_hold,
            slope_lag=args.slope_lag, max_trades_per_symbol_day=args.max_trades_per_symbol_day,
            require_vwap_reclaim=args.require_vwap_reclaim,
            min_atr_pct=args.min_atr_pct, max_atr_pct=args.max_atr_pct,
            min_vol_5m=args.min_vol_5m, rth_only=args.rth_only,
            skip_first_min=args.skip_first_min, skip_last_min=args.skip_last_min,
            midday_start=args.midday_start, midday_end=args.midday_end,
            entry_mode=args.entry_mode, breakout_n=args.breakout_n,
            pullback_tol=args.pullback_tol, side_mode=args.side_mode,
            breakeven_after_r=args.breakeven_after_r, trail_after_r=args.trail_after_r,
            trail_mult_r=args.trail_mult_r, fee_bps=0,
            commission_fixed=args.commission_fixed, cap=args.cap, risk_frac=args.risk_frac,
            qty_fixed=args.qty_fixed, min_shares=args.min_shares, max_shares=args.max_shares,
            regime_daily=regime, logger=logger
        )

        all_logs.extend(logs)
        for k, v in exit_counts.items():
            exit_totals[k] = exit_totals.get(k, 0) + v

    # Stats finales
    S = _calc_stats(all_logs, args.commission_fixed, args.cap)
    tp = exit_totals.get("TP", 0)
    st = exit_totals.get("STOP", 0)
    tr = exit_totals.get("TREND_BROKE", 0)
    tm = exit_totals.get("TIMEOUT", 0)

    # Print résumé
    title = f"Backtest Day ({args.tf}) [{args.env}]"
    print(f"📊 {title}")
    print(f"🗓️ {start.date()} → {end.date()} • {len(syms)} symb • TF {args.tf.upper()} • {args.entry_mode}/{args.side_mode}")
    print(f"📦 Trades {S['trades']} • Win {S['winrate_pct']:.1f}% • PF {S['pf']:.2f} • Sharpe {S['sharpe']:.2f} • DD {S['dd']:.2f}")
    print(f"✅TP {tp} • ❌STOP {st} • 🔁Trend {tr} • ⏰Time {tm}")
    print(f"💰 PnL: ${S['pnl']:.2f}")

    # Telegram si demandé
    if args.notify:
        tgc = cfg.get("telegram", {}) or {}
        enabled_by_default = tgc.get("enabled_by_default", True)
        if enabled_by_default:
            msg = _tg_report(title, start, end, len(syms), {
                "trades": S["trades"],
                "winrate_pct": S["winrate_pct"],
                "pf": S["pf"],
                "sharpe": S["sharpe"],
                "dd": S["dd"]
            }, args.tf, f"{args.entry_mode}/{args.side_mode}", tp, st, tr, tm, S["pnl"])
            _send_tg(cfg, msg, env=args.env)
        else:
            print("[TG] disabled in YAML")

    if log_file:
        log_file.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
