import requests
import pandas as pd
import numpy as np
import time
import datetime
import os
import json
from typing import List, Dict
import aiohttp
import asyncio
import pandas_ta as ta
import vectorbt as vbt
from pathlib import Path
import csv
import pytz
import urllib.parse, time
import ssl
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()


try:
    import pyarrow  # noqa: F401
    PARQUET_ENGINE = "pyarrow"
except Exception:
    try:
        import fastparquet  # noqa: F401
        PARQUET_ENGINE = "fastparquet"
    except Exception:
        PARQUET_ENGINE = None

def _load_env_file():
    """Простая загрузка переменных из .env без сторонних зависимостей."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

SIGNALS_CSV_FILE = "signals.csv"
ACTIVE_TRADES_FILE = "active_trades.json"

from collections import defaultdict

TELEGRAM_FAILS = 0
FMP_FAILS = defaultdict(int)  # ключ: ticker
FMP_FAIL_THRESHOLD = 5
TEL_FAIL_THRESHOLD = 5

# Небольшой кэш, чтобы не долбить API на каждом тикере
_FMP_FILTER_CACHE = {}  # key -> (expires_ts, data)
_FMP_FILTER_TTL = 600   # секунд
MTF_CTX_TTL_SEC = 180  # кэш MTF-контекста: 3 минуты

# --- Event blackout settings ---
EARNINGS_BLACKOUT_BEFORE = 2   # дней до отчётности
EARNINGS_BLACKOUT_AFTER  = 1   # дней после отчётности
SPLIT_BLACKOUT_BEFORE    = 1   # сплитов обычно хватает 1 дня до/после
SPLIT_BLACKOUT_AFTER     = 1

EVENTS_CACHE_TTL_SEC     = 3600  # кэшируем на час
_EVENTS_CACHE = {}  # {ticker: {"at": epoch_sec, "events": [{"type": "...", "date": "YYYY-MM-DD"}]}}


def redact(s: str) -> str:
    s = s.replace(os.getenv("TELEGRAM_TOKEN", ""), "***")
    for k in API_KEYS:
        if k:
            s = s.replace(k, "***")
    return s


def slog(lvl: str, msg: str, ticker: str = "-", func: str = "-", **kwargs):
    # защитимся, если где-то вызвали level="INFO" в kwargs
    if "level" in kwargs:
        kwargs.pop("level", None)
    try:
        extra = json.dumps(kwargs, ensure_ascii=False) if kwargs else ""
    except Exception:
        extra = str(kwargs)
    line = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S} | {lvl} | {ticker} | {func} | {msg} | {extra}"
    print(line)

def _save_filter_reason_snapshot(ticker: str, name: str, payload: dict):
    """
    Сохраняет причину фильтрации в data/<ticker>/ctx/filters_YYYYMMDD_HHMMSS_<name>.json
    """
    try:
        base = Path("data") / ticker / "ctx"
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        (base / f"filters_{stamp}_{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        slog("WARNING", f"save filter snapshot failed: {e}", ticker=ticker, func="_save_filter_reason_snapshot")

def log(message: str, level: str = "INFO", ticker: str | None = None, func: str | None = None, **extras):
    """Сохранена для совместимости."""
    slog(level, message, ticker, func, **extras)

def export_signal_to_csv(
    ticker: str,
    signal_text: str,
    confidence: str,
    last_price: float,
    take_profit: float,
    stop_loss: float,
    adx_value: float,
    volume: float,
    volume_ma: float,
    signal_time: str,
    direction: str | None = None,
    entry: float | None = None,
    rr: float | None = None,
    reason: str | None = None,
    trail_alert_sent: bool = False,
):
    file_exists = os.path.isfile(SIGNALS_CSV_FILE)
    fieldnames = [
        "datetime",
        "ticker",
        "signal_type",
        "confidence",
        "price",
        "take_profit",
        "stop_loss",
        "adx",
        "volume",
        "volume_ma",
        "tradingview_url",
        "direction",
        "entry",
        "rr",
        "reason",
        "trail_alert_sent",
    ]
    row = {
        "datetime": signal_time,
        "ticker": ticker,
        "signal_type": signal_text,
        "confidence": confidence,
        "price": last_price,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "adx": adx_value,
        "volume": volume,
        "volume_ma": volume_ma,
        "tradingview_url": f"https://www.tradingview.com/chart/?symbol={ticker.replace('.ME', '')}",
        "direction": direction,
        "entry": entry,
        "rr": rr,
        "reason": reason,
        "trail_alert_sent": trail_alert_sent,
    }
    try:
        with open(SIGNALS_CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        slog("ERROR", f"Ошибка экспорта сигнала в CSV: {e}", func="export_signal_to_csv")

NY = pytz.timezone("America/New_York")

def filter_regular_hours_us(df: pd.DataFrame) -> pd.DataFrame:
    """
    FMP 5m 'datetime' приходит без tz и обычно соответствует US/Eastern.
    Фильтруем по времени 09:30–16:00 и будним дням, не меняя tz.
    """
    if df is None or df.empty:
        return df

    dt = pd.to_datetime(df["datetime"], errors="coerce")
    # будние дни: 0..4
    wd_mask = dt.dt.weekday <= 4
    # время в минутах с 00:00
    mins = dt.dt.hour * 60 + dt.dt.minute
    open_mins  = 9*60 + 30   # 09:30
    close_mins = 16*60       # 16:00
    ses_mask = (mins >= open_mins) & (mins <= close_mins)

    out = df.loc[wd_mask & ses_mask].copy()
    return out.sort_values("datetime").reset_index(drop=True)

WARMUP_BARS = 100

def trim_warmup(df: pd.DataFrame) -> pd.DataFrame:
    ind_cols = ["ema50","ema20","ema21","macd","macd_signal","macd_hist","rsi","adx"]
    ready = df[ind_cols].notna().all(axis=1)
    first_ready_idx = int(ready.idxmax()) if ready.any() else 0
    cut = max(0, first_ready_idx - 5)  # небольшой буфер
    return df.iloc[cut:].reset_index(drop=True)

# Telegram/API настройки берём из переменных окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
if not API_KEYS:
    API_KEYS = ["demo"]  # Заглушка, чтобы код не упал
def validate_env():
    ok = True
    if not TELEGRAM_TOKEN or not TELEGRAM_TOKEN.strip():
        slog("ERROR", "TELEGRAM_TOKEN пуст — отключаю Telegram-уведомления", func="validate_env")
        ok = False
    if not CHAT_ID or not CHAT_ID.strip():
        slog("ERROR", "CHAT_ID пуст — отключаю Telegram-уведомления", func="validate_env")
        ok = False
    if API_KEYS == ["demo"]:
        slog("WARNING", "Используется demo API_KEYS — возможны лимиты и пустые данные", func="validate_env")
    return ok

TELEGRAM_ENABLED = validate_env()
current_api_key_index = 0  # Текущий активный ключ

def _get_filter_thresholds(ticker_cfg: dict) -> dict:
    out = dict(FILTER_DEFAULTS)
    if not ticker_cfg:
        return out
    for k in FILTER_DEFAULTS.keys():
        if k in ticker_cfg:
            out[k] = ticker_cfg[k]
    return out


# === FMP client & loaders (CANONICAL) ===

BASE_URL_US = "https://financialmodelingprep.com/api/v3"  # единая точка сборки URL

current_api_key_index = 0

def rotate_api_key():
    global current_api_key_index
    prev = current_api_key_index
    current_api_key_index = (current_api_key_index + 1) % len(API_KEYS)
    if current_api_key_index != prev:
        slog("WARNING", f"Переключен на API-ключ #{current_api_key_index+1}", func="rotate_api_key")

async def fmp_get_json(session, path: str, params: dict | None = None) -> list | dict:
    """
    Унифицированный GET к FMP c обработкой ошибок/429.
    path: БЕЗ ведущего слеша. Примеры:
      - 'historical-chart/5min/AAPL'
      - 'historical-price-full/AAPL'
      - 'stable/grades'
    """
    api_key = API_KEYS[current_api_key_index]
    q = dict(params or {})
    q["apikey"] = api_key
    url = f"{BASE_URL_US}/{path}"
    for attempt in range(1, 6):
        try:
            async with session.get(url, params=q, timeout=20) as r:
                if r.status == 429:
                    rotate_api_key()
                    await asyncio.sleep(min(2**attempt, 10))
                    continue
                r.raise_for_status()
                try:
                    return await r.json()
                except Exception:
                    txt = await r.text()
                    raise RuntimeError(f"Bad JSON from FMP: {txt[:200]}")
        except Exception as e:
            slog("WARNING", f"fmp_get_json error: {e}", func="fmp_get_json", url=url, attempt=attempt)
            await asyncio.sleep(min(2**attempt, 10))
    raise RuntimeError("fmp_get_json: все попытки исчерпаны")

def _cache_get(key: str):
    v = _FMP_FILTER_CACHE.get(key)
    if not v:
        return None
    exp, data = v
    if time.time() > exp:
        _FMP_FILTER_CACHE.pop(key, None)
        return None
    return data

def _cache_put(key: str, data, ttl=_FMP_FILTER_TTL):
    _FMP_FILTER_CACHE[key] = (time.time() + ttl, data)

# Новости/квоты (канон)
async def get_company_news(session, ticker: str) -> List[Dict]:
    path = "stock_news"
    try:
        data = await fmp_get_json(session, path, params={"tickers": ticker, "limit": 5})
        return data if isinstance(data, list) else []
    except Exception as e:
        slog("ERROR", f"Ошибка загрузки новостей: {e}", ticker=ticker, func="get_company_news")
        return []

async def get_spread(session, ticker: str) -> float:
    """Возвращает спред bid/ask в % для US акций (с защитой от пустых полей)."""
    path = f"quote/{ticker}"
    try:
        data = await fmp_get_json(session, path, params={})
        if not isinstance(data, list) or not data:
            slog("WARNING", "quote пустой", ticker=ticker, func="get_spread"); return 0.0
        q = data[0] or {}
        bid = float(q.get("bidPrice") or q.get("bid") or 0.0)
        ask = float(q.get("askPrice") or q.get("ask") or 0.0)
        if bid <= 0 or ask <= 0 or ask < bid:
            slog("WARNING", "некорректный bid/ask", ticker=ticker, func="get_spread", bid=bid, ask=ask)
            return 0.0
        return (ask - bid) / bid * 100.0
    except Exception as e:
        slog("ERROR", f"Ошибка получения спреда: {e}", ticker=ticker, func="get_spread")
        return 0.0

# События + кэш (канон)
EVENTS_CACHE_TTL_SEC = 3600
_EVENTS_CACHE: dict[str, dict] = {}

async def _save_events_local(ticker: str, events: list[dict]) -> None:
    try:
        base = Path("data") / ticker / "meta"
        base.mkdir(parents=True, exist_ok=True)
        (base / "events.json").write_text(
            json.dumps({"updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "events": events},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        slog("WARNING", f"save events failed: {e}", func="_save_events_local", ticker=ticker)

async def _load_events_cached(ticker: str) -> list[dict] | None:
    rec = _EVENTS_CACHE.get(ticker)
    if rec and (time.time() - rec.get("at", 0) < EVENTS_CACHE_TTL_SEC):
        return rec.get("events")
    return None

def _cache_events(ticker: str, events: list[dict]) -> None:
    _EVENTS_CACHE[ticker] = {"at": time.time(), "events": events}

async def get_fmp_earnings_events(session, ticker: str) -> list[dict]:
    """earnings в окне ±30 дней."""
    try:
        today = datetime.datetime.now(datetime.timezone.utc).date()
        frm = (today - datetime.timedelta(days=7)).isoformat()
        to  = (today + datetime.timedelta(days=30)).isoformat()
        data = await fmp_get_json(session, "earning_calendar",
                                  {"from": frm, "to": to, "symbol": ticker})
        events = []
        for row in (data or []):
            d = row.get("date") or row.get("fiscalDateEnding") or row.get("dateCalendar")
            if d: events.append({"type": "earnings", "date": d, "time": (row.get("time") or "").lower()})
        return events
    except Exception as e:
        slog("WARNING", f"earnings fetch failed: {e}", func="get_fmp_earnings_events", ticker=ticker)
        return []

async def get_fmp_split_events(session, ticker: str) -> list[dict]:
    """splits в окне +60 дней."""
    try:
        today = datetime.datetime.now(datetime.timezone.utc).date()
        frm = (today - datetime.timedelta(days=7)).isoformat()
        to  = (today + datetime.timedelta(days=60)).isoformat()
        data = await fmp_get_json(session, "stock_split_calendar",
                                  {"from": frm, "to": to, "symbol": ticker})
        events = []
        for row in (data or []):
            d = row.get("date") or row.get("splitDate")
            if d: events.append({"type": "split", "date": d})
        return events
    except Exception as e:
        slog("WARNING", f"split fetch failed: {e}", func="get_fmp_split_events", ticker=ticker)
        return []


# нормализатор даты столбцов FMP
def _normalize_dates(df, col="date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if "publishedDate" in df.columns:
        df["publishedDate"] = pd.to_datetime(df["publishedDate"], errors="coerce")
    return df

async def get_insider_net_flow(session, ticker: str, lookback_days: int = 30) -> float:
    """
    Возвращает нетто-поток инсайдеров за окно: USD(buys) - USD(sells).
    Использует FMP:
      - stable/latest-insider-trade?symbol=...
      - (фолбэк) stable/search-insider-trades?symbol=...
    Кэшируем в _FMP_FILTER_CACHE на _FMP_FILTER_TTL сек.
    """
    cache_key = f"insider:{ticker}:{lookback_days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return float(cached)

    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=int(lookback_days))

    def _to_float(x, default=0.0):
        try:
            if x is None: return default
            return float(str(x).replace(",", ""))
        except Exception:
            return default

    def _parse_date(d):
        # возможные поля дат в разных ответах
        for k in ("transactionDate", "date", "reportedDate", "filingDate", "fillingDate"):
            v = d.get(k)
            if v:
                try:
                    return pd.to_datetime(v)
                except Exception:
                    pass
        return None

    def _amount_usd(tr):
        # брать готовое value, иначе shares*price
        val = _to_float(tr.get("transactionValue") or tr.get("value"))
        if val == 0.0:
            shares = _to_float(tr.get("securitiesTransacted") or tr.get("transactionShares") or tr.get("shares") or tr.get("quantity") or tr.get("numberOfShares"))
            price  = _to_float(tr.get("transactionPrice") or tr.get("price") or tr.get("averagePrice") or tr.get("avgPrice"))
            val = shares * price
        return val

    def _is_buy(tr):
        t = str(tr.get("transactionType") or tr.get("acquisitionOrDisposition") or tr.get("type") or "").lower()
        # покрываем разные форматы от FMP: 'P - Purchase', 'Purchase', 'Buy', 'A'
        return ("p" in t and "purchase" in t) or ("purchase" in t) or ("buy" in t) or (t.strip() == "a")
    def _is_sell(tr):
        t = str(tr.get("transactionType") or tr.get("acquisitionOrDisposition") or tr.get("type") or "").lower()
        # 'S - Sale', 'Sale', 'Sell', 'D'
        return ("s" in t and "sale" in t) or ("sale" in t) or ("sell" in t) or (t.strip() == "d")

    buys = 0.0
    sells = 0.0

    # 1) основной источник
    try:
        latest = await fmp_get_json(session, "stable/latest-insider-trade", {"symbol": ticker, "limit": 250})
        if isinstance(latest, list):
            for tr in latest:
                dt = _parse_date(tr)
                if dt is None or dt.tzinfo is None:
                    # приводим к naive UTC сравнения
                    pass
                if dt is None or dt.to_pydatetime() < since:
                    continue
                val = _amount_usd(tr)
                if val <= 0:
                    continue
                if _is_buy(tr):
                    buys += val
                elif _is_sell(tr):
                    sells += val
    except Exception as e:
        slog("WARNING", f"latest-insider-trade failed: {e}", ticker=ticker, func="get_insider_net_flow")

    # 2) фолбэк, если первый дал пусто
    if (buys == 0.0 and sells == 0.0):
        try:
            srch = await fmp_get_json(session, "stable/search-insider-trades", {"symbol": ticker, "limit": 250})
            if isinstance(srch, list):
                for tr in srch:
                    dt = _parse_date(tr)
                    if dt is None or dt.to_pydatetime() < since:
                        continue
                    val = _amount_usd(tr)
                    if val <= 0:
                        continue
                    if _is_buy(tr):
                        buys += val
                    elif _is_sell(tr):
                        sells += val
        except Exception as e:
            slog("WARNING", f"search-insider-trades failed: {e}", ticker=ticker, func="get_insider_net_flow")

    net = float(buys - sells)
    _cache_put(cache_key, net)  # ttl = _FMP_FILTER_TTL
    return net

async def get_institutional_flow(session, ticker: str, lookback_quarters: int = 2) -> float:
    """
    Возвращает Δ total_shares между двумя последними отчетными датами 13F:
    delta_shares = total_shares[last_date] - total_shares[prev_date]
    Позитивная дельта = приток, негативная = отток.
    Источник: /api/v4/institutional-ownership/institutional-holders/symbol-ownership
    Кэшируем на ~3 часа.
    """
    cache_key = f"instflow:{ticker}:{lookback_quarters}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return float(cached)

    try:
        # 1) тянем первую страницу без фильтра даты (часто FMP возвращает список по разным датам)
        path = "api/v4/institutional-ownership/institutional-holders/symbol-ownership"
        data = await fmp_get_json(session, path, {"symbol": ticker, "page": 0})
        if not isinstance(data, list) or not data:
            _cache_put(cache_key, 0.0, ttl=3*3600)
            return 0.0

        df = pd.DataFrame(data)
        # ожидаемые поля: 'date' (или 'reportedDate'), 'shares'
        date_col = "date" if "date" in df.columns else ("reportedDate" if "reportedDate" in df.columns else None)
        if date_col is None or "shares" not in df.columns:
            _cache_put(cache_key, 0.0, ttl=3*3600)
            return 0.0

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)

        dates = (
            df.dropna(subset=[date_col])[date_col]
              .drop_duplicates()
              .sort_values()
              .tail(max(lookback_quarters, 2))
              .tolist()
        )
        if len(dates) < 2:
            _cache_put(cache_key, 0.0, ttl=3*3600)
            return 0.0

        last_date, prev_date = dates[-1], dates[-2]
        total_last = float(df.loc[df[date_col] == last_date, "shares"].sum())
        total_prev = float(df.loc[df[date_col] == prev_date, "shares"].sum())
        delta = total_last - total_prev

        _cache_put(cache_key, delta, ttl=3*3600)
        return float(delta)

    except Exception as e:
        slog("WARNING", f"institutional flow failed: {e}", ticker=ticker, func="get_institutional_flow")
        _cache_put(cache_key, 0.0, ttl=3600)  # час кэша на ошибку
        return 0.0
    
async def extended_blackout(session, ticker: str, profile: dict) -> tuple[bool, str | None]:
    """
    Расширенный blackout:
      • 'earnings-confirmed' в окне ± earnings_blackout_days
      • дивиденды (ex-date/record/payment) в окне ± dividend_blackout_days
      • макро-события высокой важности в окне ± econ_blackout_hours
    При срабатывании → (True, reason).
    """
    th = _get_filter_thresholds(profile or {})
    earn_days = int(th.get("earnings_blackout_days", 2))
    div_days = int(th.get("dividend_blackout_days", 1))
    econ_hours = int(th.get("econ_blackout_hours", 2))

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    today = now_utc.date()

    def _dump(reason: str, extra: dict):
        _save_filter_reason_snapshot(ticker, "extended_blackout", {"reason": reason, **extra})

    # === 1) Earnings (confirmed) ===
    try:
        frm = (today - datetime.timedelta(days=earn_days)).isoformat()
        to  = (today + datetime.timedelta(days=earn_days)).isoformat()
        # документация: /api/v3/earnings-calendar-confirmed?from=...&to=...&symbol=...
        earn = await fmp_get_json(session, "earnings-calendar-confirmed", {"from": frm, "to": to, "symbol": ticker})
        if isinstance(earn, list) and earn:
            # если есть совпадение по тикеру и дата попадает в окно — блокируем
            for e in earn:
                d = e.get("date") or e.get("epsCalendarDate") or e.get("fiscalDateEnding")
                if not d:
                    continue
                d = pd.to_datetime(d, errors="coerce")
                if pd.isna(d):
                    continue
                if (today - datetime.timedelta(days=earn_days)) <= d.date() <= (today + datetime.timedelta(days=earn_days)):
                    _dump("earnings_confirmed_window", {"from": frm, "to": to, "event": e})
                    slog("INFO", "Blackout: earnings-confirmed окно", ticker=ticker, func="extended_blackout", date=str(d.date()))
                    return True, "earnings_confirmed"
    except Exception as e:
        slog("WARNING", f"earnings-confirmed check failed: {e}", ticker=ticker, func="extended_blackout")

    # === 2) Dividends (calendar) ===
    try:
        frm = (today - datetime.timedelta(days=div_days)).isoformat()
        to  = (today + datetime.timedelta(days=div_days)).isoformat()
        # сначала пробуем stable/dividends-calendar?symbol=...
        div = await fmp_get_json(session, "stable/dividends-calendar", {"from": frm, "to": to, "symbol": ticker})
        if not isinstance(div, list):
            # фолбэк: api/v3 stock_dividend_calendar
            div = await fmp_get_json(session, "stock_dividend_calendar", {"from": frm, "to": to, "symbol": ticker})
        if isinstance(div, list) and div:
            for r in div:
                # возможные поля дат: 'date', 'exDate', 'recordDate', 'paymentDate'
                for key in ("exDate", "recordDate", "paymentDate", "date"):
                    d = pd.to_datetime(r.get(key), errors="coerce")
                    if pd.isna(d):
                        continue
                    if (today - datetime.timedelta(days=div_days)) <= d.date() <= (today + datetime.timedelta(days=div_days)):
                        _dump("dividend_window", {"from": frm, "to": to, "event": r, "hit": key})
                        slog("INFO", "Blackout: dividend окно", ticker=ticker, func="extended_blackout", date=str(d.date()), kind=key)
                        return True, f"dividend_{key}"
    except Exception as e:
        slog("WARNING", f"dividends-calendar check failed: {e}", ticker=ticker, func="extended_blackout")

    # === 3) Economic calendar (high-impact) ===
    try:
        # Возьмём небольшой диапазон ±1 день для выборки и вручную отфильтруем по времени
        econ_from = (now_utc - datetime.timedelta(days=1)).date().isoformat()
        econ_to   = (now_utc + datetime.timedelta(days=1)).date().isoformat()
        econ = await fmp_get_json(session, "economic_calendar", {"from": econ_from, "to": econ_to})
        if isinstance(econ, list) and econ:
            # список high-impact ключевых событий (можно расширять)
            HI = {
                "cpi", "core cpi", "pce", "core pce", "nonfarm payrolls", "nfp",
                "fomc", "fed interest rate", "rate decision", "ism manufacturing pmi",
                "ism services pmi", "ppi", "gdp", "jolts"
            }
            window_start = now_utc - datetime.timedelta(hours=econ_hours)
            window_end   = now_utc + datetime.timedelta(hours=econ_hours)

            def _parse_dt(row):
                # FMP обычно даёт 'date' и 'time' (локаль US/Eastern). Возьмём дату и время как UTC naive допущение.
                d = pd.to_datetime(row.get("date"), errors="coerce")
                t = row.get("time") or row.get("timeZone") or ""
                # попробуем склеить, если time выглядит как HH:MM
                if isinstance(t, str) and ":" in t:
                    try:
                        hh, mm = t.split(":", 1)
                        d = d + pd.to_timedelta(int(hh), unit="h") + pd.to_timedelta(int(mm[:2]), unit="m")
                    except Exception:
                        pass
                return d

            for ev in econ:
                name = str(ev.get("event") or ev.get("name") or "").lower()
                if not any(key in name for key in HI):
                    continue
                d = _parse_dt(ev)
                if pd.isna(d):
                    continue
                dt = d.to_pydatetime()
                if window_start <= dt <= window_end:
                    _dump("economic_window", {
                        "event": ev,
                        "window_start": window_start.isoformat(),
                        "window_end": window_end.isoformat()
                    })
                    slog("INFO", "Blackout: macro high-impact окно", ticker=ticker, func="extended_blackout", when=str(dt))
                    return True, "macro_high_impact"
    except Exception as e:
        slog("WARNING", f"economic-calendar check failed: {e}", ticker=ticker, func="extended_blackout")

    return False, None

async def get_press_flags(session, ticker: str, lookback_days: int = 3, block_keywords: list[str] | None = None) -> dict:
    """
    Возвращает словарь:
      {"has_red": bool, "hits": [{"keyword": str, "title": str, "published": str}]}
    Источники:
      1) api/v3/press-releases/{ticker}?limit=50
      2) (фолбэк) stable/press-releases?page=0&size=100 — фильтруем по symbol==ticker
    Кэшируем на _FMP_FILTER_TTL.
    """
    cache_key = f"press:{ticker}:{lookback_days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=int(lookback_days))
    hits: list[dict] = []
    has_red = False
    kws = [k.lower() for k in (block_keywords or [])]

    def _scan(items):
        nonlocal has_red
        for it in items:
            sym = str(it.get("symbol") or it.get("ticker") or "").upper()
            if sym and sym != ticker.upper():
                continue
            title = str(it.get("title") or it.get("headline") or "")
            body  = str(it.get("text") or it.get("content") or "")
            txt = f"{title}\n{body}".lower()

            # published date
            pd_raw = it.get("date") or it.get("publishedDate") or it.get("published_time") or it.get("published_at")
            try:
                pd_dt = pd.to_datetime(pd_raw)
            except Exception:
                pd_dt = None
            if pd_dt is None or pd_dt.to_pydatetime() < since:
                continue

            for kw in kws:
                if kw and kw in txt:
                    has_red = True
                    hits.append({"keyword": kw, "title": title, "published": str(pd_dt)})
        return

    # 1) by-symbol endpoint
    try:
        pr_by_symbol = await fmp_get_json(session, f"press-releases/{ticker}", {"limit": 50})
        if isinstance(pr_by_symbol, list):
            _scan(pr_by_symbol)
    except Exception as e:
        slog("WARNING", f"press-releases by symbol failed: {e}", ticker=ticker, func="get_press_flags")

    # 2) fallback to stable feed
    if not hits:
        try:
            feed = await fmp_get_json(session, "stable/press-releases", {"page": 0, "size": 100})
            if isinstance(feed, list):
                _scan(feed)
        except Exception as e:
            slog("WARNING", f"stable press-releases failed: {e}", ticker=ticker, func="get_press_flags")

    out = {"has_red": has_red, "hits": hits}
    _cache_put(cache_key, out)
    return out

async def get_analyst_context(session, ticker: str, lookback_days: int, last_close: float) -> dict:
    """Возвращает контекст по аналитикам: даунгрейды, долю Buy и апсайд таргета."""
    cache_key = f"analyst:{ticker}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    res = {"has_recent_downgrade": False, "buy_pct": 0.0, "pt_upside_pct": 0.0}
    try:
        from_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        grades = await fmp_get_json(session, "stable/grades", {"symbol": ticker, "from": from_date})
        if isinstance(grades, list):
            for g in grades:
                grade = str(g.get("grade", ""))
                if "downgrade" in grade.lower():
                    res["has_recent_downgrade"] = True
                    break

        hist = await fmp_get_json(session, "stable/historical-grades", {"symbol": ticker, "limit": 50})
        if isinstance(hist, list) and hist:
            buy_cnt = 0
            for h in hist:
                g = str(h.get("newGrade", h.get("grade", ""))).lower()
                if g.startswith("buy") or g.startswith("strongbuy"):
                    buy_cnt += 1
            res["buy_pct"] = 100 * buy_cnt / len(hist)

        pt = await fmp_get_json(session, "stable/price-target-consensus", {"symbol": ticker})
        try:
            item = pt[0] if isinstance(pt, list) and pt else pt
            target = float(item.get("priceTarget", item.get("target", item.get("priceTargetAverage", 0))))
            if last_close:
                res["pt_upside_pct"] = (target - float(last_close)) / float(last_close) * 100
        except Exception:
            pass
    except Exception as e:
        slog("WARNING", f"analyst context error: {e}", ticker=ticker, func="get_analyst_context")

    _cache_put(cache_key, res)
    return res

# === Technical Indicators (FMP) интеграция ===
USE_FMP_TECHNICALS = True  # включаем TI с FMP в онлайне; бэктест оставим на pandas_ta
BACKTEST_USE_FMP_TI = True  # <— включи = True, чтобы не было warm-up "хвостов"
FMP_TI_TIMEOUT = 12        # сек, таймаут запросов за TI
FMP_TI_INTERVAL = "5min"   # наш рабочий ТF в FMP TI

FILTER_DEFAULTS = {
    "earnings_blackout_days": 2,        # не торговать за N дней до/после отчёта
    "econ_blackout_hours": 2,           # не торговать N часов вокруг high-impact макро
    "analyst_recent_days": 10,          # учитывать обновления аналитиков за N дней
    "min_consensus_buy_pct": 55,        # минимум % Buy/StrongBuy по historical-grades
    "allow_downgrades": False,          # запрещать торговать, если был downgrade за N дней
    "min_pt_consensus_upside_pct": 8,   # минимум апсайда по price-target-consensus, %
    "insider_lookback_days": 30,        # окно для инсайдерских сделок
    "max_insider_sell_usd": 500000,      # суммарная продажа инсайдеров не больше этой суммы
    "institutional_lookback_quarters": 2,   # сколько последних отчетных дат сравнивать
    "block_on_institutional_outflow": True,  # блокировать при чистом оттоке
    "dividend_blackout_days": 1,
    "press_lookback_days": 3,                 # сколько дней назад смотреть PR
    "press_block_keywords": [                 # «красные» слова/фразы (настраивается в профиле тикера)
        "guidance cut", "sec probe", "sec investigation",
        "fda hold", "clinical hold", "recall", "restatement",
        "offering", "equity offering", "secondary offering",
        "downgrade", "short report"
    ],
}

# Параметры стратегии


def validate_df(
    df: pd.DataFrame,
    required_cols=("datetime", "open", "high", "low", "close", "volume"),
    ticker: str | None = None,
    tf: str = "5m",
) -> bool:
    """Базовая валидация и очистка данных OHLC."""
    if df is None or df.empty:
        slog("WARNING", "Пустой DataFrame", ticker=ticker, func="validate_df")
        return False
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        slog("WARNING", f"Отсутствуют столбцы {missing}", ticker=ticker, func="validate_df")
        return False
    df.drop_duplicates("datetime", keep="last", inplace=True)
    df.sort_values("datetime", inplace=True)
    num_cols = [c for c in required_cols if c != "datetime"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    nan_ratio = df[num_cols].isna().mean().max()
    if nan_ratio > 0.005:
        slog("WARNING", f"NaN доля {nan_ratio:.3%}", ticker=ticker, func="validate_df")
    df[num_cols] = df[num_cols].fillna({"volume": 0})
    df[num_cols] = df[num_cols].ffill().bfill()
    mask = (
        (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["high"] >= df["low"])
    )
    removed = len(df) - mask.sum()
    if removed > 0:
        slog("WARNING", f"Удалено битых баров: {removed}", ticker=ticker, func="validate_df")
    df.drop(df.index[~mask], inplace=True)
    return True

# === Backtest export utils ===
from pathlib import Path

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def dump_csv(df: pd.DataFrame, path: str | Path) -> str:
    """Пишет CSV; если имя заканчивается на .gz — включаем gzip-сжатие."""
    path = str(path)
    if path.endswith(".gz"):
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)
    return path

def export_bt_bundle(ticker: str,
                     raw_df: pd.DataFrame | None,
                     ds_df: pd.DataFrame | None,
                     trades: list[dict] | None,
                     equity: list[float] | None,
                     out_dir: str | Path) -> dict:
    """
    Пишем четыре файла (по наличию):
      TICKER_raw_bars.csv.gz   — сырые бары (до индикаторов)
      TICKER_dataset.csv.gz    — датасет после add_indicators*
      TICKER_trades.csv        — сделки
      TICKER_equity.csv        — кривая капитала
    Возвращает словарь с путями.
    """
    out = {}
    d = _ensure_dir(out_dir)
    if raw_df is not None and not raw_df.empty:
        out["raw"] = dump_csv(raw_df, d / f"{ticker}_raw_bars.csv.gz")
    if ds_df is not None and not ds_df.empty:
        out["dataset"] = dump_csv(ds_df, d / f"{ticker}_dataset.csv.gz")
    if trades is not None:
        import pandas as _pd
        out["trades"] = dump_csv(_pd.DataFrame(trades), d / f"{ticker}_trades.csv")
    if equity is not None:
        import pandas as _pd
        out["equity"] = dump_csv(_pd.DataFrame({"equity": equity}), d / f"{ticker}_equity.csv")
    return out

def persist_bars(df: pd.DataFrame, ticker: str, tf: str = "5m") -> None:
    """Сохраняем бары на диск: Parquet (если есть движок) или CSV.gz. Без дублей по datetime."""
    try:
        # базовая санация
        if df is None or df.empty:
            return
        if "datetime" not in df.columns:
            return

        base = Path("data") / ticker / tf
        base.mkdir(parents=True, exist_ok=True)
        # разбиваем по дате файла (YYYY-MM-DD)
        df["__date"] = df["datetime"].dt.strftime("%Y-%m-%d")

        for day, chunk in df.groupby("__date"):
            chunk = chunk.drop(columns=["__date"])
            # путь файла
            if PARQUET_ENGINE:
                fpath = base / f"{day}.parquet"
                if fpath.exists():
                    # читаем, конкатим, дедупим
                    old = pd.read_parquet(fpath, engine=PARQUET_ENGINE)
                    merged = pd.concat([old, chunk], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["datetime"]).sort_values("datetime")
                else:
                    merged = chunk.sort_values("datetime")
                merged.to_parquet(fpath, index=False, engine=PARQUET_ENGINE)
            else:
                fpath = base / f"{day}.csv.gz"
                if fpath.exists():
                    old = pd.read_csv(fpath, compression="gzip", parse_dates=["datetime"])
                    merged = pd.concat([old, chunk], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["datetime"]).sort_values("datetime")
                    merged.to_csv(fpath, index=False, compression="gzip")
                else:
                    chunk.sort_values("datetime").to_csv(fpath, index=False, compression="gzip")
    except Exception as e:
        slog("ERROR", f"persist_bars failed: {e}", func="persist_bars", ticker=ticker)


def load_persisted_bars(ticker: str, tf: str = "5m", days: int = 5) -> pd.DataFrame:
    """Грузим последние N дней сохранённых баров. Parquet → CSV.gz (если нет движка)."""
    base = Path("data") / ticker / tf
    if not base.exists():
        return pd.DataFrame()
    # выбираем последние N файлов по имени
    files = sorted(base.glob("*.parquet" if PARQUET_ENGINE else "*.csv.gz"))
    if not files:
        return pd.DataFrame()
    files = files[-days:]  # последние N
    frames = []
    for f in files:
        try:
            if f.suffix == ".parquet":
                frames.append(pd.read_parquet(f, engine=PARQUET_ENGINE))
            else:
                frames.append(pd.read_csv(f, compression="gzip", parse_dates=["datetime"]))
        except Exception as e:
            slog("WARNING", f"skip file {f.name}: {e}", func="load_persisted_bars", ticker=ticker)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # дедуп и сортировка
    if "datetime" in df.columns:
        df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df

def persist_indicators(df: pd.DataFrame, ticker: str, tf: str = "5m") -> None:
    """Сохраняем рассчитанные индикаторы (стандартные колонки из add_indicators*_).
       Пишем посуточно, без дублей по datetime."""
    try:
        if df is None or df.empty or "datetime" not in df.columns:
            return
        # оставим только полезные столбцы
        cols = [c for c in df.columns if c in {
            "datetime","ema9","ema20","ema21","ema50",
            "macd","macd_signal","macd_hist",
            "rsi","rsi_5","adx","atr","vwap","typical_price","body","body_sma20","volume_ma"
        }]
        if len(cols) <= 1:
            return
        data = df[cols].copy()
        base = Path("data") / ticker / tf / "indicators"
        base.mkdir(parents=True, exist_ok=True)
        data["__date"] = data["datetime"].dt.strftime("%Y-%m-%d")
        for day, chunk in data.groupby("__date"):
            chunk = chunk.drop(columns=["__date"]).sort_values("datetime")
            fpath = base / f"{day}.parquet" if PARQUET_ENGINE else base / f"{day}.csv.gz"
            if fpath.exists():
                try:
                    old = (pd.read_parquet(fpath, engine=PARQUET_ENGINE)
                           if fpath.suffix == ".parquet"
                           else pd.read_csv(fpath, compression="gzip", parse_dates=["datetime"]))
                    merged = (pd.concat([old, chunk], ignore_index=True)
                              .drop_duplicates(subset=["datetime"])
                              .sort_values("datetime"))
                except Exception:
                    merged = chunk
            else:
                merged = chunk
            if PARQUET_ENGINE:
                merged.to_parquet(fpath, index=False, engine=PARQUET_ENGINE)
            else:
                merged.to_csv(fpath, index=False, compression="gzip")
    except Exception as e:
        slog("ERROR", f"persist_indicators failed: {e}", func="persist_indicators", ticker=ticker)

REQUEST_DELAY = 300  # 5 минут в секундах
MARKET_CHECK_INTERVAL = 60  # Проверка рынка каждую минуту
DATA_FETCH_DELAY = 120  # Задержка после формирования свечи (2 минуты)

# Тикеры и индивидуальные настройки
def load_tickers_config():
    try:
        with open('tickers_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        log(f"Ошибка загрузки конфига тикеров: {e}. Используем дефолтные.", "ERROR")
        return {
            "AAPL": {"market": "US"}, 
            "SBER.ME": {"market": "RU"}
        }

TICKERS = load_tickers_config()

def rotate_api_key():
    global current_api_key_index
    prev = current_api_key_index
    current_api_key_index = (current_api_key_index + 1) % len(API_KEYS)
    if current_api_key_index != prev:
        slog("WARNING", f"Переключен на API-ключ #{current_api_key_index+1}", func="rotate_api_key")


def load_active_trades() -> Dict[str, Dict]:
    if os.path.exists(ACTIVE_TRADES_FILE):
        try:
            with open(ACTIVE_TRADES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_active_trades(trades: Dict[str, Dict]):
    with open(ACTIVE_TRADES_FILE, "w") as f:
        json.dump(trades, f)


active_trades = load_active_trades()


def register_active_trade(ticker: str, **info):
    info.setdefault("trail_alert_sent", False)
    active_trades[ticker] = info
    save_active_trades(active_trades)
    slog("INFO", "Зарегистрирована сделка", ticker=ticker, func="register_active_trade", **info)


async def send_simple_message_async(session, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}
    try:
        async with session.post(url, data=data) as resp:
            ok = resp.status == 200
            slog(
                "INFO" if ok else "ERROR",
                "Отправка Telegram",
                func="send_simple_message_async",
                status=resp.status,
                payload=redact(json.dumps(data)),
            )
            return ok
    except Exception as e:
        slog("ERROR", f"Ошибка отправки Telegram: {e}", func="send_simple_message_async")
    return False

def fetch_5m_range_sync(session_requests: requests.Session,
                        ticker: str,
                        start_date: str,
                        end_date: str,
                        chunk_days: int = 3,
                        retries: int = 4) -> pd.DataFrame:
    """
    Возвращает 5m бары за [start_date, end_date] включительно.

    Алгоритм:
      1) Пытаемся взять одним запросом по КОЛИЧЕСТВУ баров (timeseries),
         чтобы покрыть нужное окно + запас.
      2) Если охвата не хватает (мин. дата > start_date), добираем "дыры"
         через from/to чанками по 'chunk_days'.

    Почему так: на части кластеров FMP параметр from/to для 5m может
    возвращать урезанный ответ. Параметр timeseries обычно отдаёт
    максимально возможное число баров подряд — им и пользуемся.
    """
    # оценим, сколько баров нам нужно на окно: 78 баров/биржевой день
    try:
        d0 = pd.to_datetime(start_date).date()
        d1 = pd.to_datetime(end_date).date()
        days_total = (d1 - d0).days + 1
    except Exception:
        days_total = 30
    BARS_PER_DAY = 78
    need_bars = int(days_total * BARS_PER_DAY * 1.3)  # +30% запас
    need_bars = max(need_bars, 1200)                  # не меньше 1200 (≈ 2 недель)
    need_bars = min(need_bars, 5000)                  # не больше 5000, чтобы не душить API

    base_ts = f"{BASE_URL_US}/historical-chart/5min/{ticker}"
    # 1) timeseries одним куском
    try:
        r = session_requests.get(base_ts, params={"apikey": API_KEYS[current_api_key_index],
                                                  "timeseries": need_bars}, timeout=20)
        if r.status_code == 429:
            rotate_api_key()
            r = session_requests.get(base_ts, params={"apikey": API_KEYS[current_api_key_index],
                                                      "timeseries": need_bars}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            df = pd.DataFrame(data).rename(columns={"date": "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        else:
            df = pd.DataFrame()
    except Exception as e:
        slog("WARNING", f"5m timeseries fetch failed: {e}", func="fetch_5m_range_sync", ticker=ticker)
        df = pd.DataFrame()

    # если уже хватает охвата — вернём
    if not df.empty:
        if df["datetime"].min() <= pd.to_datetime(start_date) and df["datetime"].max() >= pd.to_datetime(end_date):
            return df

    # 2) fallback: добираем старшие дни чанками from/to
    out = [df] if not df.empty else []
    cur = pd.to_datetime(start_date).date()
    endd = pd.to_datetime(end_date).date()
    while cur <= endd:
        cur_to = min(cur + datetime.timedelta(days=chunk_days - 1), endd)
        q = {"apikey": API_KEYS[current_api_key_index], "from": cur.isoformat(), "to": cur_to.isoformat()}
        ok = False
        for attempt in range(1, retries + 1):
            try:
                r = session_requests.get(base_ts, params=q, timeout=15)
                if r.status_code == 429:
                    rotate_api_key()
                    time.sleep(min(2**attempt, 8))
                    continue
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and data:
                    dfi = pd.DataFrame(data).rename(columns={"date": "datetime"})
                    dfi["datetime"] = pd.to_datetime(dfi["datetime"])
                    out.append(dfi)
                ok = True
                break
            except Exception as e:
                slog("WARNING", f"5m chunk {q['from']}→{q['to']} failed: {e}",
                     func="fetch_5m_range_sync", ticker=ticker, attempt=attempt)
                rotate_api_key()
                time.sleep(min(2**attempt, 8))
        if not ok:
            slog("WARNING", f"skip chunk {q['from']}→{q['to']}", func="fetch_5m_range_sync", ticker=ticker)
        cur = cur_to + datetime.timedelta(days=1)

    if not out:
        return pd.DataFrame()

    all_df = (pd.concat(out, ignore_index=True)
                .sort_values("datetime")
                .drop_duplicates("datetime")
                .reset_index(drop=True))
    return all_df    
    
def get_us_intraday_sync(session, ticker: str, interval: int = 5, limit: int = 100, retries: int = 5) -> pd.DataFrame:
    for _ in range(retries):
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}min/{ticker}?apikey={API_KEYS[current_api_key_index]}"
        slog("DEBUG", "Загружаем данные", ticker=ticker, func="get_us_intraday_sync", url=url)
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 429:
                slog("WARNING", "429 Too Many Requests", ticker=ticker, func="get_us_intraday_sync")
                rotate_api_key()
                time.sleep(1)
                continue
            response.raise_for_status()
            data = response.json()
            if not data or not isinstance(data, list):
                slog("ERROR", "Пустой ответ", ticker=ticker, func="get_us_intraday_sync")
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df["date"])
            df = df.sort_values("datetime").reset_index(drop=True)
            if len(df) > limit:
                df = df.iloc[-limit:]
            if not validate_df(df, ticker=ticker):
                return pd.DataFrame()
            persist_bars(df, ticker)
            return df
        except Exception as e:
            slog("ERROR", f"Ошибка загрузки 5m данных: {e}", ticker=ticker, func="get_us_intraday_sync")
            rotate_api_key()
            time.sleep(1.5)
    slog("CRITICAL", "Все ключи исчерпаны", ticker=ticker, func="get_us_intraday_sync")
    return pd.DataFrame()

async def get_intraday_data(session, ticker: str, market: str, interval: int = 5, days: int = 2) -> pd.DataFrame | None:
    if market != "US":
        return None
    url = f"{BASE_URL_US}/historical-chart/{interval}min/{ticker}?apikey={API_KEYS[current_api_key_index]}"
    for attempt in range(1, 6):  # до 5 попыток
        slog("DEBUG", "Загружаем данные", ticker=ticker, func="get_intraday_data", url=url, attempt=attempt)
        try:
            async with session.get(url, timeout=15) as response:
                if response.status == 429:
                    rotate_api_key()
                    await asyncio.sleep(min(2**attempt, 10))
                    continue
                response.raise_for_status()
                data = await response.json()
                df = pd.DataFrame(data)
                if df.empty:
                    slog("WARNING", "Пустой ответ от API", ticker=ticker, func="get_intraday_data")
                    FMP_FAILS[ticker] += 1
                    break
                df = df.rename(columns={"date":"datetime"})
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime").reset_index(drop=True)
                if not validate_df(df, ticker=ticker):
                    FMP_FAILS[ticker] += 1
                    break
                persist_bars(df, ticker)
                # сбросим счётчик ошибок на успех
                if FMP_FAILS[ticker] > 0:
                    FMP_FAILS[ticker] = 0
                return df
        except Exception as e:
            slog("ERROR", f"Ошибка загрузки US данных (5m): {e}", ticker=ticker, func="get_intraday_data")
            rotate_api_key()
            await asyncio.sleep(min(2**attempt, 10))
    # если дошли сюда — провал
    FMP_FAILS[ticker] += 1
    if FMP_FAILS[ticker] >= FMP_FAIL_THRESHOLD and TELEGRAM_ENABLED:
        # отправим алерт ОДИН раз и сразу обнулим на будущее окно
        await send_simple_message_async(session, f"⚠️ {ticker}: {FMP_FAILS[ticker]} неудачных загрузок подряд (FMP)")
        FMP_FAILS[ticker] = 0
    return None

async def get_us_chart(session, ticker: str, chart_tf: str = "1hour", limit: int = 400) -> pd.DataFrame:
    """
    chart_tf: '1min'|'5min'|'15min'|'30min'|'1hour'
    """
    url = f"{BASE_URL_US}/historical-chart/{chart_tf}/{ticker}?apikey={API_KEYS[current_api_key_index]}"
    try:
        async with session.get(url, timeout=15) as r:
            if r.status == 429:
                rotate_api_key()
                return await get_us_chart(session, ticker, chart_tf, limit)
            r.raise_for_status()
            raw = await r.json()
            df = pd.DataFrame(raw)
            if df.empty: 
                return pd.DataFrame()
            df = df.rename(columns={"date":"datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").iloc[-limit:].reset_index(drop=True)
            validate_df(df, ticker=ticker, tf=chart_tf)
            persist_bars(df, ticker, tf=chart_tf)  # сохраняем 1h бары
            return df
    except Exception as e:
        slog("ERROR", f"get_us_chart({chart_tf}) failed: {e}", ticker=ticker, func="get_us_chart")
        return pd.DataFrame()


async def get_us_daily(session, ticker: str, limit: int = 400) -> pd.DataFrame:
    """
    Дневки через historical-price-full (надёжней для D1).
    """
    url = f"{BASE_URL_US}/historical-price-full/{ticker}?timeseries={limit}&apikey={API_KEYS[current_api_key_index]}"
    try:
        async with session.get(url, timeout=15) as r:
            if r.status == 429:
                rotate_api_key()
                return await get_us_daily(session, ticker, limit)
            r.raise_for_status()
            j = await r.json()
            data = (j or {}).get("historical", [])
            df = pd.DataFrame(data)
            if df.empty: 
                return pd.DataFrame()
            df = df.rename(columns={"date":"datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            # Приводим к единым именам
            need = {"open","high","low","close","volume","datetime"}
            miss = need - set(df.columns)
            if miss:
                # FMP у daily всегда даёт O/H/L/C/Volume
                pass
            validate_df(df, ticker=ticker, tf="1d")
            persist_bars(df, ticker, tf="1d")
            return df
    except Exception as e:
        slog("ERROR", f"get_us_daily failed: {e}", ticker=ticker, func="get_us_daily")
        return pd.DataFrame()

async def get_mtf_context(session, ticker: str) -> dict:
    """
    Возвращает агрегированный контекст:
      - h1_trend: 'up'|'down'|'flat' по EMA50 vs EMA200 + ADX>20
      - d1_rsi: float, d1_macd_hist: float, d1_adx: float
      - allow_long/allow_short: итоговые флаги по профилю (заполнятся на этапе фильтра).
    """
    # --- КЭШ: если в последние N минут контекст уже считался, берём его ---
    cache_key = f"mtf:{ticker}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # быстрый возврат из кэша

    ctx = {"h1_trend":"flat", "d1_rsi":None, "d1_macd_hist":None, "d1_adx":None}
    try:
        # 1) 1H EMA50/200
        ema50_h1 = await _fmp_ti_call_interval(session, ticker, "1hour", "ema", {"period":50})
        ema200_h1 = await _fmp_ti_call_interval(session, ticker, "1hour", "ema", {"period":200})
        adx_h1 = await _fmp_ti_call_interval(session, ticker, "1hour", "adx", {"period":14})
        if all([x is not None and not x.empty for x in [ema50_h1, ema200_h1, adx_h1]]):
            # найдём имена value столбцов
            def vcol(df, cands=("ema","EMA","value")):
                for c in cands:
                    if c in df.columns: return c
                    if c.upper() in df.columns: return c.upper()
                    if c.lower() in df.columns: return c.lower()
                return None
            e50 = ema50_h1.iloc[-1][vcol(ema50_h1)]
            e200 = ema200_h1.iloc[-1][vcol(ema200_h1)]
            adxv = adx_h1.iloc[-1][vcol(adx_h1, ("adx","ADX","value"))]
            if pd.notna(e50) and pd.notna(e200) and pd.notna(adxv):
                if adxv >= 20:
                    ctx["h1_trend"] = "up" if e50 > e200 else "down"
                else:
                    ctx["h1_trend"] = "flat"

        # 2) D1 RSI/MACD/ADX
        rsi_d1  = await _fmp_ti_call_interval(session, ticker, "daily", "rsi",  {"period":14})
        macd_d1 = await _fmp_ti_call_interval(session, ticker, "daily", "macd", {"fast":12,"slow":26,"signal":9})
        adx_d1  = await _fmp_ti_call_interval(session, ticker, "daily", "adx",  {"period":14})
        if rsi_d1 is not None and not rsi_d1.empty:
            ctx["d1_rsi"] = float(rsi_d1.iloc[-1][[c for c in rsi_d1.columns if c.lower() in ("rsi","value")][0]])
        if macd_d1 is not None and not macd_d1.empty:
            # ищем histogram
            macd_hist_col = [c for c in macd_d1.columns if c.lower() in ("histogram","hist","macd_hist","value")]
            if macd_hist_col:
                ctx["d1_macd_hist"] = float(macd_d1.iloc[-1][macd_hist_col[0]])
        if adx_d1 is not None and not adx_d1.empty:
            ctx["d1_adx"] = float(adx_d1.iloc[-1][[c for c in adx_d1.columns if c.lower() in ("adx","value")][0]])

        # 3) Сохраняем контекст (как JSON снапшот)
        try:
            base = Path("data") / ticker / "ctx"
            base.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            (base / f"mtf_{stamp}.json").write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    except Exception as e:
        slog("ERROR", f"MTF context failed: {e}", ticker=ticker, func="get_mtf_context")

    # --- КЭШ: запоминаем на MTF_CTX_TTL_SEC ---
    _cache_put(cache_key, ctx, ttl=MTF_CTX_TTL_SEC)
    return ctx

async def _fmp_ti_call(session, ticker: str, ti_type: str, params: dict | None = None) -> pd.DataFrame | None:
    """
    Вытягивает один индикатор из FMP TI:
    /technical_indicator/{interval}/{ticker}?type=...&...&apikey=...
    Возвращает DF с ['datetime', <value_col>] где value_col == ti_type (или производное).
    """
    try:
        q = {"type": ti_type, "apikey": API_KEYS[current_api_key_index]}
        if params:
            q.update(params)
        url = f"{BASE_URL_US}/technical_indicator/{FMP_TI_INTERVAL}/{ticker}"
        async with session.get(url, params=q, timeout=FMP_TI_TIMEOUT) as resp:
            if resp.status == 429:
                rotate_api_key()
                return await _fmp_ti_call(session, ticker, ti_type, params)
            resp.raise_for_status()
            data = await resp.json()
            if not isinstance(data, list) or not data:
                slog("WARNING", f"TI пуст ({ti_type})", ticker=ticker, func="_fmp_ti_call")
                return None
            df = pd.DataFrame(data)
            # ожидаемые ключи в ответе: 'date' и столбец с индикатором (обычно по имени ti_type)
            if "date" not in df.columns:
                slog("WARNING", f"TI без поля date ({ti_type})", ticker=ticker, func="_fmp_ti_call")
                return None
            df = df.rename(columns={"date": "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            # найдём столбец со значением индикатора (FMP кладет имя индикатора, иногда с капсом)
            value_cols = [c for c in df.columns if c.lower() in {ti_type.lower(), "value", "signal", "histogram"}]
            # сохраним ВСЁ, что пришло (для macd пригодятся несколько колонок)
            try:
                raw_dir = Path("data") / ticker / FMP_TI_INTERVAL / "ti_raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
                raw_file = raw_dir / f"{ti_type}_{stamp}.json"
                # сохраняем первые ~200 точек, чтобы не распухало
                trimmed = data[-200:] if isinstance(data, list) and len(data) > 200 else data
                with open(raw_file, "w", encoding="utf-8") as f:
                    json.dump(trimmed, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        slog("ERROR", f"FMP TI error ({ti_type}): {e}", ticker=ticker, func="_fmp_ti_call")
        return None
    
def _fmp_ti_call_sync(session_requests: requests.Session, ticker: str, ti_type: str,
                      interval: str = "5min", params: dict | None = None, timeout: int = 12) -> pd.DataFrame | None:
    try:
        q = {"type": ti_type, "apikey": API_KEYS[current_api_key_index]}
        if params:
            q.update(params)
        url = f"{BASE_URL_US}/technical_indicator/{interval}/{ticker}"
        for attempt in range(1, 5):
            r = session_requests.get(url, params=q, timeout=timeout)
            if r.status_code == 429:
                rotate_api_key()
                time.sleep(min(2**attempt, 8))
                continue
            if 500 <= r.status_code < 600:
                time.sleep(min(2**attempt, 8))
                continue
            r.raise_for_status()
            data = r.json()
            break
        else:
            return None
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data).rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        slog("WARNING", f"FMP TI sync error ({ti_type}@{interval}): {e}", func="_fmp_ti_call_sync", ticker=ticker)
        return None

def add_indicators_fmp_sync(session_requests: requests.Session, df: pd.DataFrame,
                            ticker: str, market: str, interval: str = "5min") -> pd.DataFrame | None:
    """Синхронно добавляет к df TI из FMP; если чего-то нет — вернёт None (пусть fallback на pandas_ta)."""
    try:
        if df is None or df.empty:
            return None

        ema9  = _fmp_ti_call_sync(session_requests, ticker, "ema",  interval, {"period": 9})
        ema20 = _fmp_ti_call_sync(session_requests, ticker, "ema",  interval, {"period": 20})
        ema21 = _fmp_ti_call_sync(session_requests, ticker, "ema",  interval, {"period": 21})
        ema50 = _fmp_ti_call_sync(session_requests, ticker, "ema",  interval, {"period": 50})
        rsi14 = _fmp_ti_call_sync(session_requests, ticker, "rsi",  interval, {"period": 14})
        rsi5  = _fmp_ti_call_sync(session_requests, ticker, "rsi",  interval, {"period": 5})
        macd  = _fmp_ti_call_sync(session_requests, ticker, "macd", interval, {"fast":12,"slow":26,"signal":9})
        adx   = _fmp_ti_call_sync(session_requests, ticker, "adx",  interval, {"period": 14})
        atr   = _fmp_ti_call_sync(session_requests, ticker, "atr",  interval, {"period": 14})
        vwap  = _fmp_ti_call_sync(session_requests, ticker, "vwap", interval, None)

        packs = [ema9, ema20, ema21, ema50, rsi14, rsi5, macd, adx, atr, vwap]
        if any(x is None or x.empty for x in packs):
            return None

        def pick(df_, candidates):
            for c in candidates:
                if c in df_.columns: return c
                if c.upper() in df_.columns: return c.upper()
                if c.lower() in df_.columns: return c.lower()
            return None

        def merge_ti(base, ti_df, colmap):
            cols = ["datetime"] + list(colmap.keys())
            return base.merge(ti_df[cols].rename(columns=colmap), on="datetime", how="left")

        out = df.copy()
        out = merge_ti(out, ema9,  { pick(ema9,  ["ema","EMA","value"]): "ema9" })
        out = merge_ti(out, ema20, { pick(ema20, ["ema","EMA","value"]): "ema20" })
        out = merge_ti(out, ema21, { pick(ema21, ["ema","EMA","value"]): "ema21" })
        out = merge_ti(out, ema50, { pick(ema50, ["ema","EMA","value"]): "ema50" })
        out = merge_ti(out, rsi14, { pick(rsi14, ["rsi","RSI","value"]): "rsi" })
        out = merge_ti(out, rsi5,  { pick(rsi5,  ["rsi","RSI","value"]): "rsi_5" })

        macd_col   = pick(macd, ["macd","MACD","value"])
        signal_col = pick(macd, ["signal","MACD_signal","macd_signal","signal_line"])
        hist_col   = pick(macd, ["histogram","hist","macd_hist","value"])  # иногда 'value' = hist
        cmap = {}
        if macd_col:   cmap[macd_col] = "macd"
        if signal_col: cmap[signal_col] = "macd_signal"
        if hist_col:   cmap[hist_col] = "macd_hist"
        out = merge_ti(out, macd, cmap)

        out = merge_ti(out, adx,  { pick(adx,  ["adx","ADX","value"]): "adx" })
        out = merge_ti(out, atr,  { pick(atr,  ["atr","ATR","value"]): "atr" })
        out = merge_ti(out, vwap, { pick(vwap, ["vwap","VWAP","value"]): "vwap" })

        # служебные
        out["typical_price"] = (out["high"] + out["low"] + out["close"]) / 3
        out["body"] = (out["close"] - out["open"]).abs()
        out["body_sma20"] = out["body"].rolling(20).mean()
        if market == "US":
            out["volume_ma"] = out["volume"].rolling(20).mean()

        return out
    except Exception as e:
        slog("ERROR", f"add_indicators_fmp_sync failed: {e}", func="add_indicators_fmp_sync", ticker=ticker)
        return None    
    
async def _fmp_ti_call_interval(session, ticker: str, interval: str, ti_type: str, params: dict | None = None) -> pd.DataFrame | None:
    """То же, что _fmp_ti_call, но с явным interval (e.g. '1hour', 'daily')."""
    try:
        q = {"type": ti_type, "apikey": API_KEYS[current_api_key_index]}
        if params:
            q.update(params)
        url = f"{BASE_URL_US}/technical_indicator/{interval}/{ticker}"
        async with session.get(url, params=q, timeout=FMP_TI_TIMEOUT) as resp:
            if resp.status == 429:
                rotate_api_key()
                return await _fmp_ti_call_interval(session, ticker, interval, ti_type, params)
            resp.raise_for_status()
            data = await resp.json()
            if not isinstance(data, list) or not data:
                slog("WARNING", f"TI пуст ({ti_type}@{interval})", ticker=ticker, func="_fmp_ti_call_interval")
                return None
            df = pd.DataFrame(data).rename(columns={"date":"datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        slog("ERROR", f"FMP TI error ({ti_type}@{interval}): {e}", ticker=ticker, func="_fmp_ti_call_interval")
        return None    
    

async def add_indicators_fmp(session, df: pd.DataFrame, ticker: str, market: str) -> pd.DataFrame | None:
    """
    Пытается достроить к df столбцы индикаторов из FMP TI.
    При любой проблеме возвращает None (чтобы можно было упасть на pandas_ta).
    """
    try:
        if df is None or df.empty:
            return None

        # EMA
        ema9  = await _fmp_ti_call(session, ticker, "ema",  {"period": 9})
        ema20 = await _fmp_ti_call(session, ticker, "ema",  {"period": 20})
        ema21 = await _fmp_ti_call(session, ticker, "ema",  {"period": 21})
        ema50 = await _fmp_ti_call(session, ticker, "ema",  {"period": 50})

        # RSI (14) и короткий RSI (5)
        rsi14 = await _fmp_ti_call(session, ticker, "rsi",  {"period": 14})
        rsi5  = await _fmp_ti_call(session, ticker, "rsi",  {"period": 5})

        # MACD (12,26,9)
        macd  = await _fmp_ti_call(session, ticker, "macd", {"fast": 12, "slow": 26, "signal": 9})

        # ADX(14), ATR(14), VWAP (у FMP обычно без параметров)
        adx   = await _fmp_ti_call(session, ticker, "adx",  {"period": 14})
        atr   = await _fmp_ti_call(session, ticker, "atr",  {"period": 14})
        vwap  = await _fmp_ti_call(session, ticker, "vwap", None)

        # Любого из них может не быть — в таком случае отдадим None и фолбэкнемся на pandas_ta:
        packs = [ema9, ema20, ema21, ema50, rsi14, rsi5, macd, adx, atr, vwap]
        if any(x is None or x.empty for x in packs):
            slog("WARNING", "Часть TI пусты — фолбэк на pandas_ta", ticker=ticker, func="add_indicators_fmp")
            return None

        # Нормализуем имена колонок и мерджим в исходный DF по datetime (inner/left join)
        out = df.copy()

        def _merge_ti(base: pd.DataFrame, ti_df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
            # colmap: {"src_col": "dst_col"} (задаём маппинг исходных имен в целевые)
            cols = ["datetime"] + list(colmap.keys())
            ti_small = ti_df[cols].rename(columns=colmap)
            return base.merge(ti_small, on="datetime", how="left")

        # Попробуем угадать названия value-столбцов от FMP:
        # EMA/RSI/ADX/ATR обычно присылаются с колонкой, совпадающей с названием индикатора
        # MACD — часто даёт несколько колонок: macd, signal, histogram
        # VWAP — колонка 'vwap' или 'VWAP'
        def _find_first(df_, candidates):
            for c in candidates:
                if c in df_.columns: return c
                if c.upper() in df_.columns: return c.upper()
                if c.lower() in df_.columns: return c.lower()
            return None

        out = _merge_ti(out, ema9,  { _find_first(ema9,  ["ema","EMA","value"]): "ema9" })
        out = _merge_ti(out, ema20, { _find_first(ema20, ["ema","EMA","value"]): "ema20" })
        out = _merge_ti(out, ema21, { _find_first(ema21, ["ema","EMA","value"]): "ema21" })
        out = _merge_ti(out, ema50, { _find_first(ema50, ["ema","EMA","value"]): "ema50" })

        out = _merge_ti(out, rsi14, { _find_first(rsi14, ["rsi","RSI","value"]): "rsi" })
        out = _merge_ti(out, rsi5,  { _find_first(rsi5,  ["rsi","RSI","value"]): "rsi_5" })

        macd_col   = _find_first(macd, ["macd","MACD","value"])
        signal_col = _find_first(macd, ["signal","MACD_signal","macd_signal","signal_line"])
        hist_col   = _find_first(macd, ["histogram","hist","macd_hist"])
        cmap = {}
        if macd_col:   cmap[macd_col] = "macd"
        if signal_col: cmap[signal_col] = "macd_signal"
        if hist_col:   cmap[hist_col] = "macd_hist"
        out = _merge_ti(out, macd, cmap)

        out = _merge_ti(out, adx,  { _find_first(adx,  ["adx","ADX","value"]): "adx" })
        out = _merge_ti(out, atr,  { _find_first(atr,  ["atr","ATR","value"]): "atr" })
        out = _merge_ti(out, vwap, { _find_first(vwap, ["vwap","VWAP","value"]): "vwap" })

        # досчитаем служебные:
        out["typical_price"] = (out["high"] + out["low"] + out["close"]) / 3
        out["body"] = (out["close"] - out["open"]).abs()
        out["body_sma20"] = out["body"].rolling(20).mean()
        if market == "US":
            out["volume_ma"] = out["volume"].rolling(20).mean()

        return out
    except Exception as e:
        slog("ERROR", f"add_indicators_fmp failed: {e}", ticker=ticker, func="add_indicators_fmp")
        return None    

async def add_indicators(df: pd.DataFrame, market: str, session=None, ticker: str | None = None) -> pd.DataFrame:
    """
    Унифицированный вход: если включены FMP TI и есть session+ticker — пробуем FMP,
    иначе — локальный pandas_ta.
    """
    if USE_FMP_TECHNICALS and session is not None and ticker:
        enriched = await add_indicators_fmp(session, df, ticker, market)
        if enriched is not None:
            return enriched
        # фолбэк
    return add_indicators_pandas(df, market)

# Технические индикаторы: RSI, MACD, EMA20/50
def add_indicators_pandas(df: pd.DataFrame, market: str) -> pd.DataFrame:
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    df["ema9"] = ta.ema(df["close"], length=9)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema50"] = ta.ema(df["close"], length=50)
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["rsi_5"] = ta.rsi(df["close"], length=5)
    df["adx"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["body"] = (df["close"] - df["open"]).abs()
    df["body_sma20"] = df["body"].rolling(20).mean()
    if market == "US":
        df["volume_ma"] = df["volume"].rolling(20).mean()
    return df

def _dump_candidates_once(rows: list[dict], ticker: str):
    """Пишем всех кандидатов одним файлом candidates.csv, выравнивая схему колонок."""
    if not rows:
        return
    from pathlib import Path
    import pandas as pd

    date_tag = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    out_dir = _ensure_dir(Path("backtests") / date_tag / ticker)
    out_csv = out_dir / "candidates.csv"

    df_new = pd.DataFrame(rows)

    # Если файл уже есть — выровняем колонки объединением множества
    if out_csv.exists():
        try:
            df_old = pd.read_csv(out_csv)
            all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))  # сохраняем порядок
            df_old = df_old.reindex(columns=all_cols)
            df_new = df_new.reindex(columns=all_cols)
            # перезапишем старое + добавим новые
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            # если вдруг не смогли прочитать — просто допишем новые с их схемой
            df_out = df_new
    else:
        df_out = df_new

    df_out.to_csv(out_csv, index=False)
    slog("DEBUG", "Saved candidates (final)", ticker=ticker, func="_dump_candidates_once",
         rows=len(df_new), total=len(df_out), file=str(out_csv.name))

def detect_breakout_retest_signal(df: pd.DataFrame, params: Dict, ticker: str | None = None,
                                  diag_buffer: list[dict] | None = None)-> dict | None:
    """Определяет сигнал стратегии Breakout+Retest."""
    _safe_ticker = str(ticker or params.get("profile") or params.get("symbol") or params.get("ticker") or params.get("name") or "UNKNOWN")
    _p = params.get("profile", params) if isinstance(params, dict) else {}
    DIAG_BASE_COLS = [
        "datetime","stage","ticker","side",
        "close","ema9","ema20","ema21","ema50",
        "rsi_5","macd_hist","adx","atr","vwap",
        "volume","volume_ma","reason"
    ]
    # вытащим часто используемые пороги (с дефолтами, если ключа нет)
    _th = {
        "ema_fast": int(_p.get("ema_fast", 9)),
        "ema_slow": int(_p.get("ema_slow", 20)),
        "rsi_min": float(_p.get("rsi_min", 45)),
        "rsi_max": float(_p.get("rsi_max", 70)),
        "volume_ma": int(_p.get("volume_ma", 20)),
        "volume_mult": float(_p.get("volume_mult", 1.6)),
        "tp_rr": float(_p.get("tp_rr", 2.0)),
        "min_atr_ratio": float(_p.get("min_atr_ratio", 0.0022)),
    }
    required = params.get("consol_lookback", 5) + 5
    if len(df) < required:
        slog(
            "DEBUG",
            "Недостаточно баров",
            ticker=ticker,
            func="detect_breakout_retest_signal",
            have=len(df),
            need=required,
        )
        return None

    N = params.get("consol_lookback", 5)
    impulse_body_sma = params.get("impulse_body_sma", 20)
    impulse_len_mult = params.get("impulse_len_mult", 1.0)
    volume_mult = params.get("volume_mult", 1.0)
    volume_ma_len = params.get("volume_ma", 20)
    ema_filter = params.get("ema_filter", False)
    ema_fast_key = f"ema{params.get('ema_fast', 9)}"
    ema_slow_key = f"ema{params.get('ema_slow', 20)}"
    rsi_min = params.get("rsi_min", 0)
    rsi_max = params.get("rsi_max", 100)
    macd_hist_pos = params.get("macd_hist_pos", False)
    retest_close_above = params.get("retest_close_above", True)
    tp_rr = params.get("tp_rr", 2.0)
    entry_mode = params.get("entry_mode", "break_high")

    # Собираем строки диагностики для CSV
    # === BEGIN PATCH: diagnostics (stable, with safe ticker) ===
    _cand_rows: list[dict] = []

    def _g(df: pd.DataFrame, col: str, idx: int = -1, default=np.nan):
        try:
            return float(df[col].iloc[idx])
        except Exception:
            return default

    def _diag_side(df: pd.DataFrame) -> str:
        try:
            c = _g(df, "close")
            ema50 = _g(df, "ema50")
            macdh = _g(df, "macd_hist", idx=-2)
            if not np.isnan(ema50) and c >= ema50:
                return "LONG"
            if not np.isnan(macdh) and macdh > 0:
                return "LONG"
            return "SHORT"
        except Exception:
            return "UNKNOWN"

    def _cand_add(stage: str, df: pd.DataFrame, reason: str, extra: dict | None = None):
        """Добавляет строку кандидата с базовыми индикаторами и причиной."""
        when = ""
        try:
            when = str(pd.to_datetime(df["datetime"].iloc[-1]))
        except Exception:
            pass

        row = {
            "datetime": when,
            "stage": stage,                  # accepted | rejected
            "ticker": _safe_ticker,          # берём уже нормализованный тикер
            "side": _diag_side(df),

            # базовые поля (стабильная схема)
            "close": _g(df, "close"),
            "ema9":  _g(df, "ema9"),
            "ema20": _g(df, "ema20"),       # оставляем ema20, чтобы не плодить разные заголовки
            "ema21": _g(df, "ema21"),
            "ema50": _g(df, "ema50"),
            "rsi_5": _g(df, "rsi_5", idx=-2),
            "macd_hist": _g(df, "macd_hist", idx=-2),
            "adx": _g(df, "adx"),
            "atr": _g(df, "atr"),
            "vwap": _g(df, "vwap"),
            "volume": _g(df, "volume"),
            "volume_ma": _g(df, "volume_ma"),

            "reason": reason,
        }
        try:
            prof = _p if isinstance(_p, dict) else {}
            row["cfg_rsi_min"]         = prof.get("rsi_min")
            row["cfg_rsi_max"]         = prof.get("rsi_max")
            row["cfg_volume_mult"]     = prof.get("volume_mult")
            row["cfg_impulse_len_mult"]= prof.get("impulse_len_mult")
            row["cfg_min_atr_ratio"]   = prof.get("min_atr_ratio")
            row["cfg_ema_fast"]        = prof.get("ema_fast")
            row["cfg_ema_slow"]        = prof.get("ema_slow")
            row["cfg_entry_mode"]      = prof.get("entry_mode")
            row["cfg_tp_rr"]           = prof.get("tp_rr")
        except Exception:
            pass
        if isinstance(extra, dict):
            for k, v in extra.items():
                try:
                    row[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
                except Exception:
                    row[k] = v
        _cand_rows.append(row)

    def _cand_dump():
        """Пишет candidates.csv; при несовпадении схемы — пишет в candidates_v2.csv."""
        if not _cand_rows:
            return
        from pathlib import Path as _P
        import pandas as _pd
        _date_tag = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        out_dir = _ensure_dir(_P("backtests") / _date_tag / _safe_ticker)
        out_csv = out_dir / "candidates.csv"
        df_out = _pd.DataFrame(_cand_rows)

        try:
            if out_csv.exists():
                with out_csv.open("r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                if set(map(str.strip, header)) != set(df_out.columns):
                    out_csv = out_dir / "candidates_v2.csv"
                    df_out.to_csv(out_csv, index=False)
                else:
                    df_out.to_csv(out_csv, mode="a", header=False, index=False)
            else:
                df_out.to_csv(out_csv, index=False)
            slog("DEBUG", "Saved candidates", ticker=_safe_ticker, func="detect_breakout_retest_signal",
                 rows=len(df_out), file=out_csv.name)
        except Exception as e:
            slog("WARNING", f"candidates dump failed: {e}", ticker=_safe_ticker, func="detect_breakout_retest_signal")

    lookback_df = df.iloc[-(N + 2):-2]
    if len(lookback_df) < N:
        return None
    consol_high = lookback_df["high"].max()
    consol_low = lookback_df["low"].min()

    impulse = df.iloc[-2]
    retest = df.iloc[-1]
    body_sma = df["body"].rolling(impulse_body_sma).mean().iloc[-2]
    volume_ma = df["volume"].rolling(volume_ma_len).mean().iloc[-2]
    if pd.isna(body_sma) or pd.isna(volume_ma):
        return None

    direction = None
    reasons = []
    ctx = {}
    failed = False
    reasons: list[str] = []

    def fail(msg: str, **kw):
        nonlocal failed, reasons
        failed = True
        reasons.append(msg)
        # приложим известные численные поля, если они уже посчитаны
        extra = {"reason": msg}
        for k in ("consol_high","consol_low","entry_mode","tp_rr","level","entry","sl","tp","rr","touch_dist_pct","expected_side"):
            if k in kw and kw[k] is not None:
                extra[k] = kw[k]
        _cand_add("rejected", df, msg, extra)
    # LONG условия
    expected_side = "LONG"
    if (
        impulse["close"] > consol_high
        and impulse["body"] >= impulse_len_mult * body_sma
        and impulse["volume"] >= volume_mult * volume_ma
    ):
        if ema_filter:
            ema_fast = df.get(ema_fast_key, df["ema9"]).iloc[-2]
            ema_slow = df.get(ema_slow_key, df["ema20"]).iloc[-2]
            if not (impulse["close"] > ema_fast > ema_slow):
                fail("EMA фильтр")
        rsi = df["rsi_5"].iloc[-2]
        macd_hist = df["macd_hist"].iloc[-2]
        if not (np.isfinite(rsi) and np.isfinite(macd_hist)):
            fail("NaN индикаторов"); return None
        if not np.isfinite(rsi):
            fail("RSI NaN")
        if not np.isfinite(macd_hist):
            fail("MACD NaN")
        if not (rsi_min <= rsi <= rsi_max):
            fail("RSI out of range", expected_side=expected_side,
                rsi=rsi, rsi_min=rsi_min, rsi_max=rsi_max)
        if macd_hist_pos and macd_hist <= 0:
            fail("MACD histogram not positive", expected_side=expected_side, macd_hist=macd_hist)
        last_close = float(df["close"].iloc[-1])
        tol_bps = 0.0002 * last_close
        tol_atr = 0.02 * float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0  # у тебя колонка называется "atr"
        tol = max(tol_bps, tol_atr)

        retest_open  = float(retest["open"])
        retest_close = float(retest["close"])
        retest_high  = float(retest["high"])
        retest_low   = float(retest["low"])

        # коснулся уровня (LONG)
        touched = (retest_low <= consol_high + tol) and (retest_close >= consol_high - tol)
        if not touched:
            # на сколько «не дотянулись» до уровня
            touch_gap = min(retest_low - (consol_high + tol), (consol_high - tol) - retest_close)
            touch_dist_pct = float(touch_gap) / float(max(1e-9, consol_high)) * 100.0
            touch_dir = "below" if retest_low > consol_high else "above"
            fail("Ретест не коснулся уровня", expected_side=expected_side,
                tol=tol, consol_high=consol_high, touch_dist_pct=touch_dist_pct, touch_dir=touch_dir)

        # форма ретеста: допускаем "уродливую" (pin-like) свечу
        body  = abs(retest_close - retest_open)
        range_ = retest_high - retest_low
        ok_pin = (range_ > 0) and (body / range_ <= 0.25)   # можешь ослабить до 0.5 при желании
        if not ok_pin:
            fail("Свеча ретеста не подходит", expected_side=expected_side,body=body, range=range_, ratio=(body/range_ if range_>0 else None))
        entry = retest["high"] if entry_mode == "break_high" else retest["close"]
        sl = min(retest["low"], consol_high - 0.01)
        if params.get("atr_mult") and "atr" in df.columns:
            sl = min(sl, entry - params["atr_mult"] * df["atr"].iloc[-1])
        tp = entry + tp_rr * (entry - sl)
        if entry <= sl or tp <= entry:
            fail("Некорректные уровни", entry=entry, sl=sl, tp=tp)
        rr = (tp - entry) / (entry - sl)
        reason = (
            f"Импульс x{impulse['body']/body_sma:.2f}, объём x{impulse['volume']/volume_ma:.2f}, "
            f"RSI={rsi:.1f}, MACD_hist={macd_hist:.2f}"
        )
        if failed:
            slog("DEBUG", "Сигнал отклонён", func="detect_breakout_retest_signal", ticker=ticker,
                reason="|".join(reasons))
            if isinstance(diag_buffer, list) and _cand_rows:
                diag_buffer.extend(_cand_rows)
            return None
        sig = {
            "direction": "LONG",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "level": consol_high,
            "reason": reason,
        }
        _cand_add("accepted", df, sig.get("reason", "accepted"), {
            "direction": sig.get("direction"),
            "entry": sig.get("entry"),
            "sl": sig.get("sl"),
            "tp": sig.get("tp"),
            "rr": sig.get("rr"),
            "level": sig.get("level"),
            "expected_side": "LONG",  # или "SHORT" в соответствующей ветке
            "consol_high": consol_high,  # для LONG
        })
        if isinstance(diag_buffer, list) and _cand_rows:
            diag_buffer.extend(_cand_rows)
        return sig

    # SHORT условия
    expected_side = "SHORT"
    if (
        impulse["close"] < consol_low
        and impulse["body"] >= impulse_len_mult * body_sma
        and impulse["volume"] >= volume_mult * volume_ma
    ):
        if ema_filter:
            ema_fast = df.get(ema_fast_key, df["ema9"]).iloc[-2]
            ema_slow = df.get(ema_slow_key, df["ema20"]).iloc[-2]
            if not (impulse["close"] < ema_fast < ema_slow):
                fail("EMA фильтр")
        rsi = df["rsi_5"].iloc[-2]
        macd_hist = df["macd_hist"].iloc[-2]
        if not (np.isfinite(rsi) and np.isfinite(macd_hist)):
            fail("NaN индикаторов"); return None
        if not (rsi_min <= rsi <= rsi_max):
            fail("RSI out of range", expected_side=expected_side,
                    rsi=rsi, rsi_min=rsi_min, rsi_max=rsi_max)
        if macd_hist_pos and macd_hist >= 0:
            fail("MACD histogram not positive", expected_side=expected_side, macd_hist=macd_hist)
        last_close = float(df["close"].iloc[-1])
        tol_bps = 0.0003 * last_close
        tol_atr = 0.02 * float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
        tol = max(tol_bps, tol_atr)

        retest_open  = float(retest["open"])
        retest_close = float(retest["close"])
        retest_high  = float(retest["high"])
        retest_low   = float(retest["low"])

        touched = (retest_high >= consol_low - tol) and (retest_close <= consol_low + tol)
        if not touched:
            # на сколько «не дотянулись» до уровня
            touch_gap = min(retest_low - (consol_high + tol), (consol_high - tol) - retest_close)
            touch_dist_pct = float(touch_gap) / float(max(1e-9, consol_high)) * 100.0
            touch_dir = "below" if retest_low > consol_high else "above"
            fail("Ретест не коснулся уровня", expected_side=expected_side,
                tol=tol, consol_high=consol_high, touch_dist_pct=touch_dist_pct, touch_dir=touch_dir)

        body  = abs(retest_close - retest_open)
        range_ = retest_high - retest_low
        ok_pin = (range_ > 0) and (body / range_ <= 0.25)
        if not ok_pin:
            fail("Свеча ретеста не подходит", expected_side=expected_side, body=body, range=range_, ratio=(body/range_ if range_>0 else None))
        entry = retest["low"] if entry_mode == "break_high" else retest["close"]
        sl = max(retest["high"], consol_low + 0.01)
        if params.get("atr_mult") and "atr" in df.columns:
            sl = max(sl, entry + params["atr_mult"] * df["atr"].iloc[-1])
        tp = entry - tp_rr * (sl - entry)
        if entry >= sl or tp >= entry:
            fail("Некорректные уровни", entry=entry, sl=sl, tp=tp)
        rr = (entry - tp) / (sl - entry)
        reason = (
            f"Импульс x{impulse['body']/body_sma:.2f}, объём x{impulse['volume']/volume_ma:.2f}, "
            f"RSI={rsi:.1f}, MACD_hist={macd_hist:.2f}"
        )
        if failed:
            slog("DEBUG", "Сигнал отклонён", func="detect_breakout_retest_signal", ticker=ticker,
                reason="|".join(reasons))
            if isinstance(diag_buffer, list) and _cand_rows:
                diag_buffer.extend(_cand_rows)
            return None
        sig = {
            "direction": "SHORT",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "level": consol_low,
            "reason": reason,
        }
        slog("INFO", "Найден сигнал", ticker=ticker, func="detect_breakout_retest_signal", **sig)
        _cand_add("accepted", df, sig.get("reason", "accepted"), {
            "direction": sig.get("direction"),
            "entry": sig.get("entry"),
            "sl": sig.get("sl"),
            "tp": sig.get("tp"),
            "rr": sig.get("rr"),
            "level": sig.get("level"),
            "expected_side": "SHORT",  # или "SHORT" в соответствующей ветке
            "consol_low":  consol_low,   # для SHORT
        })
        if isinstance(diag_buffer, list) and _cand_rows:
            diag_buffer.extend(_cand_rows)
        return sig

        # финальная сводка отказа
    if reasons:
        # безопасно добавляем метрики, если они считались
        extra = {
            "rsi": float(rsi) if "rsi" in locals() else None,
            "macd_hist": float(macd_hist) if "macd_hist" in locals() else None,
        }
        if "impulse" in locals():
            try:
                extra["impulse_mult"] = float(impulse["body"] / body_sma)
                extra["volume_mult"]  = float(impulse["volume"] / volume_ma)
            except Exception:
                pass
        extra.update(ctx)

        slog(
            "DEBUG",
            "Сигнал отклонён",
            func="detect_breakout_retest_signal",
            ticker=ticker,
            reason="|".join(reasons),
            **extra,
        )
        return None
    _cand_dump()
    return None

def generate_breakout_signal_payload(ticker: str, sig: Dict, trail_trigger: float) -> Dict:
    message = (
        f"🔔 Breakout+Retest сигнал: {ticker} ({sig['direction']})\n\n"
        f"Вход: {sig['entry']:.2f}\n"
        f"SL:   {sig['sl']:.2f}\n"
        f"TP:   {sig['tp']:.2f}\n"
        f"RR:   1:{sig['rr']:.2f}\n\n"
        f"Условия:\n• {sig['reason']}\n\n"
        f"⚙️ При {trail_trigger}R → трейлинг/частичная фиксация\n"
        f"🕒 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    clean_ticker = ticker.replace('.ME', '')
    button_url = f"https://www.tradingview.com/chart/?symbol={clean_ticker}"
    return {
        "chat_id": CHAT_ID,
        "text": message,
        "reply_markup": json.dumps({
            "inline_keyboard": [[{"text": "📈 Открыть график", "url": button_url}]]
        }),
    }

async def send_telegram_signal_with_button_async(session, payload: Dict):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, timeout=10) as response:
                ok = response.status == 200
                if not ok:
                    text = await response.text()
                slog(
                    "INFO" if ok else "ERROR",
                    "Telegram ответ",
                    func="send_telegram_signal_with_button_async",
                    attempt=attempt + 1,
                    status=response.status,
                    payload=redact(json.dumps(payload)[:1000]),
                )
                if ok:
                    return True
        except Exception as e:
            slog(
                "ERROR",
                f"Ошибка отправки сообщения с кнопкой: {e}",
                func="send_telegram_signal_with_button_async",
                attempt=attempt + 1,
            )
        await asyncio.sleep(2)
    return False

async def monitor_active_trades(session):
    """Проверяет активные сделки и отправляет уведомления по TP/SL/1.5R."""
    while True:
        to_remove = []
        for ticker, trade in list(active_trades.items()):
            df = await get_intraday_data(session, ticker, trade.get("market", "US"), interval=5, days=1)
            if df is None or df.empty:
                continue
            price = df["close"].iloc[-1]
            entry, sl, tp = trade["entry"], trade["sl"], trade["tp"]
            direction = trade["direction"]
            trigger_r = trade.get("trail_trigger_r", 1.5)
            slog(
                "DEBUG",
                "Мониторинг сделки",
                ticker=ticker,
                func="monitor_active_trades",
                price=price,
                entry=entry,
                sl=sl,
                tp=tp,
            )
            if direction == "LONG":
                trigger_price = entry + trigger_r * (entry - sl)
                if not trade.get("trail_alert_sent") and price >= trigger_price:
                    await send_simple_message_async(session, f"✅ {ticker}: достигнут {trigger_r}R")
                    slog(
                        "INFO",
                        "Достигнут уровень трейлинга",
                        ticker=ticker,
                        func="monitor_active_trades",
                        price=price,
                        trigger_price=trigger_price,
                    )
                    trade["trail_alert_sent"] = True
                    save_active_trades(active_trades)
                if price >= tp:
                    await send_simple_message_async(session, f"🎯 {ticker}: достигнут Take Profit")
                    slog("INFO", "TP", ticker=ticker, func="monitor_active_trades", price=price)
                    to_remove.append(ticker)
                elif price <= sl:
                    await send_simple_message_async(session, f"🛑 {ticker}: Stop Loss")
                    slog("INFO", "SL", ticker=ticker, func="monitor_active_trades", price=price)
                    to_remove.append(ticker)
            else:
                trigger_price = entry - trigger_r * (sl - entry)
                if not trade.get("trail_alert_sent") and price <= trigger_price:
                    await send_simple_message_async(session, f"✅ {ticker}: достигнут {trigger_r}R")
                    slog(
                        "INFO",
                        "Достигнут уровень трейлинга",
                        ticker=ticker,
                        func="monitor_active_trades",
                        price=price,
                        trigger_price=trigger_price,
                    )
                    trade["trail_alert_sent"] = True
                    save_active_trades(active_trades)
                if price <= tp:
                    await send_simple_message_async(session, f"🎯 {ticker}: достигнут Take Profit")
                    slog("INFO", "TP", ticker=ticker, func="monitor_active_trades", price=price)
                    to_remove.append(ticker)
                elif price >= sl:
                    await send_simple_message_async(session, f"🛑 {ticker}: Stop Loss")
                    slog("INFO", "SL", ticker=ticker, func="monitor_active_trades", price=price)
                    to_remove.append(ticker)
        for t in to_remove:
            active_trades.pop(t, None)
            save_active_trades(active_trades)
            slog("INFO", "Сделка завершена", ticker=t, func="monitor_active_trades")
        await asyncio.sleep(60)

def should_trade_now(df: pd.DataFrame, market: str, profile: Dict) -> bool:
    trade_hours = profile.get("trade_hours", ["09:35", "16:00"])
    start = datetime.datetime.strptime(trade_hours[0], "%H:%M").time()
    end = datetime.datetime.strptime(trade_hours[1], "%H:%M").time()
    last_bar_time = df["datetime"].iloc[-1].time()
    # Защита: последний бар должен быть не старше DATA_FETCH_DELAY + 1 бар
    last_dt = df["datetime"].iloc[-1]
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    bar_age_sec = (now_utc - (last_dt if last_dt.tzinfo is None else last_dt.tz_convert("UTC").to_pydatetime())).total_seconds()
    if bar_age_sec > (DATA_FETCH_DELAY + 60*5):  # 5 минут для 5m TF
        slog("DEBUG", "Старый последний бар, пропуск", func="should_trade_now", age_sec=bar_age_sec)
        return False
    if not (start <= last_bar_time <= end):
        slog(
            "DEBUG",
            "Вне торгового времени",
            func="should_trade_now",
            start=str(start),
            end=str(end),
            current=str(last_bar_time),
        )
        return False

    atr_ratio = df["atr"].iloc[-1] / df["close"].iloc[-1]
    min_atr = profile.get("min_atr_ratio", 0.0)
    if atr_ratio < min_atr:
        slog(
            "DEBUG",
            "ATR слишком низкий",
            func="should_trade_now",
            atr_ratio=atr_ratio,
            min_atr=min_atr,
        )
        return False
    
    min_adx = profile.get("min_adx", 18)  # по умолчанию 18–20
    adx_val = float(df["adx"].iloc[-1]) if "adx" in df.columns else 0.0
    if adx_val < min_adx:
        slog("DEBUG", "ADX низкий", func="should_trade_now", adx=adx_val, min_adx=min_adx)
        return False

    if "volume_ma" in df.columns and profile.get("volume_mult"):
        vol = df["volume"].iloc[-1]
        vol_ma = df["volume_ma"].iloc[-1]
        if vol < vol_ma * profile.get("volume_mult", 1.0):
            slog(
                "DEBUG",
                "Объем недостаточный",
                func="should_trade_now",
                volume=float(vol),
                volume_ma=float(vol_ma),
                mult=profile.get('volume_mult',1.0),
            )
            return False

    if profile.get("strategy") != "breakout_retest" and "ema21" in df.columns and "ema50" in df.columns:
        ema21 = df["ema21"].iloc[-1]
        ema50 = df["ema50"].iloc[-1]
        if ema21 <= ema50:
            slog(
                "DEBUG",
                "Тренд не подтвержден",
                func="should_trade_now",
                ema21=float(ema21),
                ema50=float(ema50),
            )
            return False
    # === MTF фильтры ===
    mtf = profile.get("mtf", {})
    require_h1_alignment = mtf.get("require_h1_alignment", True)
    require_d1_sanity    = mtf.get("require_d1_sanity", True)

    # Если анализ контекста уже был получен в analyze_ticker и передан в profile
    ctx = profile.get("_mtf_ctx")
    if ctx and (require_h1_alignment or require_d1_sanity):
        # h1 alignment
        if require_h1_alignment:
            # Для стратегии breakout_retest: long только по тренду 'up', short только по 'down'
            h1_trend = ctx.get("h1_trend", "flat")
            if h1_trend == "flat":
                slog("DEBUG", "H1 тренд flat — пропуск", func="should_trade_now", h1_trend=h1_trend)
                return False

        # d1 sanity
        if require_d1_sanity:
            d1_rsi = ctx.get("d1_rsi")
            if d1_rsi is not None:
                # не покупаем при экстремальном перекупе, не шортим при экстремальной перепродаже
                rsi_ovb = mtf.get("d1_rsi_overbought", 75)
                rsi_ovs = mtf.get("d1_rsi_oversold", 25)
                # Ничего не знаем о направлении на этом этапе, поэтому просто фильтр «экстремумов»
                if d1_rsi >= rsi_ovb or d1_rsi <= rsi_ovs:
                    slog("DEBUG", "D1 RSI экстремальный — пропуск", func="should_trade_now", d1_rsi=d1_rsi)
                    return False
    slog("DEBUG", "Базовые фильтры пройдены", func="should_trade_now")
    return True

    
async def _save_events_local(ticker: str, events: list[dict]) -> None:
    try:
        base = Path("data") / ticker / "meta"
        base.mkdir(parents=True, exist_ok=True)
        f = base / "events.json"
        payload = {"updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "events": events}
        f.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        slog("WARNING", f"save events failed: {e}", func="_save_events_local", ticker=ticker)

async def _load_events_cached(ticker: str) -> list[dict] | None:
    rec = _EVENTS_CACHE.get(ticker)
    if rec and (time.time() - rec.get("at", 0) < EVENTS_CACHE_TTL_SEC):
        return rec.get("events")
    return None

def _cache_events(ticker: str, events: list[dict]) -> None:
    _EVENTS_CACHE[ticker] = {"at": time.time(), "events": events}

def _is_date_in_blackout(target: datetime.date, event_date: datetime.date,
                         before: int, after: int) -> bool:
    return (event_date - datetime.timedelta(days=before)) <= target <= (event_date + datetime.timedelta(days=after))

async def is_event_blackout(session, ticker: str, profile: dict | None = None) -> tuple[bool, str | None]:
    """
    True/reason если сегодня попадаем в чёрное окно по событиям (earnings/splits).
    Кэширует результаты и сохраняет на диск для прозрачности.
    """
    # Переопределения из профиля тикера (если заданы)
    eb = int(profile.get("earnings_blackout_before", EARNINGS_BLACKOUT_BEFORE)) if profile else EARNINGS_BLACKOUT_BEFORE
    ea = int(profile.get("earnings_blackout_after",  EARNINGS_BLACKOUT_AFTER))  if profile else EARNINGS_BLACKOUT_AFTER
    sb = int(profile.get("split_blackout_before",    SPLIT_BLACKOUT_BEFORE))    if profile else SPLIT_BLACKOUT_BEFORE
    sa = int(profile.get("split_blackout_after",     SPLIT_BLACKOUT_AFTER))     if profile else SPLIT_BLACKOUT_AFTER

    cached = await _load_events_cached(ticker)
    if cached is None:
        earnings = await get_fmp_earnings_events(session, ticker)
        splits   = await get_fmp_split_events(session, ticker)
        events   = earnings + splits
        _cache_events(ticker, events)
        await _save_events_local(ticker, events)
    else:
        events = cached

    today = datetime.datetime.now(datetime.timezone.utc).date()
    for ev in events:
        try:
            d = datetime.date.fromisoformat(ev["date"][:10])
        except Exception:
            continue
        if ev["type"] == "earnings" and _is_date_in_blackout(today, d, eb, ea):
            return True, f"Blackout: earnings {d} (±{eb}/{ea}d)"
        if ev["type"] == "split"    and _is_date_in_blackout(today, d, sb, sa):
            return True, f"Blackout: split {d} (±{sb}/{sa}d)"
    return False, None    

# Основной цикл анализа
# Основной цикл анализа
async def analyze_ticker(session, ticker: str, market: str):
    slog("INFO", "Анализ тикера", ticker=ticker, func="analyze_ticker")

    # === 1) Загрузка баров (локальный кэш + API) ===
    local_df = load_persisted_bars(ticker, "5m", days=5)
    api_df = await get_intraday_data(session, ticker, market, interval=5, days=2)
    if api_df is not None and not api_df.empty:
        df = pd.concat([local_df, api_df]).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    else:
        df = local_df

    # === 2) Валидация и RAW dump (для бэктеста/прозрачности) ===
    if df is None or df.empty or len(df) < 60 or not validate_df(df, ticker=ticker, tf="5m"):
        slog("WARNING", "Недостаточно/некорректные данные", ticker=ticker, func="analyze_ticker")
        return
    _date_tag = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    _out_dir = Path("backtests") / _date_tag / ticker
    try:
        export_bt_bundle(ticker=ticker, raw_df=df.copy(), ds_df=None, trades=None, equity=None, out_dir=_out_dir)
    except Exception as _e:
        slog("WARNING", f"RAW dump fail: {_e}", ticker=ticker, func="analyze_ticker")

    # === 3) Индикаторы и DATASET dump ===
    df = await add_indicators(df, market, session=session, ticker=ticker)  # FMP TI -> fallback pandas_ta
    persist_indicators(df, ticker, "5m")  # суточные срезы индикаторов (как раньше)
    try:
        export_bt_bundle(ticker=ticker, raw_df=None, ds_df=df.copy(), trades=None, equity=None, out_dir=_out_dir)
    except Exception as _e:
        slog("WARNING", f"DATASET dump fail: {_e}", ticker=ticker, func="analyze_ticker")

    # === 4) Подготовка профиля и MTF-контекст (кэш) ===
    profile = dict(TICKERS.get(ticker, {}).get("profile", {}))
    try:
        mtf_ctx = _cache_get(f"mtf:{ticker}") or await get_mtf_context(session, ticker)
        profile["_mtf_ctx"] = mtf_ctx
        profile["_mtf_ctx_cached_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        slog("DEBUG", "MTF контекст", ticker=ticker, func="analyze_ticker",
             h1_trend=mtf_ctx.get("h1_trend"), d1_rsi=mtf_ctx.get("d1_rsi"),
             d1_macd_hist=mtf_ctx.get("d1_macd_hist"), d1_adx=mtf_ctx.get("d1_adx"))
    except Exception as e:
        slog("WARNING", f"MTF контекст недоступен: {e}", ticker=ticker, func="analyze_ticker")

    # === 5) Событийный blackout (earnings/dividends/macro) — ранний выход ===
    try:
        do_blk, blk_reason = await extended_blackout(session, ticker, profile)
        slog("DEBUG", "Extended blackout check", ticker=ticker, func="analyze_ticker",
             result=bool(do_blk), reason=blk_reason)
        if do_blk:
            return
    except Exception as e:
        slog("WARNING", f"extended_blackout failed: {e}", ticker=ticker, func="analyze_ticker")

    # === 6) Аналитики (grades / historical-grades / price-target-consensus) ===
    try:
        th = _get_filter_thresholds(profile)
        last_close = float(df["close"].iloc[-1])
        analyst = await get_analyst_context(session, ticker=ticker,
                                            lookback_days=int(th.get("analyst_recent_days", 10)),
                                            last_close=last_close)
        slog("DEBUG", "Analyst context", ticker=ticker, func="analyze_ticker",
             has_recent_downgrade=analyst.get("has_recent_downgrade"),
             buy_pct=round(float(analyst.get("buy_pct", 0)), 2),
             pt_upside_pct=round(float(analyst.get("pt_upside_pct", 0)), 2))
        # 6.1 запрет на свежие даунгрейды
        if not bool(th.get("allow_downgrades", False)) and analyst.get("has_recent_downgrade"):
            slog("INFO", "Пропуск: свежий downgrade", ticker=ticker, func="analyze_ticker")
            return
        # 6.2 минимальная доля Buy
        if float(analyst.get("buy_pct", 0.0)) < float(th.get("min_consensus_buy_pct", 55.0)):
            slog("INFO", "Пропуск: низкая доля Buy/StrongBuy", ticker=ticker, func="analyze_ticker",
                 buy_pct=float(analyst.get("buy_pct", 0.0)))
            return
        # 6.3 минимальный апсайд
        if float(analyst.get("pt_upside_pct", 0.0)) < float(th.get("min_pt_consensus_upside_pct", 8.0)):
            slog("INFO", "Пропуск: малый апсайд price-target", ticker=ticker, func="analyze_ticker",
                 pt_upside_pct=float(analyst.get("pt_upside_pct", 0.0)))
            return
        profile["_analyst_ctx"] = analyst
    except Exception as e:
        slog("WARNING", f"Analyst filters failed: {e}", ticker=ticker, func="analyze_ticker")

    # === 7) Инсайдеры (нетто USD за окно) ===
    try:
        th = _get_filter_thresholds(profile)
        lookback = int(th.get("insider_lookback_days", 30))
        net_usd = await get_insider_net_flow(session, ticker, lookback_days=lookback)
        net_sell_usd = max(0.0, -float(net_usd))
        slog("DEBUG", "Insider net flow", ticker=ticker, func="analyze_ticker",
             lookback_days=lookback, net_usd=round(float(net_usd), 2),
             net_sell_usd=round(net_sell_usd, 2), max_sell=float(th.get("max_insider_sell_usd", 500000.0)))
        if net_sell_usd > float(th.get("max_insider_sell_usd", 500000.0)):
            slog("INFO", "Пропуск: сильные инсайдерские продажи", ticker=ticker, func="analyze_ticker")
            return
    except Exception as e:
        slog("WARNING", f"Insider filter failed: {e}", ticker=ticker, func="analyze_ticker")

    # === 8) Институционалы (13F) — дельта акций ===
    try:
        th = _get_filter_thresholds(profile)
        lookback_q = int(th.get("institutional_lookback_quarters", 2))
        inst_delta = await get_institutional_flow(session, ticker, lookback_quarters=lookback_q)
        slog("DEBUG", "Institutional flow", ticker=ticker, func="analyze_ticker",
             lookback_quarters=lookback_q, delta_shares=round(float(inst_delta), 2),
             block_on_outflow=bool(th.get("block_on_institutional_outflow", True)))
        if float(inst_delta) < 0 and bool(th.get("block_on_institutional_outflow", True)):
            slog("INFO", "Пропуск: институциональный отток по 13F", ticker=ticker, func="analyze_ticker")
            return
    except Exception as e:
        slog("WARNING", f"Institutional filter failed: {e}", ticker=ticker, func="analyze_ticker")

    # === 9) Финальный тех-фильтр ===
    if not should_trade_now(df, market, profile):
        slog("DEBUG", "Фильтры не пройдены", ticker=ticker, func="analyze_ticker")
        return

    # === 10) Детект сигнала и действия ===
    if profile.get("strategy") == "breakout_retest":
        sig = detect_breakout_retest_signal(df, profile, ticker=ticker)
        # MTF-направление (жёсткий контроль выравнивания)
        mtf = profile.get("mtf", {})
        align = mtf.get("require_h1_alignment", True)
        if sig and align and "_mtf_ctx" in profile:
            h1_trend = profile["_mtf_ctx"].get("h1_trend", "flat")
            if (sig["direction"] == "LONG" and h1_trend != "up") or (sig["direction"] == "SHORT" and h1_trend != "down"):
                slog("DEBUG", "Сигнал отвергнут MTF направлением", ticker=ticker, func="analyze_ticker",
                     dir=sig["direction"], h1_trend=h1_trend)
                sig = None

        if sig:
            payload = generate_breakout_signal_payload(ticker, sig, profile.get("trail_trigger_r", 1.5))
            sent = await send_telegram_signal_with_button_async(session, payload)
            if sent:
                register_active_trade(
                    ticker, market=market, trail_trigger_r=profile.get("trail_trigger_r", 1.5),
                    opened_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **sig
                )
                export_signal_to_csv(
                    ticker=ticker, signal_text="Breakout+Retest", confidence="n/a",
                    last_price=sig["entry"], take_profit=sig["tp"], stop_loss=sig["sl"],
                    adx_value=(df["adx"].iloc[-1] if "adx" in df.columns else 0),
                    volume=(df["volume"].iloc[-1] if "volume" in df.columns else 0),
                    volume_ma=(df["volume_ma"].iloc[-1] if "volume_ma" in df.columns else 0),
                    signal_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    direction=sig["direction"], entry=sig["entry"], rr=sig["rr"], reason=sig.get("reason"),
                    trail_alert_sent=False,
                )
            return
        else:
            slog("DEBUG", "Breakout+Retest не найден", ticker=ticker, func="analyze_ticker")
            return

async def send_bot_status(session, status: str):
    """Отправляет текущий статус бота в Telegram (если включено)."""
    if not TELEGRAM_ENABLED:
        slog("INFO", f"(mute) Статус: {status}", func="send_bot_status")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    message = f"🤖 Статус бота: {status}\nВремя: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        async with session.post(url, data=data) as response:
            if response.status == 200:
                slog("INFO", "Статус отправлен", func="send_bot_status")
                return True
            else:
                text = await response.text()
                slog("ERROR", f"Ошибка отправки статуса [{response.status}]: {text}", func="send_bot_status")
    except Exception as e:
        slog("ERROR", f"Ошибка отправки статуса бота: {e}", func="send_bot_status")
    return False

async def heartbeat(session, interval_hours=6):
    """Периодически шлёт в Telegram, что бот работает. Накапливает ошибки и шлёт алерт после N подряд."""
    global TELEGRAM_FAILS
    while True:
        ok = await send_bot_status(session, "Бот работает")
        if ok:
            if TELEGRAM_FAILS > 0:
                TELEGRAM_FAILS = 0
        else:
            TELEGRAM_FAILS += 1
            slog("WARNING", f"Сбой отправки статуса (#{TELEGRAM_FAILS})", func="heartbeat")
            if TELEGRAM_FAILS >= TEL_FAIL_THRESHOLD and TELEGRAM_ENABLED:
                # Попробуем «простое» сообщение альтернативно
                await send_simple_message_async(session, f"⚠️ {TELEGRAM_FAILS} ошибок статуса подряд")
                TELEGRAM_FAILS = 0  # сбросим, чтобы не спамить
        await asyncio.sleep(interval_hours * 3600)



def run_selftest():
    """Быстрая проверка окружения и доступов."""
    ok = True
    # a) проверка Telegram
    if not TELEGRAM_TOKEN or not CHAT_ID:
        slog("WARNING", "Telegram токен/чат не настроены", func="selftest")
    else:
        try:
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                              data={"chat_id": CHAT_ID, "text": "✅ Self-test: Telegram OK"})
            slog("INFO", f"Telegram status {r.status_code}", func="selftest")
        except Exception as e:
            slog("ERROR", f"Telegram fail: {e}", func="selftest"); ok = False

    # b) проверка FMP ключей
    if not API_KEYS or API_KEYS == ["demo"]:
        slog("WARNING", "API_KEYS не заданы или demo", func="selftest"); ok = False
    else:
        try:
            r = requests.get(f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={API_KEYS[0]}", timeout=10)
            slog("INFO", f"FMP quote status {r.status_code}", func="selftest")
            if r.status_code != 200:
                ok = False
        except Exception as e:
            slog("ERROR", f"FMP fail: {e}", func="selftest"); ok = False

    # c) подгрузка 5m по одному тикеру
    sess = requests.Session()
    df = get_us_intraday_sync(sess, "AAPL", interval=5, limit=120)
    if df is None or df.empty or any(col not in df.columns for col in ["open","high","low","close","volume","datetime"]):
        slog("ERROR", "Intraday AAPL fail", func="selftest"); ok = False
    else:
        slog("INFO", f"AAPL bars: {len(df)}", func="selftest")

    # d) индикаторы + детектор
    if df is not None and not df.empty:
        df = add_indicators(df, "US")
        sig = detect_breakout_retest_signal(df, TICKERS.get("AAPL",{}).get("profile", {}), ticker="AAPL")
        slog("INFO", f"Detector: {'OK' if sig or sig is None else 'OK'}", func="selftest")

    # e) persist/load
    try:
        persist_bars(df, "AAPL", "5m")
        df2 = load_persisted_bars("AAPL", "5m", days=1)
        slog("INFO", f"Persist/load OK, rows={len(df2)}", func="selftest")
    except Exception as e:
        slog("ERROR", f"Persist/load fail: {e}", func="selftest"); ok = False

    print("\nSELFTEST:", "OK ✅" if ok else "PROBLEMS ⚠️")
    return ok


from dataclasses import dataclass

@dataclass
class BTParams:
    fees_pct: float = 0.0005       # комиссия за сделку (0.05%) per side
    slippage_pct: float = 0.0002   # слиппедж (0.02%) на вход/выход
    partial_at_r: float = 1.5      # частичная фиксация при 1.5R
    partial_frac: float = 0.5      # доля фиксации при достижении триггера
    entry_on_next_bar: bool = True # вход со следующего бара
    min_trades: int = 10

def _fill_price(side: str, intended: float, bar_open: float, slippage_pct: float, prefer_conservative=True):
    """Цена исполнения: с учётом гэпа и слиппеджа."""
    # Для честности: если next open хуже intended, берём худшую из цен
    if side == "LONG":
        px = max(intended, bar_open)
        px *= (1 + slippage_pct)
    else:
        px = min(intended, bar_open)
        px *= (1 - slippage_pct)
    return px

def _exit_bar_fill_long(open_, high, low, tp, sl, slippage_pct):
    """Возврат: ('tp'|'sl'|'none', price_filled) c консервативной логикой очередности."""
    # gap cases
    if open_ >= tp: return "tp", open_ * (1 - slippage_pct)
    if open_ <= sl: return "sl", open_ * (1 + slippage_pct)
    # intrabar touches
    hit_tp = high >= tp
    hit_sl = low <= sl
    if hit_tp and hit_sl:
        # консервативно считаем, что сначала задело SL
        return "sl", sl * (1 + slippage_pct)
    if hit_tp: return "tp", tp * (1 - slippage_pct)
    if hit_sl: return "sl", sl * (1 + slippage_pct)
    return "none", None

def _exit_bar_fill_short(open_, high, low, tp, sl, slippage_pct):
    if open_ <= tp: return "tp", open_ * (1 + slippage_pct)
    if open_ >= sl: return "sl", open_ * (1 - slippage_pct)
    hit_tp = low <= tp
    hit_sl = high >= sl
    if hit_tp and hit_sl:
        return "sl", sl * (1 - slippage_pct)
    if hit_tp: return "tp", tp * (1 + slippage_pct)
    if hit_sl: return "sl", sl * (1 - slippage_pct)
    return "none", None

def _apply_fees_notional(notional: float, fees_pct: float) -> float:
    return abs(notional) * fees_pct

def backtest_breakout_retest(df: pd.DataFrame, profile: dict, bt: BTParams, ticker: str) -> dict:
    """Событийный бэктест нашей стратегии с учётом partial@1.5R → BE, комиссий и слиппеджа."""
    trades = []
    equity = [0.0]  # PnL накопленный в $
    cash = 0.0
    cand_buffer: list[dict] = []   # <— единый буфер кандидатов на весь прогон
    pos = None  # dict(side, entry, sl, tp, qty_frac_remaining, be_moved)
    risk_per_trade_usd = 100.0  # фиксированный риск на сделку в $ (можно вынести в профиль)
    diag_rows: list[dict] = []

    for i in range(len(df) - 1):
        # работаем только на данных "до текущего момента", чтобы не подсматривать в будущее:
        window = df.iloc[:i+1].copy()
        bar_next = df.iloc[i+1]

        if pos is None:
            # ПЕРЕПРАВИЛИ: передаём тикер и "окно" (не весь df), чтобы не было утечки будущей свечи
            sig = detect_breakout_retest_signal(window, profile, ticker=ticker, diag_buffer=cand_buffer)
            if sig:
                if sig["direction"] == "LONG":
                    risk_per_share = max(sig["entry"] - sig["sl"], 1e-6)
                else:
                    risk_per_share = max(sig["sl"] - sig["entry"], 1e-6)
                qty = risk_per_trade_usd / risk_per_share

                entry_fill = _fill_price(
                    "LONG" if sig["direction"] == "LONG" else "SHORT",
                    sig["entry"],
                    bar_next["open"],
                    bt.slippage_pct
                )
                cash -= _apply_fees_notional(entry_fill * qty, bt.fees_pct)
                pos = {
                    "side": sig["direction"],
                    "entry": entry_fill,
                    "sl": sig["sl"],
                    "tp": sig["tp"],
                    "qty": qty,
                    "trail_trigger": profile.get("trail_trigger_r", bt.partial_at_r),
                    "partial_done": False,
                    "be_moved": False,
                }
                continue

        # 2) если позиция открыта — проверяем 1.5R (partial + BE) и TP/SL
        if pos is not None:
            side = pos["side"]
            open_, high, low = bar_next["open"], bar_next["high"], bar_next["low"]
            # уровни
            if side == "LONG":
                R = pos["entry"] - pos["sl"]
                trigger_px = pos["entry"] + pos["trail_trigger"] * R
                # partial @ 1.5R, если ещё не было
                if not pos["partial_done"]:
                    # проверяем касание триггера
                    hit = (open_ >= trigger_px) or (high >= trigger_px)
                    if hit:
                        # частично закрыть
                        exit_px = (open_ if open_ >= trigger_px else trigger_px) * (1 - bt.slippage_pct)
                        realized = (exit_px - pos["entry"]) * pos["qty"] * bt.partial_frac
                        cash += realized
                        cash -= _apply_fees_notional(exit_px * pos["qty"] * bt.partial_frac, bt.fees_pct)
                        pos["qty"] *= (1 - bt.partial_frac)
                        pos["partial_done"] = True
                        # перевод SL в BE
                        pos["sl"] = pos["entry"]
                        pos["be_moved"] = True
                # затем проверяем TP/SL на остаток
                evt, px = _exit_bar_fill_long(open_, high, low, pos["tp"], pos["sl"], bt.slippage_pct)
                if evt != "none":
                    # закрываем остаток
                    realized = (px - pos["entry"]) * pos["qty"]
                    cash += realized
                    cash -= _apply_fees_notional(px * pos["qty"], bt.fees_pct)
                    trades.append({
                        "ticker": ticker,
                        "side": "LONG",
                        "entry": pos["entry"],
                        "exit": px,
                        "sl": pos["sl"],
                        "tp": pos["tp"],
                        "partial_done": pos["partial_done"],
                        "be": pos["be_moved"]
                    })
                    pos = None
            else:
                R = pos["sl"] - pos["entry"]
                trigger_px = pos["entry"] - pos["trail_trigger"] * R
                if not pos["partial_done"]:
                    hit = (open_ <= trigger_px) or (low <= trigger_px)
                    if hit:
                        exit_px = (open_ if open_ <= trigger_px else trigger_px) * (1 + bt.slippage_pct)
                        realized = (pos["entry"] - exit_px) * pos["qty"] * bt.partial_frac
                        cash += realized
                        cash -= _apply_fees_notional(exit_px * pos["qty"] * bt.partial_frac, bt.fees_pct)
                        pos["qty"] *= (1 - bt.partial_frac)
                        pos["partial_done"] = True
                        pos["sl"] = pos["entry"]  # BE
                        pos["be_moved"] = True
                evt, px = _exit_bar_fill_short(open_, high, low, pos["tp"], pos["sl"], bt.slippage_pct)
                if evt != "none":
                    realized = (pos["entry"] - px) * pos["qty"]
                    cash += realized
                    cash -= _apply_fees_notional(px * pos["qty"], bt.fees_pct)
                    trades.append({
                        "ticker": ticker,
                        "side": "SHORT",
                        "entry": pos["entry"],
                        "exit": px,
                        "sl": pos["sl"],
                        "tp": pos["tp"],
                        "partial_done": pos["partial_done"],
                        "be": pos["be_moved"]
                    })
                    pos = None

        equity.append(cash)

    try:
        _dump_candidates_once(cand_buffer, ticker)
    except Exception as e:
        slog("WARNING", f"final candidates dump failed: {e}", ticker=ticker, func="backtest_breakout_retest")

    # ===== метрики =====
    eq = np.array(equity, dtype=float)
    ret_total = cash
    # посчитаем R-результаты и PF/WR
    wins, losses, gross_p, gross_l = 0, 0, 0.0, 0.0
    for t in trades:
        pnl = (t["exit"] - t["entry"]) if t["side"]=="LONG" else (t["entry"] - t["exit"])
        if pnl > 0:
            wins += 1; gross_p += pnl
        else:
            losses += 1; gross_l += abs(pnl)
    n_tr = wins + losses
    pf = (gross_p / gross_l) if gross_l > 0 else float("inf")
    winrate = wins / n_tr if n_tr > 0 else 0.0

    # MaxDD (по equity)
    cummax = np.maximum.accumulate(eq)
    dd = (eq - cummax)
    maxdd_abs = dd.min()
    maxdd_pct = (maxdd_abs / (abs(cummax.max()) + 1e-9)) * 100.0 if cummax.max() != 0 else 0.0

    # CAGR (по календарным дням span)
    if len(df) > 1:
        days_span = max((df["datetime"].iloc[-1] - df["datetime"].iloc[0]).days, 1)
    else:
        days_span = 1
    start_capital = 10000.0
    eq = np.array(equity, dtype=float)
    equity_curve = start_capital + eq

    cummax = np.maximum.accumulate(equity_curve)
    dd = equity_curve - cummax
    maxdd_pct = 100.0 * dd.min() / (cummax.max() + 1e-9)

    end_capital = equity_curve[-1]
    days_span = max((df["datetime"].iloc[-1] - df["datetime"].iloc[0]).days, 1)
    cagr = (end_capital / start_capital) ** (365.0 / days_span) - 1.0

    return {
        "trades": trades,
        "n_trades": n_tr,
        "pf": pf,
        "winrate": winrate,
        "cagr": cagr,
        "maxdd_pct": maxdd_pct,
        "equity": eq.tolist(),
        "gross_profit": gross_p,
        "gross_loss": gross_l,
        "cash": cash,
    }

def format_bt_report(ticker: str, res: dict) -> str:
    return (
        f"\n📊 Бэктест: {ticker}\n"
        f"├─ Сделок: {res['n_trades']}\n"
        f"├─ PF: {res['pf']:.2f} | Win-rate: {res['winrate']*100:.1f}%\n"
        f"├─ CAGR: {res['cagr']*100:.2f}% | MaxDD: {res['maxdd_pct']:.2f}%\n"
        f"└─ PnL: ${res['cash']:.2f}\n"
    )

def prepare_df_for_bt(session, ticker: str, market: str, days: int = 30) -> pd.DataFrame:
    """
    Синхронная подготовка данных под бэктест:
      1) грузим 5m бары за [today-(days+warmup), today] диапазоном (fallback — чанки),
      2) validate + RAW dump (как пришло, до любых фильтров),
      3) фильтр регулярных часов (US/Eastern 09:30–16:00),
      4) индикаторы: FMP TI (если включён BACKTEST_USE_FMP_TI) либо локально через pandas_ta,
         при локальном расчёте — trim_warmup, чтобы убрать «NaN-полку»,
      5) DATASET dump (после индикаторов).
    Возвращает готовый датафрейм для детектора.
    """
    try:
        # 0) диапазон с тёплым стартом (ещё +5 календарных дней)
        end_d   = datetime.datetime.now(datetime.timezone.utc).date()
        start_d = end_d - datetime.timedelta(days=days + 5)

        # 1) загрузка 5m диапазоном (fallback — чанки по 3 дня)
        df = fetch_5m_range_sync(session, ticker, start_d.isoformat(), end_d.isoformat(), chunk_days=3)
        slog("DEBUG", "RAW coverage", ticker=ticker, func="prepare_df_for_bt",
            raw_min=str(df["datetime"].min()), raw_max=str(df["datetime"].max()), rows=len(df))
        if df is None or df.empty or len(df) < 60:
            slog("WARNING", "Нет данных/слишком короткая история для BT", ticker=ticker, func="prepare_df_for_bt")
            return pd.DataFrame()

        # 2) validate + RAW dump (ДО любых фильтров и индикаторов!)
        if not validate_df(df, ticker=ticker, tf="5m"):
            slog("WARNING", "validate_df: не прошло", ticker=ticker, func="prepare_df_for_bt")
            return pd.DataFrame()

        _date_tag = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        _out_dir = Path("backtests") / _date_tag / ticker
        try:
            export_bt_bundle(
                ticker=ticker,
                raw_df=df.copy(),    # как пришло с сервера
                ds_df=None,
                trades=None,
                equity=None,
                out_dir=_out_dir
            )
        except Exception as _e:
            slog("WARNING", f"RAW dump fail: {_e}", ticker=ticker, func="prepare_df_for_bt")

        # 3) фильтр регулярных часов (America/New_York 09:30–16:00, будние)
        df = filter_regular_hours_us(df)
        if df is None or df.empty:
            slog("WARNING", "После фильтра торговых часов — пусто", ticker=ticker, func="prepare_df_for_bt")
            return pd.DataFrame()

        # 4) индикаторы
        use_fmp_ti = bool(globals().get("BACKTEST_USE_FMP_TI", True))
        if use_fmp_ti:
            # TI из API: без warm-up (сервер уже посчитал на всю историю)
            df_ti = add_indicators_fmp_sync(session, df.copy(), ticker, market)
            if df_ti is None or df_ti.empty:
                slog("WARNING", "FMP TI недоступны, fallback на локальный расчёт", ticker=ticker, func="prepare_df_for_bt")
                df = add_indicators_pandas(df, market)
                df = trim_warmup(df)
            else:
                df = df_ti
        else:
            # локальный расчёт индикаторов -> нужен trim_warmup
            df = add_indicators_pandas(df, market)
            df = trim_warmup(df)

        if df is None or df.empty:
            slog("WARNING", "После расчёта индикаторов — пусто", ticker=ticker, func="prepare_df_for_bt")
            return pd.DataFrame()

        # 5) DATASET dump (после индикаторов)
        try:
            export_bt_bundle(
                ticker=ticker,
                raw_df=None,
                ds_df=df.copy(),     # на этом датафрейме будет работать детектор
                trades=None,
                equity=None,
                out_dir=_out_dir
            )
        except Exception as _e:
            slog("WARNING", f"DATASET dump fail: {_e}", ticker=ticker, func="prepare_df_for_bt")

        slog("DEBUG", "BT дата-срез",
             ticker=ticker, func="prepare_df_for_bt",
             rows=len(df),
             start=str(df["datetime"].iloc[0]),
             end=str(df["datetime"].iloc[-1]))

        return df

    except Exception as e:
        slog("ERROR", f"prepare_df_for_bt failed: {e}", ticker=ticker, func="prepare_df_for_bt")
        return pd.DataFrame()

def run_backtest(session, ticker: str, market: str, days: int = 15,
                 fees_pct: float = 0.0005, slippage_pct: float = 0.0002,
                 partial_at_r: float = 1.5, partial_frac: float = 0.5):
    slog("INFO", f"Бэктест {ticker}", ticker=ticker, func="run_backtest")
    df = prepare_df_for_bt(session, ticker, market, days=days)
    if df.empty:
        slog("ERROR", "Недостаточно данных для бэктеста", ticker=ticker, func="run_backtest")
        return None
    slog(
    "INFO",
    "BT дата",
    ticker=ticker,
    func="run_backtest",
    rows=int(len(df)),
    first=str(df["datetime"].iloc[0]),
    last=str(df["datetime"].iloc[-1]),
    o=float(df["open"].iloc[0]),
    c=float(df["close"].iloc[-1]),
    )
    profile = TICKERS[ticker].get("profile", {})
    btparams = BTParams(
        fees_pct=fees_pct,
        slippage_pct=slippage_pct,
        partial_at_r=partial_at_r,
        partial_frac=partial_frac
    )
    res = backtest_breakout_retest(df, profile, btparams, ticker = ticker)
    print(format_bt_report(ticker, res))
    # === NEW: экспорт сделок/кривой
    _date_tag = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    _out_dir = Path("backtests") / _date_tag / ticker
    try:
        export_bt_bundle(
            ticker=ticker,
            raw_df=None,
            ds_df=None,
            trades=res.get("trades", []),
            equity=res.get("equity", []),
            out_dir=_out_dir
        )
        # краткий summary по тикеру — удобно для склейки в общий отчёт
        import pandas as _pd
        _summary_row = _pd.DataFrame([{
            "ticker": ticker,
            "n_trades": res.get("n_trades", 0),
            "pf": res.get("pf", 0.0),
            "winrate_pct": 100.0 * res.get("winrate", 0.0),
            "cagr_pct": 100.0 * res.get("cagr", 0.0),
            "maxdd_pct": res.get("maxdd_pct", 0.0),
            "cash": res.get("cash", 0.0),
            "days": days,
            "fees_pct": fees_pct,
            "slippage_pct": slippage_pct,
        }])
        # appending-friendly путь
        _sum_file = _ensure_dir(_out_dir) / f"{ticker}_summary.csv"
        mode = "a" if _sum_file.exists() else "w"
        header = not _sum_file.exists()
        _summary_row.to_csv(_sum_file, index=False, mode=mode, header=header)
    except Exception as _e:
        slog("WARNING", f"Экспорт сделок/эквити не удался: {_e}", ticker=ticker, func="run_backtest")

    return res

def run_full_backtest(days: int = 15, fees_pct: float = 0.0005, slippage_pct: float = 0.0002):
    session = requests.Session()
    print(f"\n=== Бэктест по всем тикерам (посл. {days} дней) ===\n")
    table = []
    rows_for_csv = []  
    for ticker, info in TICKERS.items():
        profile = info.get("profile", {})
        if profile.get("strategy") != "breakout_retest":
            continue
        market = info["market"]
        try:
            res = run_backtest(session, ticker, market, days=days,
                               fees_pct=fees_pct, slippage_pct=slippage_pct)
            if res:
                table.append([ticker, res["n_trades"], res["pf"], res["winrate"]*100, res["cagr"]*100, res["maxdd_pct"]])
                rows_for_csv.append({
                    "date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                    "ticker": ticker,
                    "n_trades": res["n_trades"],
                    "pf": res["pf"],
                    "winrate_pct": 100.0 * res["winrate"],
                    "cagr_pct": 100.0 * res["cagr"],
                    "maxdd_pct": res["maxdd_pct"],
                    "cash": res["cash"],
                    "days": days,
                    "fees_pct": fees_pct,
                    "slippage_pct": slippage_pct
                })
        except Exception as e:
            print(f"Ошибка бэктеста для {ticker}: {e}")
    if table:
        # простая печать таблички
        hdr = ["Ticker","Trades","PF","Win%","CAGR%","MaxDD%"]
        widths = [max(len(str(x)) for x in col) for col in zip(hdr, *table)]
        rowfmt = " | ".join("{:<" + str(w) + "}" for w in widths)
        print(rowfmt.format(*hdr))
        for row in table:
            print(rowfmt.format(row[0], row[1], f"{row[2]:.2f}", f"{row[3]:.1f}", f"{row[4]:.2f}", f"{row[5]:.2f}"))

    if rows_for_csv:
        from pathlib import Path as _P
        import pandas as _pd
        _date_tag = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        _root = _ensure_dir(_P("backtests") / _date_tag)
        _all_summary = _pd.DataFrame(rows_for_csv)
        dump_csv(_all_summary, _root / "summary.csv")
        print(f"\nSaved summary -> {(_root / 'summary.csv').as_posix()}")


def autotune_params_for_ticker(session, ticker: str, market: str, days: int = 15) -> dict:
    """Грид-поиск по impulse_len_mult, volume_mult, min_atr_ratio.
       Целевая функция: максимальный PF, при n_trades>=BTParams.min_trades.
       В случае равенства — выше WinRate, ниже MaxDD."""
    df = prepare_df_for_bt(session, ticker, market, days=days)
    if df.empty:
        slog("ERROR", "Нет данных для автоподбора", ticker=ticker, func="autotune")
        return {}

    base_profile = dict(TICKERS[ticker].get("profile", {}))
    candidates_impulse = [1.6, 1.7, 1.8, 1.9, 2.0]
    candidates_volume  = [1.3, 1.4, 1.5, 1.6]
    candidates_atr     = [0.0022, 0.0025, 0.0028, 0.0032, 0.0035]

    best = None
    best_res = None
    btparams = BTParams()

    for imp in candidates_impulse:
        for vol in candidates_volume:
            for atr_min in candidates_atr:
                prof = base_profile.copy()
                prof["impulse_len_mult"] = imp
                prof["volume_mult"] = vol
                prof["min_atr_ratio"] = atr_min
                res = backtest_breakout_retest(df, prof, btparams, ticker = ticker)
                if res["n_trades"] < btparams.min_trades:
                    continue
                score = (res["pf"], res["winrate"], -res["maxdd_pct"])  # tuple сравнивается лексикографически
                if best is None or score > best:
                    best, best_res = score, (imp, vol, atr_min, res)

    if best_res is None:
        slog("WARNING", "Не найдено параметров с достаточным числом сделок", ticker=ticker, func="autotune")
        return {}

    imp, vol, atr_min, res = best_res
    slog("INFO", "Лучшие параметры", ticker=ticker, func="autotune",
         impulse_len_mult=imp, volume_mult=vol, min_atr_ratio=atr_min,
         pf=res["pf"], winrate=res["winrate"], maxdd=res["maxdd_pct"])
    return {"impulse_len_mult": imp, "volume_mult": vol, "min_atr_ratio": atr_min, "metrics": res}

def save_autotune_to_config(ticker: str, tuned: dict, config_path="tickers_config.json"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    if ticker not in cfg:
        cfg[ticker] = {"market":"US","profile":{}}
    prof = cfg[ticker].setdefault("profile", {})
    for k in ("impulse_len_mult","volume_mult","min_atr_ratio"):
        if k in tuned:
            prof[k] = tuned[k]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    slog("INFO", "Обновлён конфиг параметрами автоподбора", ticker=ticker, func="save_autotune_to_config")                

# Запуск каждое утро
async def main_loop():
    last_fetch_time = {}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=_SSL_CTX)) as session:
        await send_bot_status(session, "Бот запущен ✅")
        asyncio.create_task(heartbeat(session))
        asyncio.create_task(monitor_active_trades(session))
        try:
            while True:
                now = datetime.datetime.now()

                tasks = []
                for ticker, info in TICKERS.items():
                    market = info.get("market", "US")
                    profile = info.get("profile", {})

                    # опционально: честь allow_trade из конфига
                    if profile and (profile.get("allow_trade", True) is False):
                        slog("DEBUG", "Торговля отключена в профиле", ticker=ticker, func="main_loop")
                        continue

                    last_fetch = last_fetch_time.get(ticker, datetime.datetime.min)
                    if (now - last_fetch).total_seconds() >= REQUEST_DELAY:
                        last_fetch_time[ticker] = now
                        tasks.append(asyncio.create_task(analyze_ticker(session, ticker, market)))

                if tasks:
                    await asyncio.gather(*tasks)

                await asyncio.sleep(MARKET_CHECK_INTERVAL)
        except Exception as e:
            log(f"Критическая ошибка в main_loop: {e}", "CRITICAL")
            await send_bot_status(session, f"Критическая ошибка: {e}")
            raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Торговый бот")
    parser.add_argument("--backtest", type=str, help="Запустить бэктест для тикера (например: AAPL)")
    parser.add_argument("--backtest_all", action="store_true", help="Запустить бэктест по всем тикерам")
    parser.add_argument("--autotune", type=str, help="Автоподбор параметров для тикера (например: AAPL)")
    parser.add_argument("--selftest", action="store_true", help="Проверка окружения и доступов")
    parser.add_argument("--days", type=int, default=15, help="Дней истории для бэктеста/автоподбора")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.autotune:
        t = args.autotune.upper()
        if t not in TICKERS:
            log(f"Тикер {t} не найден в конфигурации", "ERROR")
        else:
            sess = requests.Session()
            tuned = autotune_params_for_ticker(sess, t, TICKERS[t]["market"], days=args.days)
            if tuned:
                print(format_bt_report(t, tuned["metrics"]))
                save_autotune_to_config(t, tuned)
    elif args.backtest_all:
        run_full_backtest(days=args.days)
    elif args.backtest:
        t = args.backtest.upper()
        if t not in TICKERS:
            log(f"Тикер {t} не найден в конфигурации", "ERROR")
        else:
            sess = requests.Session()
            run_backtest(sess, t, TICKERS[t]["market"], days=args.days)
    else:
        log("Бот запущен в режиме реального времени")
        asyncio.run(main_loop())