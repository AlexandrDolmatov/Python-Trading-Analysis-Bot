"""Offline onboarding checks; uses real project functions and no remote APIs."""
import asyncio
import importlib
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
bot = importlib.import_module('pro_trading_bot')
np, pd = bot.np, bot.pd

checks = []

def passed(name):
    checks.append(name)
    print('PASS:', name)

with tempfile.TemporaryDirectory(prefix='trading-smoke-') as temp:
    os.chdir(temp)
    times = pd.DatetimeIndex([
        day + pd.Timedelta(hours=9, minutes=30 + 5 * bar)
        for day in pd.bdate_range('2026-10-19', periods=4)
        for bar in range(78)
    ])
    i = np.arange(len(times))
    close = 100 + .025 * i + .7 * np.sin(i / 5)
    opening = close + .15 * np.cos(i / 3)
    raw = pd.DataFrame({
        'datetime': times, 'open': opening,
        'high': np.maximum(opening, close) + .5,
        'low': np.minimum(opening, close) - .5,
        'close': close, 'volume': 1000 + 100 * (i % 15),
    })
    dirty = pd.concat([raw, raw.iloc[:2]], ignore_index=True)
    assert bot.validate_df(dirty, ticker='ONBOARDING')
    assert len(dirty) == len(raw)
    assert dirty.datetime.is_monotonic_increasing
    passed('OHLCV validation and deduplication: 312 bars')

    outside = raw.iloc[:1].copy()
    outside['datetime'] = pd.Timestamp('2026-10-17 08:00')
    filtered = bot.filter_regular_hours_us(pd.concat([raw, outside]))
    assert len(filtered) == len(raw)
    passed('US trading-session filtering')

    enriched = asyncio.run(bot.add_indicators(raw.copy(), 'US'))
    indicators = ['ema9', 'ema20', 'ema21', 'ema50', 'macd', 'macd_signal',
                  'macd_hist', 'rsi', 'rsi_5', 'adx', 'atr', 'vwap', 'volume_ma']
    assert np.isfinite(enriched[indicators].tail(100).to_numpy()).all()
    # Floating point operations may produce 100.00000000000001.
    assert enriched.rsi.dropna().between(-1e-10, 100 + 1e-10).all()
    assert (enriched.atr.dropna() > 0).all()
    passed('13 local indicators with pandas_ta')

    assert bot.PARQUET_ENGINE == 'pyarrow'
    bot.persist_bars(raw.copy(), 'ONBOARDING')
    bot.persist_bars(raw.copy(), 'ONBOARDING')
    loaded = bot.load_persisted_bars('ONBOARDING', days=4)
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), raw, check_dtype=False)
    assert len(list(Path('data/ONBOARDING/5m').glob('*.parquet'))) == 4
    passed('Parquet round-trip and repeat-write deduplication')

    # A consolidated price series followed by a breakout, retest, entry and TP.
    scenario = raw.iloc[:100].copy()
    scenario['open'] = 100 + .1 * np.sin(np.arange(100))
    scenario['close'] = scenario['open'] + .05
    scenario['high'], scenario['low'], scenario['volume'] = 101., 99., 1000.
    for offset, values in enumerate([
        (100., 103.5, 99.8, 103., 5000.),
        (101.2, 102., 100.9, 101.3, 1000.),
        (102.1, 102.4, 102., 102.2, 1000.),
        (104., 106., 103., 105., 1000.),
    ]):
        row = scenario.iloc[-1].copy()
        row['datetime'] = scenario.datetime.iloc[-1] + pd.Timedelta(minutes=5)
        row[['open', 'high', 'low', 'close', 'volume']] = values
        scenario.loc[len(scenario)] = row
    scenario = bot.add_indicators_pandas(scenario, 'US')
    diagnostics = []
    signal = bot.detect_breakout_retest_signal(scenario.iloc[:102].copy(), {},
                                               ticker='ONBOARDING', diag_buffer=diagnostics)
    assert signal is not None and signal['direction'] == 'LONG'
    assert signal['sl'] < signal['entry'] < signal['tp']
    assert abs(signal['rr'] - 2.) < 1e-8
    assert any(row['stage'] == 'accepted' for row in diagnostics)
    result = bot.backtest_breakout_retest(scenario.copy(), {}, bot.BTParams(), 'ONBOARDING')
    assert {'trades', 'n_trades', 'cash', 'equity'} <= result.keys()
    assert result['n_trades'] == len(result['trades'])
    assert result['n_trades'] >= 1
    assert np.isfinite(result['equity']).all()
    passed('Signal diagnostics and offline backtest pipeline')

    portfolio = bot.vbt.Portfolio.from_holding(pd.Series([100., 110., 120.]), init_cash=1000.)
    assert abs(float(portfolio.final_value()) - 1200.) < 1e-8
    passed('vectorbt/Numba portfolio computation')

print(f'OFFLINE SMOKE: {len(checks)}/{len(checks)} passed; no external integrations exercised')
