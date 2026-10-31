"""Regression check for the self-test's async indicator call; no network IO."""
import unittest
from unittest.mock import AsyncMock, Mock, patch
import pandas as pd
import pro_trading_bot as bot


class SelftestTests(unittest.TestCase):
    def test_awaits_indicators_before_running_detector(self):
        frame = pd.DataFrame({'datetime': [pd.Timestamp('2024-10-08')],
                              'open': [1.], 'high': [2.], 'low': [.5],
                              'close': [1.5], 'volume': [100.]})
        enriched = frame.assign(ema9=1.5)
        with patch.object(bot, 'TELEGRAM_TOKEN', ''), \
             patch.object(bot, 'API_KEYS', ['unit-test-key']), \
             patch.object(bot.requests, 'get', return_value=Mock(status_code=200)), \
             patch.object(bot, 'get_us_intraday_sync', return_value=frame), \
             patch.object(bot, 'add_indicators', new_callable=AsyncMock, return_value=enriched) as indicators, \
             patch.object(bot, 'detect_breakout_retest_signal', return_value=None) as detector, \
             patch.object(bot, 'persist_bars'), \
             patch.object(bot, 'load_persisted_bars', return_value=frame):
            self.assertTrue(bot.run_selftest())
            indicators.assert_awaited_once()
            self.assertIs(detector.call_args.args[0], enriched)


if __name__ == '__main__':
    unittest.main()
