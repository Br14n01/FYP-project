import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.features import add_indicators, download_price_data


class TestComputeIndicators(unittest.TestCase):
    def test_add_indicators_creates_expected_columns(self):
        df = download_price_data("AAPL", start="2024-01-01")
        df_ta = add_indicators(df)

        expected_columns = [
            "rsi_5",
            "roc_10",
            "mom_10",
            "cci_20",
            "wr_14",
            "macd",
            "sma_5",
            "ema_5",
            "vwma_20",
            "atr_14",
            "obv",
        ]

        for column in expected_columns:
            self.assertIn(column, df_ta.columns)

        non_null_columns = [col for col in expected_columns if df_ta[col].notna().any()]
        self.assertTrue(non_null_columns)


if __name__ == "__main__":
    unittest.main()
