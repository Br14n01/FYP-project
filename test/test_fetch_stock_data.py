import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.features import download_price_data


class TestFetchStockData(unittest.TestCase):
    def test_download_price_data_returns_ohlcv(self):
        df = download_price_data("AAPL", start="2024-01-01")

        self.assertFalse(df.empty)
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(column, df.columns)


if __name__ == "__main__":
    unittest.main()
