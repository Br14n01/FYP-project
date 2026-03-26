import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.finnhub_news import fetch_historical_news


class TestFetchTodayNews(unittest.TestCase):
    def test_fetch_today_news_returns_valid_dataframe(self):
        load_dotenv(PROJECT_ROOT / ".env")

        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)

        df = fetch_historical_news(
            "AAPL",
            start_date=today.isoformat(),
            end_date=tomorrow.isoformat(),
            save=False,
        )

        # News volume can legitimately be zero for a specific day, so this is a
        # smoke test that validates the function call and returned schema.
        self.assertIsNotNone(df)

        if not df.empty:
            for column in ["date", "datetime", "headline", "summary", "source", "url", "category"]:
                self.assertIn(column, df.columns)


if __name__ == "__main__":
    unittest.main()
