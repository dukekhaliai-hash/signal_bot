# config.py
TELEGRAM_TOKEN = "8625644689:AAFTjTwYssM1H5ORmfztVB5fZHI-iMD05Y4"
TIMEZONE = "UTC+1"           # Default user timezone, adjustable via /timezone
MIN_CONFIDENCE = 80        # Final confidence threshold
RNN_THRESHOLD = 85         # RNN probability required
STRUCTURE_SCORE_THRESHOLD = 70
ATR_MIN = 0.001            # 0.1% of price
ATR_MAX = 0.008            # 0.8%
SPREAD_MAX = 0.015         # 1.5%
MONITORED_PAIRS = [
    "EUR/USD(OTC)",
    "GBP/USD(OTC)",
    "USD/JPY(OTC)",
    "AUD/USD(OTC)",
    "USD/CAD(OTC)"
]

# config.py (additional line)
SUBSCRIBERS_FILE = "subscribers.txt"   # store chat IDs, one per line