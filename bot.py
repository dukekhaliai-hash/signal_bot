import os
import logging
import random
import datetime
import pytz
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# DUMMY MARKET DATA
# -------------------------------------------------------------------
class MarketData:
    """Simulate live M5 candles and indicators for each pair."""
    def __init__(self, pairs):
        self.pairs = pairs
        self.candles: Dict[str, pd.DataFrame] = {}
        self._init_fake_data()

    def _init_fake_data(self):
        np.random.seed(42)
        base_time = datetime.datetime.now(pytz.utc)
        for pair in self.pairs:
            dates = pd.date_range(end=base_time, periods=200, freq='5T')
            close = np.random.normal(1.1000, 0.001, 200).cumsum() + 1.1000
            high = close + abs(np.random.normal(0, 0.0002, 200))
            low = close - abs(np.random.normal(0, 0.0002, 200))
            open_ = close + np.random.normal(0, 0.0001, 200)
            volume = np.random.exponential(100, 200)
            df = pd.DataFrame({
                'timestamp': dates,
                'open': open_, 'high': high, 'low': low, 'close': close,
                'volume': volume
            })
            self.candles[pair] = df.set_index('timestamp')

    def get_latest_candles(self, pair, n=60):
        return self.candles[pair].iloc[-n:].copy()

    def update_candle(self, pair):
        df = self.candles[pair]
        last_time = df.index[-1]
        new_time = last_time + datetime.timedelta(minutes=5)
        last_close = df['close'].iloc[-1]
        new_close = last_close * (1 + np.random.normal(0, 0.0002))
        new_high = max(last_close, new_close) + abs(np.random.normal(0, 0.0001))
        new_low = min(last_close, new_close) - abs(np.random.normal(0, 0.0001))
        new_open = last_close + np.random.normal(0, 0.0001)
        new_vol = np.random.exponential(100)
        new_row = pd.DataFrame({
            'timestamp': [new_time],
            'open': [new_open], 'high': [new_high], 'low': [new_low],
            'close': [new_close], 'volume': [new_vol]
        }).set_index('timestamp')
        self.candles[pair] = pd.concat([df, new_row])[-200:]


# -------------------------------------------------------------------
# FEATURE ENGINEERING (simplified)
# -------------------------------------------------------------------
def compute_indicators(candles: pd.DataFrame) -> dict:
    close = candles['close'].values
    high = candles['high'].values
    low = candles['low'].values
    sma50 = pd.Series(close).rolling(50).mean().iloc[-1]
    norm_price = (close[-1] - sma50) / sma50
    # ATR(14)
    tr = np.maximum(high[1:] - low[1:], abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1]))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.std(close)
    atr_ratio = atr / close[-1]
    # RSI(14)
    delta = np.diff(close[-15:])
    gain = np.mean(delta[delta > 0]) if any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if any(delta < 0) else 1e-10
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Bollinger %B
    rolling_std = pd.Series(close).rolling(20).std().iloc[-1]
    rolling_mean = pd.Series(close).rolling(20).mean().iloc[-1]
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    bb_percent = (close[-1] - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
    # spread
    spread = (high[-1] - low[-1]) / close[-1]
    return {
        'norm_price': norm_price,
        'atr_ratio': atr_ratio,
        'rsi': rsi,
        'bb_percent': bb_percent,
        'spread': spread,
        'close': close[-1]
    }


# -------------------------------------------------------------------
# DUMMY MODELS
# -------------------------------------------------------------------
class DummyRNN:
    def predict(self, features: dict) -> dict:
        up_prob = random.random()
        down_prob = 1.0 - up_prob if up_prob > 0.5 else random.random()
        if random.random() < 0.2:  # occasional strong signal
            up_prob = 0.88 + random.random() * 0.1
            down_prob = 1.0 - up_prob
        direction = 'UP' if up_prob >= down_prob else 'DOWN'
        flip_delay = random.randint(5, 30)
        return {
            'up_prob': up_prob,
            'down_prob': down_prob,
            'direction': direction,
            'flip_delay': flip_delay
        }


class DummySentiment:
    def get_sentiment(self, pair) -> float:
        return random.uniform(-1, 1)


# -------------------------------------------------------------------
# PAIR SCANNER
# -------------------------------------------------------------------
class PairScanner:
    def __init__(self, market_data: MarketData):
        self.market_data = market_data

    def score_pair(self, pair) -> float:
        candles = self.market_data.get_latest_candles(pair, 60)
        ind = compute_indicators(candles)
        if ind['atr_ratio'] < ATR_MIN or ind['atr_ratio'] > ATR_MAX:
            return 0.0
        if ind['spread'] > SPREAD_MAX:
            return 0.0
        high = candles['high'].values
        low = candles['low'].values
        hh = (high[-10] == max(high[-20:]))
        ll = (low[-10] == min(low[-20:]))
        rsi_divergence = 10 if (ind['rsi'] > 70 or ind['rsi'] < 30) else 0
        bb_width = (candles['close'].rolling(20).std().iloc[-1]) / candles['close'].rolling(20).mean().iloc[-1]
        bb_squeeze = 20 if bb_width < 0.002 else 0
        score = 50 + (10 if hh else 0) + (10 if ll else 0) + rsi_divergence + bb_squeeze
        return min(score, 100)

    def scan(self) -> List[str]:
        candidates = []
        for pair in MONITORED_PAIRS:
            score = self.score_pair(pair)
            if score >= STRUCTURE_SCORE_THRESHOLD:
                candidates.append(pair)
                logger.info(f"Pair {pair} score: {score:.0f} (candidate)")
            else:
                logger.info(f"Pair {pair} score: {score:.0f} (skipped)")
        return candidates


# -------------------------------------------------------------------
# SIGNAL DATA STRUCTURE
# -------------------------------------------------------------------
@dataclass
class Signal:
    pair: str
    direction: str
    trade_at: datetime.datetime
    accuracy: int
    created_at: datetime.datetime


# -------------------------------------------------------------------
# SIGNAL FUSION
# -------------------------------------------------------------------
class SignalFusion:
    def __init__(self, rnn_model, sentiment_model):
        self.rnn = rnn_model
        self.sentiment = sentiment_model

    def generate(self, pair, features: dict) -> Optional[Signal]:
        rnn_out = self.rnn.predict(features)
        if rnn_out['up_prob'] < RNN_THRESHOLD / 100 and rnn_out['down_prob'] < RNN_THRESHOLD / 100:
            return None

        sentiment_score = self.sentiment.get_sentiment(pair)
        rnn_prob = rnn_out['up_prob'] if rnn_out['direction'] == 'UP' else rnn_out['down_prob']
        final_conf = 0.6 * rnn_prob + 0.4 * ((sentiment_score + 1) / 2)
        if final_conf * 100 < MIN_CONFIDENCE:
            return None

        if rnn_out['direction'] == 'UP' and sentiment_score > 0:
            direction = 'BUY'
        elif rnn_out['direction'] == 'DOWN' and sentiment_score < 0:
            direction = 'SELL'
        else:
            return None

        now = datetime.datetime.now(pytz.utc)
        minutes_to_add = 5 - (now.minute % 5)
        candle_open = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=minutes_to_add)
        trade_at = candle_open + datetime.timedelta(seconds=rnn_out['flip_delay'])
        accuracy = int(final_conf * 100)

        return Signal(
            pair=pair,
            direction=direction,
            trade_at=trade_at,
            accuracy=accuracy,
            created_at=now
        )


# -------------------------------------------------------------------
# TELEGRAM SIGNAL BOT (everything merged)
# -------------------------------------------------------------------
class SignalBot:
    def __init__(self, token):
        self.app = ApplicationBuilder().token(token).build()
        self.scheduler = BackgroundScheduler(timezone=pytz.utc)
        self.scheduler.start()
        self.market_data = MarketData(MONITORED_PAIRS)
        self.scanner = PairScanner(self.market_data)
        self.rnn = DummyRNN()
        self.sentiment = DummySentiment()
        self.fusion = SignalFusion(self.rnn, self.sentiment)

        self.paused = False
        self.user_timezone = pytz.timezone(TIMEZONE)
        self.recent_signals = []

        # ---- Subscriber management (file-based) ----
        self.subscribers_file = SUBSCRIBERS_FILE
        self.subscribers = self._load_subscribers()

        # ---- Register commands ----
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("pairs", self.pairs_cmd))
        self.app.add_handler(CommandHandler("recent", self.recent))
        self.app.add_handler(CommandHandler("accuracy", self.accuracy))
        self.app.add_handler(CommandHandler("pause", self.pause))
        self.app.add_handler(CommandHandler("resume", self.resume))
        self.app.add_handler(CommandHandler("timezone", self.set_timezone))
        self.app.add_handler(CommandHandler("unsubscribe", self.unsubscribe))

        # ---- Schedule jobs ----
        self.scheduler.add_job(self.update_candles, 'interval', minutes=5)
        self.scheduler.add_job(self.scan_and_generate, 'interval', seconds=30)

    # ---- Subscriber file helpers ----
    def _load_subscribers(self):
        if not os.path.exists(self.subscribers_file):
            return set()
        with open(self.subscribers_file, 'r') as f:
            return set(line.strip() for line in f if line.strip().isdigit())

    def _save_subscribers(self):
        with open(self.subscribers_file, 'w') as f:
            for chat_id in self.subscribers:
                f.write(f"{chat_id}\n")

    def _add_subscriber(self, chat_id: int):
        chat_id_str = str(chat_id)
        if chat_id_str not in self.subscribers:
            self.subscribers.add(chat_id_str)
            self._save_subscribers()
            logger.info(f"New subscriber added: {chat_id}")

    # ---- Telegram command handlers ----
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        self._add_subscriber(chat_id)
        await update.message.reply_text(
            "Pocket Options Signal Bot (Manual Alerts Only)\n"
            "You are now subscribed to signals.\n"
            "Commands: /status, /pairs, /recent, /accuracy, /pause, /resume, /timezone <zone>, /unsubscribe"
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = f"Monitored pairs: {len(MONITORED_PAIRS)}\n"
        msg += f"Paused: {self.paused}\n"
        msg += f"Last signal: {self.recent_signals[-1] if self.recent_signals else 'None'}"
        await update.message.reply_text(msg)

    async def pairs_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        scores = []
        for pair in MONITORED_PAIRS:
            score = self.scanner.score_pair(pair)
            scores.append((pair, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top3 = scores[:3]
        msg = "Top 3 pairs:\n" + "\n".join([f"{p}: {s:.0f}" for p, s in top3])
        await update.message.reply_text(msg)

    async def recent(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.recent_signals:
            msg = "No signals yet."
        else:
            msg = "Last 5 signals:\n"
            for sig in self.recent_signals[-5:]:
                trade_time_local = sig.trade_at.astimezone(self.user_timezone).strftime('%I:%M:%S %p')
                msg += f"{sig.pair} {sig.direction} @ {trade_time_local} ({sig.accuracy}%)\n"
        await update.message.reply_text(msg)

    async def accuracy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Accuracy tracking is not yet implemented. It will be available soon.")

    async def pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.paused = True
        await update.message.reply_text("Alerts paused.")

    async def resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.paused = False
        await update.message.reply_text("Alerts resumed.")

    async def set_timezone(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            tz_name = context.args[0] if context.args else "UTC"
            self.user_timezone = pytz.timezone(tz_name)
            await update.message.reply_text(f"Timezone set to {tz_name}")
        except Exception as e:
            await update.message.reply_text(f"Invalid timezone. Use e.g. US/Eastern. Error: {e}")

    async def unsubscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = str(update.effective_chat.id)
        self.subscribers.discard(chat_id)
        self._save_subscribers()
        await update.message.reply_text("You have been unsubscribed from signals.")

    # ---- Core trading logic ----
    def update_candles(self):
        for pair in MONITORED_PAIRS:
            self.market_data.update_candle(pair)

    def scan_and_generate(self):
        if self.paused:
            return
        # For demo, also update candles to generate movement faster
        for pair in MONITORED_PAIRS:
            self.market_data.update_candle(pair)

        candidates = self.scanner.scan()
        for pair in candidates:
            features = compute_indicators(self.market_data.get_latest_candles(pair, 60))
            signal = self.fusion.generate(pair, features)
            if signal:
                self.schedule_signal(signal)

    def schedule_signal(self, signal: Signal):
        send_time = signal.trade_at - datetime.timedelta(minutes=5)
        logger.info(f"Scheduling signal for {signal.pair} at {send_time} (trade at {signal.trade_at})")
        if send_time <= datetime.datetime.now(pytz.utc):
            # Send immediately via the event loop
            asyncio.run_coroutine_threadsafe(
                self.send_signal(signal), self.app.bot.loop or asyncio.get_event_loop()
            )
        else:
            self.scheduler.add_job(
                lambda: asyncio.run_coroutine_threadsafe(
                    self.send_signal(signal), self.app.bot.loop or asyncio.get_event_loop()
                ),
                DateTrigger(run_date=send_time),
                id=f"signal_{signal.pair}_{signal.trade_at.timestamp()}"
            )

    async def send_signal(self, signal: Signal):
        trade_time_local = signal.trade_at.astimezone(self.user_timezone).strftime('%I:%M:%S %p')
        msg = (
            "POCKET OPTIONS\n\n"
            f"{signal.pair}\n"
            "M5\n"
            f"TRADE AT: {trade_time_local}\n"
            f"{signal.direction}\n\n"
            f"ACCURACY: {signal.accuracy}%"
        )

        # Broadcast to all subscribers
        for chat_id_str in self.subscribers.copy():
            try:
                await self.app.bot.send_message(chat_id=int(chat_id_str), text=msg)
                logger.info(f"Sent signal to {chat_id_str}")
            except Exception as e:
                logger.error(f"Failed to send to {chat_id_str}: {e}")
                if "Forbidden" in str(e) or "deactivated" in str(e):
                    self.subscribers.discard(chat_id_str)
                    self._save_subscribers()

        # Remember last signals
        self.recent_signals.append(signal)
        if len(self.recent_signals) > 5:
            self.recent_signals.pop(0)

    def run(self):
        logger.info("Bot starting...")
        self.app.run_polling()


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    bot = SignalBot(TELEGRAM_TOKEN)
    bot.run()