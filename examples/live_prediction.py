import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, time
import yfinance as yf
import torch
import sys

# If Kronos modules are in a parent directory
sys.path.append("../")

from model import Kronos, KronosTokenizer, KronosPredictor


# def fetch_recent_data(ticker="NVDA", interval="5m", period="5d"):
#     """
#     Fetch latest 5-minute stock data from Yahoo Finance.
#     """
#     df = yf.download(ticker, period=period, interval=interval)
#     df = df.xs(ticker, axis=1, level='Ticker')

#     df.reset_index(inplace=True)
#     df.rename(columns={
#         'Datetime': 'timestamps',
#         'Open': 'open',
#         'High': 'high',
#         'Low': 'low',
#         'Close': 'close',
#         'Volume': 'volume'
#     }, inplace=True)
#     print(f"Fetched {len(df)} rows of data for {ticker}. Latest: {df['timestamps'].iloc[-1]}")
#     return df

def generate_future_timestamps(last_time, pred_len, freq='5min'):
    last_time = pd.Timestamp(last_time)

    # If it's the last candle of the day (15:55)
    if last_time.strftime('%H:%M:%S') == '15:55:00':
        # Move to the next day, same time, keep tz
        next_start = (last_time + timedelta(days=1)).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
    else:
        # Otherwise, just add one interval
        next_start = last_time

    return pd.date_range(start=next_start + timedelta(minutes=5), periods=pred_len, freq=freq)


# def generate_future_timestamps(last_time, pred_len, freq='5min'):
#     """
#     Generate timestamps into the future.
#     """
#     return pd.date_range(start=last_time + timedelta(minutes=5), periods=pred_len, freq=freq)


def plot_forecast(history_df, pred_df):
    """
    Plot historical close price and forecasted future close price.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history_df['timestamps'], history_df['close'],
            label='History', color='blue', linewidth=1.5)
    ax.plot(pred_df.index, pred_df['close'],
            label='Forecast', color='red', linestyle='--', linewidth=1.5)
    
    # --- Add connecting line between last history point and first forecast point ---
    last_time = history_df['timestamps'].iloc[-1]
    last_close = history_df['close'].iloc[-1]
    first_time = pred_df.index[0]
    first_close = pred_df['close'].iloc[0]
    ax.plot([last_time, first_time], [last_close, first_close],
            color='gray', linestyle=':', linewidth=1.2, label='Gap Connector')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Close Price', fontsize=12)
    ax.set_title('NVDA Next Few Hours Forecast (5-minute intervals)', fontsize=14)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # ---- 1. Fetch latest market data ----
    #df = fetch_recent_data("NVDA", interval="5m", period="5d")
    df = pd.read_csv("./data/NVDA_5m_Data.csv")
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # ---- 2. Load model & tokenizer ----
    print("Loading Kronos model and tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # ---- 3. Prepare input & prediction timestamps ----
    lookback = 400        # number of candles used as input
    pred_len = 72         # predict next 6 hours (5m × 72 = 6h)

    start_label = df.index[-lookback]
    x_df = df.loc[start_label:, ['open', 'high', 'low', 'close', 'volume']]

    x_timestamp = df.loc[start_label:, 'timestamps']
    print(x_timestamp)


    last_time = df['timestamps'].iloc[-1]
    y_timestamp = generate_future_timestamps(last_time, pred_len)
    y_timestamp = pd.Series(y_timestamp, name='timestamps')
    print(y_timestamp)

    # ---- 4. Predict future prices ----
    print(f"Predicting next {pred_len * 5} minutes (~6 hours)...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    pred_df.index = y_timestamp

    # ---- 5. Display and plot results ----
    print("\nForecasted Data (head):")
    print(pred_df.head())

    plot_forecast(df.iloc[-lookback:], pred_df)


if __name__ == "__main__":
    main()
