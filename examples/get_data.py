import yfinance as yf
from datetime import datetime, timedelta

TICKER = 'NVDA'
END_DATE = datetime.today()
# Start date for one year of daily data (needs to be longer for 5m data due to max limits)
# Note: For high-frequency data, yfinance has limitations (often max 60 days for 1-minute, 730 days for 5-minute).
# Let's set the period to a safe 60 days to ensure 5m data is available.
START_DATE = END_DATE - timedelta(days=60) 

# Download the data with the 5-minute interval
nvda_5m_data = yf.Ticker(TICKER).history(
    start=START_DATE, 
    end=END_DATE, 
    # CRUCIAL: Set the interval parameter
    interval='5m' 
)

# Reset index so the DatetimeIndex becomes a column
nvda_5m_data = nvda_5m_data.reset_index()

# Rename columns to your preferred names
nvda_5m_data.columns = [
    'timestamps', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stocksplit'
]

# Rename the index for clarity and save
# nvda_5m_data.index.name = 'timestamps'
FILENAME = f'./data/{TICKER}_5m_Data.csv'
nvda_5m_data.to_csv(FILENAME)

print(f"5-minute data for {TICKER} saved to {FILENAME}.")