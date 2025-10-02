import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(format='%(asctime)s  - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_ohlcv(ticker: str, start : str = '2020-01-01',end : str=None,interval: str='1d') -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance(ticker):
    Args:
        ticker (str): ticker symbol
        start (str, optional): start date
        end (str, optional): end date
        interval (str, optional): interval

    Returns:
        pd.DataFrame: OHLCV data
    """

    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    logger.info(f'Fetching OHLCV data from {ticker} from {start} to  {end}')

    try:
        tk = yf.Ticker(ticker)

        df = tk.history(start=start, end=end, interval=interval, auto_adjust=False)

        if df.empty:
            logger.warning(f'No data found for {ticker} from {start} to  {end}')
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits':'splits'
        })

        df['ticker'] = ticker

        initial_rows = len(df)
        df = df.dropna(subset=['open','high','low','close'])

        if len(df) < initial_rows:
            logger.warning(f'Deleted {initial_rows - len(df)} rows due to insufficient data')

        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f'Downloaded {len(df)} rows for ticker {ticker}')

        return df


    except Exception as e:
        logger.error(f'Error while fetching data for ticker {ticker}: {str(e)}')
        return pd.DataFrame()



def fetch_multiple_tickers(tickers: list, **kwargs)-> pd.DataFrame:
    """
    Fetching data for multiple tickers onto one dataframe.
    Args:
        tickers (list): list of tickers
        **kwargs:
    Returns:
     pd.DataFrame: OHLCV data

    """
    logger.info(f'Fetching data for multiple tickers: {tickers}')
    all_data = []

    for ticker in tickers:
        df = fetch_ohlcv(ticker,**kwargs)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        logger.warning("No data found for multiple tickers")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    logger.info(f'Combined data for multiple tickers: {combined}')
    return combined

def save_to_csv(df:pd.DataFrame, path: str):
    """
    Save data to CSV.
    Args:
        df (pd.DataFrame): OHLCV data
        path (str): path to save to

    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f'Saved data to {path}')


def get_ticker_info(ticker: str) ->dict:
    """
    Fetching metadata about ticker (sector, industry etc
    Args:
        ticker (str): ticker symbol

    Returns:
        dict: metadata
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        return {
            'ticker': ticker,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'country': info.get('country', 'N/A')
        }
    except Exception as e:
        logger.error(f'Error while fetching info for ticker {ticker}: {str(e)}')
        return {'ticker': ticker, 'error': str(e)}


help(fetch_ohlcv)
help(fetch_multiple_tickers)


if __name__ == '__main__':
    print("\n=== Test 1: Pojedyncza akcja ===")
    df_aapl = fetch_ohlcv('AAPL', start='2023-01-01')
    print(df_aapl.head())
    print(f"\nShape: {df_aapl.shape}")

    save_to_csv(df_aapl, '../../data/raw/AAPL.csv')

    # Test wielu akcji
    print("\n=== Test 2: Wiele akcji ===")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    df_multiple = fetch_multiple_tickers(tickers, start='2023-01-01')
    print(df_multiple.groupby('ticker').size())

    # Zapis połączonych danych
    save_to_csv(df_multiple, '../../data/raw/multi_stocks.csv')

    # Test metadanych
    print("\n=== Test 3: Metadane ===")
    info = get_ticker_info('AAPL')
    print(info)


