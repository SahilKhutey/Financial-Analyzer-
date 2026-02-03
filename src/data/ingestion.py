"""
Data Ingestion Layer.
Handles fetching market data from APIs (YFinance) or local files.
"""
import pandas as pd
import yfinance as yf
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataFactory:
    """
    Unified interface for retrieving market data.
    Modes:
    - 'api': Fetch from Yahoo Finance
    - 'csv': Load from local CSV
    """
    
    @staticmethod
    def get_data(
        ticker: str, 
        source: str = 'api', 
        period: str = '2y', 
        interval: str = '1d',
        csv_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.
        
        Args:
            ticker: Symbol (e.g. 'BTC-USD', 'AAPL')
            source: 'api' or 'csv'
            period: Lookback period (api only)
            interval: Timeframe (api only)
            csv_path: Path to CSV file (csv only)
            
        Returns:
            pd.DataFrame: OHLCV data with DatetimeIndex
        """
        if source == 'api':
            return DataFactory._fetch_yfinance(ticker, period, interval)
        elif source == 'csv':
            return DataFactory._load_csv(csv_path)
        else:
            raise ValueError(f"Unknown source: {source}")

    @staticmethod
    def _fetch_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
        logger.info(f"Fetching {ticker} from Yahoo Finance...")
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Normalize Headers (yfinance can return MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure standard columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required]
            return df
            
        except Exception as e:
            logger.error(f"API Fetch Error: {e}")
            raise

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        if not path or not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
            
        logger.info(f"Loading data from {path}...")
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        return df
