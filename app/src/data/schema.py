import pandas as pd
from typing import List, Dict, Any, Optional


REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close']
OPTIONAL_COLUMNS = ['Adj_Close', 'Volume', 'Symbol']


COLUMN_SYNONYMS = {
    'Date': ['date', 'timestamp', 'time', 'datetime'],
    'Open': ['open', 'opening', 'open_price'],
    'High': ['high', 'highest', 'high_price'],
    'Low': ['low', 'lowest', 'low_price'],
    'Close': ['close', 'closing', 'close_price'],
    'Adj_Close': ['adj_close', 'adjclose', 'adjusted_close', 'adj close', 'adjusted close'],
    'Volume': ['volume', 'vol', 'trade_volume'],
    'Symbol': ['symbol', 'ticker', 'stock', 'asset']
}

def validate_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Validate a DataFrame against the expected schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        warnings.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    
    if 'Date' in df.columns:
        try:
            pd.to_datetime(df['Date'])
        except Exception:
            warnings.append("Date column could not be parsed as datetime")
    
    
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"{col} column is not numeric")
    
    
    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isna().any():
            warnings.append(f"{col} column contains missing values")
    
    return warnings

def infer_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer column mapping from DataFrame columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping canonical columns to DataFrame columns
    """
    
    column_mapping = {}
    
    
    df_columns_lower = {col.lower(): col for col in df.columns}
    
    
    for canonical_col, synonyms in COLUMN_SYNONYMS.items():
        
        if canonical_col.lower() in df_columns_lower:
            column_mapping[canonical_col] = df_columns_lower[canonical_col.lower()]
            continue
        
        
        for synonym in synonyms:
            if synonym in df_columns_lower:
                column_mapping[canonical_col] = df_columns_lower[synonym]
                break
    
    return column_mapping

def apply_column_mapping(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Apply column mapping to DataFrame.
    
    Args:
        df: Input DataFrame
        column_mapping: Dictionary mapping canonical columns to DataFrame columns
        
    Returns:
        DataFrame with standardized columns
    """
    
    df_mapped = pd.DataFrame()
    
    
    for canonical_col, df_col in column_mapping.items():
        if df_col in df.columns:
            df_mapped[canonical_col] = df[df_col]
    
    
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df_mapped.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    
    if 'Adj_Close' not in df_mapped.columns and 'Close' in df_mapped.columns:
        df_mapped['Adj_Close'] = df_mapped['Close']
    
    if 'Volume' not in df_mapped.columns:
        df_mapped['Volume'] = float('nan')
    
    if 'Symbol' not in df_mapped.columns:
        df_mapped['Symbol'] = 'UNKNOWN'
    
    
    df_mapped['Date'] = pd.to_datetime(df_mapped['Date'])
    
    
    df_mapped.sort_values('Date', inplace=True)
    
    return df_mapped