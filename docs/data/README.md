# Data Module

The Data module is responsible for providing market data to the system and maintaining clean separation between training and testing datasets, a critical requirement for preventing look-ahead bias in optimization.

## Data Handler

```
DataHandler (Abstract)
  └── initialize(context)
  └── update_bars()
  └── get_latest_bar(symbol)
  └── get_latest_bars(symbol, N=1)
  └── reset()
  └── set_active_split(split_name)
```

The DataHandler is responsible for loading and providing market data

## Data Models

The Bar class represents OHLCV (Open, High, Low, Close, Volume) market data with standardized fields and conversion methods.

## Historical Data Handler

The HistoricalDataHandler implements the DataHandler interface for backtesting

## Time Series Splitter

The TimeSeriesSplitter handles the creation of training and testing datasets