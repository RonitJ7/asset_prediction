import pandas as pd

returns = pd.read_csv('processed_data/daily_returns.csv', index_col=0, parse_dates=True)
future_1d_returns = returns.shift(-1).dropna(how='all')
print(future_1d_returns.head())
future_1d_returns.to_csv('processed_data/future_1day_returns.csv')