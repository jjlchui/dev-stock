from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import requests

key_path = "D:\\Development\\Alpha_Vantage.txt"

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NFX&interval=1min&apikey=key_path'
r = requests.get(url)
data = r.json()
print(data)


ts = TimeSeries(key=open(key_path,'r').read(), output_format='pandas')
data = ts.get_daily(symbol='NFX', outputsize='full')[0]
data.columns = ["open","high","low","close","volume"]
data = data.iloc[::-1]


from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
ts = TimeSeries(key=key_path, output_format='pandas')
data, meta_data = ts.get_intraday(symbol='AAPL',interval='1min', outputsize='full')
pprint(data.head(2))




all_tickers = ["NFX"]
close_prices = pd.DataFrame()
api_call_count = 1

start_time = time.time()
for ticker in all_tickers:
    data = ts.get_intraday(symbol=ticker,interval='1min', outputsize='compact')[0]
    api_call_count+=1
    data.columns = ["open","high","low","close","volume"]
    data = data.iloc[::-1]
    close_prices[ticker] = data["close"]
    if api_call_count==5:
        api_call_count = 1
        time.sleep(60 - ((time.time() - start_time) % 60.0))
