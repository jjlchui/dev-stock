import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from stocktrends import Renko
import stocktrends
from stocktrends import indicators

df = pd.read_csv("D:\\Development\\flask dev\\stock\\data\\2022-10-06 NQ=F USTime_out_stock_data.csv")
df.columns=['Datetime','Open','High','Low','Close', 'Volume' ]
len(df)


def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df_r = DF[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    #df_r.reset_index(inplace=True)
    df_r.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df_r)
    df2.brick_size = 20
    renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
    return renko_df 

df_renko = renko_DF(df)       
df_renko_final = df_renko.groupby('date').first()
df_renko_final = df_renko_final.reset_index()
df_renko_final = df_renko_final.rename(columns={'date': 'Datetime'})
df_renko_final['Datetimes1'] = df_renko_final['Datetime'].shift()

df.Datetime = pd.to_datetime(df.Datetime)

idx_max_tmp = df.resample('20min', on='Datetime')["Close"].agg(lambda x: np.nan if x.isna().all() else x.idxmax())

#idx_max = np.where(, idx_max_tmp, np.nan)
idx_max=idx_max_tmp

max_close_idx = np.where(df.index.isin(idx_max), df.Close, np.nan)
df['max_close_idx'] = max_close_idx

val_max = df.resample('20min', on='Datetime')["Close"].agg(lambda x: np.nan if x.isna().all() else x.max())
max_close_val = np.where(df.Close.isin(val_max), df.Close, np.nan)
df['max_close_val'] = max_close_val

#### MIN - Buy ###

idx_min_tmp = df.resample('20min', on='Datetime')["Close"].agg(lambda x: np.nan if x.isna().all() else x.idxmin())

#idx_max = np.where(, idx_max_tmp, np.nan)
idx_min=idx_min_tmp

min_close_idx = np.where(df.index.isin(idx_min), df.Close, np.nan)
df['min_close_idx'] = min_close_idx

val_min = df.resample('20min', on='Datetime')["Close"].agg(lambda x: np.nan if x.isna().all() else x.min())
min_close_val= np.where(df.Close.isin(val_min), df.Close, np.nan)
df['min_close_val'] = min_close_val

##### NEW DF #####

#df_new = pd.concat([df, df_renko_final])
df_new_m = df_renko_final.merge(df,on='Datetime',how='left')

df['failure_x'] = np.where((df['start_time'] <= df['date']) & (df['date'] <=  df['end_time']), df['failure_y'], df['failure_x'])