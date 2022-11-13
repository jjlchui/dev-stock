
import pandas as pd
import numpy as np
import statsmodels.api as sm


def slope(ser, n):
    #"function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n, len(ser)+1):
        y=ser[i-n:i]
        x=np.array(range(n))
        y_scales = (y-y.min())/(y.max()-y.min())
        x_scales = (x-x.min())/(x.max()-x.min())
        x_scales = sm.add_constant(x_scales)
        model = sm.OLS(y_scales, x_scales)
        results = model.fit()
        
        slopes.append(results.params[-1])
        #results.summary()
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def ma_strategy(data):
    df = data.copy()
    Buy=[]
    Sell=[]
    Record=[]
    position=False
        
    df['ma1s1'] = df['MA10'].shift(-1)
    df['ma1s2'] = df['MA10'].shift(-2)
    df['ma2s1'] = df['MA20'].shift(-1)
    df['ma2s2'] = df['MA20'].shift(-2)
    
    df['slope20'] = slope(df.Close, 20)
    df['slope10'] = slope(df.Close, 10)
    
    df['slope20s1p'] = df['slope20'].shift(1)
    df['slope10s1p'] = df['slope10'].shift(1)
    
    df['buy_pt'] = np.where((((df.MA10 <= df.MA20) & (df.ma1s1 > df.ma2s1)) | 
                ((df.MA10 >= df.MA20) & (df.ma1s1 < df.ma2s1))),
                "y", "n")
    df['sell_pt'] =  np.where((((df.MA10 <= df.MA20) & (df.ma1s1 > df.ma2s1)) | 
                ((df.MA10 >= df.MA20) & (df.ma1s1 < df.ma2s1))),       
                "y", "n")
    
    df['buy_pt_s1'] = df['buy_pt'].shift(1)
    df['sell_pt_s1'] = df['sell_pt'].shift(1)
    
    df['slope_chg_dn'] = np.where(((df.slope20 < 0) & (df.slope20s1p > 0)),"y", "n")
    df['slope_chg_up'] = np.where(((df.slope20 > 0) & (df.slope20s1p < 0)),"y", "n")
    
    df['slope_chg_up_s1'] = df['slope_chg_up'].shift(1)
    df['slope_chg_dn_s1'] = df['slope_chg_dn'].shift(1)

    Buy = np.where(
                     (df.buy_pt_s1 == "y") &
                     (df.MA10 >= df.MA20) &
                     ((df['slope_chg_up'] == "y")  | (df['slope_chg_up_s1'] == 'y')),
                      df.Close, "NaN")
    
    
    Sell = np.where(
                (df.sell_pt_s1 == "y") &
                (df.MA10 <= df.MA20) &
                ((df['slope_chg_dn'] == "y") | (df['slope_chg_dn_s1'] == 'y')),
                df.Close, "NaN")

    #df.to_csv("ma.csv")
    return Buy, Sell

def macd_strategy(data):
    
    df = data.copy()
    df['macds1'] = df['MACD_12_26_9'].shift(-1)
    df['macdss1'] = df['MACDs_12_26_9'].shift(-1)
    df['macdss1'] = df['MACDs_12_26_9'].shift(-1)
    
    df['buy_pt'] = np.where((((df.MACD_12_26_9 <= df.MACDs_12_26_9) & (df.macds1 > df.macdss1)) | 
                ((df.MACD_12_26_9 >= df.MACDs_12_26_9) & (df.macds1  < df.macdss1))), "y", "n")
    df['sell_pt'] = np.where((((df.MACD_12_26_9 <= df.MACDs_12_26_9) & (df.macds1 > df.macdss1)) | 
                ((df.MACD_12_26_9  >= df.MACDs_12_26_9) & (df.macds1 < df.macdss1))), "y", "n")
    
    df['buy_pt_s1'] = df['buy_pt'].shift(1)
    df['sell_pt_s1'] = df['sell_pt'].shift(1)
    
    macd_max_20 = df.Close.tail(20).max()
    macd_min_20 = df.Close.tail(20).min()
    macd_max_40 = df.Close.tail(40).max()
    macd_min_40 = df.Close.tail(40).min()
    
    
    df['macdh_s1'] = df['MACDh_12_26_9'].shift(1)
    df['macdh_trend_abv_up'] = np.where(((df['MACDh_12_26_9'] < 0) & (df['macdh_s1'] > 0)), "y", "n")
    df['macdh_trend_abv_dn'] = np.where(((df['MACDh_12_26_9'] > 0) & (df['macdh_s1'] < 0)), "y", "n")
    df['macdh_trend_blw_up'] = np.where(((df['MACDh_12_26_9'] > 0) & (df['macdh_s1'] < 0)), "y", "n")
    df['macdh_trend_blw_dn'] = np.where(((df['MACDh_12_26_9'] < 0) & (df['macdh_s1'] > 0)), "y", "n")
    
    #df['macdh_vol'] = np.where((df['MACD_12_26_9'] >= 2 & df['MACDs_12_26_9'] <= -2), "y", "n")
    
    Buy = np.where((df['buy_pt_s1'] == 'y') & 
                   ((df['macdh_trend_abv_up'] == "y") | (df['macdh_trend_blw_up'] == "y")) &
                   (df['macds1'] >= df['macdss1']),
                    df['Close'], "NaN"),
    
    Sell = np.where((df['sell_pt_s1'] == 'y') & 
                    (df['macdh_trend_abv_up'] == "y") &
                    (df['macds1'] <= df['macdss1']),
                     df['Close'], "NaN"),
      
    
    

    #df.to_csv("macd.csv")
    return Buy, Sell
