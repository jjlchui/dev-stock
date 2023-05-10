from dash import html,dash_table
from dash import dcc
#import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from flask_caching.backends import FileSystemCache
#from dash_extensions.callback import CallbackCache, Trigger
import plotly.graph_objects as go
import dash
import os
from datetime import datetime
import pandas as pd
import pandas_ta as ta
from dash.exceptions import PreventUpdate
import numpy as np
from decimal import Decimal
from scipy.stats import linregress
import pytz
from flask_caching import Cache
import feather
from stocktrends import Renko
from datetime import timedelta
import math
import sqlite3 

############ SQLITE3 new coding ###########
                
#@app.callback(Output('df_value', 'data'), Input('update_value', 'n_intervals'))
#@cache.memoize(timeout=timeout)





def usny_curtime():
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    fmt = '%Y-%m-%d %H:%M:%S'
    time_stamp = nyc_datetime.strftime(fmt)
    return time_stamp

def file_name(filename):
    #cwd = os.getcwd()
    path = "D:\\Development"
    file_path = path + "\\InteractiveBrokers dev\\data\\"
    time_stamp = usny_curtime()
    #time_stamp = time_stamp =  datetime.now().strftime('%Y-%m-%d')
    timefile = os.path.join(file_path + str(time_stamp[0:11]) +"MNQ " + filename)
    return timefile



def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    #DF = DF.set_index("Datetime")
    df_r = DF[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    #df_r.reset_index(inplace=True)
    df_r.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df_r)
    df2.brick_size = 12
    renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
    return renko_df 


    
def bs_count(df,a, as1, idx):
    
    chk_repeat=[0]*(idx)
    for i in range(idx, len(a)):
        if (((a[i] + as1[i-1] > 1) | 
            (a[i] + as1[i-1] < -1))|
            ((a[i] == 0) & (as1[i-1] == 1)) | 
            ((a[i] == 0) & (as1[i-1] == -1)) 
            ):
                df.loc[i, 'as1'] = df.loc[i-1, 'as1']
                chk_repeat.append(5)


        elif (((a[i] == 1) & (as1[i-1] == 0)) | 
             ((a[i] == -1) & (as1[i-1] == 0)) |
              ((a[i] == 0) & (as1[i-1] == 0))
             ):
                df.loc[i, 'as1'] = df.loc(axis=0)[i, 'a']
                chk_repeat.append(df.as1[i-1])

    
        elif (((a[i] == 1) & (as1[i-1] == -1)) | 
             ((a[i] == -1) & (as1[i-1] == 1)) 
             ):
                df.loc[i, 'as1'] = 0
                chk_repeat.append(df.as1[i-1])
        else:
                df.loc[i, 'as1'] = 0
                chk_repeat.append(5)
    
        
    return chk_repeat


def create_dash(flask_app):

        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/Renko_tws_reconcile/"), prevent_initial_callbacks=True)


        ### Preformance Turning
        cache = Cache(app.server, config={
            "CACHE_TYPE": "SimpleCache",
        })

        app.config.suppress_callback_exceptions = True
        timeout = 20
        #cc = CallbackCache(cache=FileSystemCache(cache_dir="cache"))

        #app.css.append_css({"external_url": "/static/css/style.css"})
        app.css.config.serve_locally = True

        GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)


        app.layout = html.Div([

            html.Link(rel='stylesheet', href='/static/css/style.css'),
            
            ##### Store data
                dcc.Store(id='df_value'),
                dcc.Store(id='df_sl'),
                dcc.Store(id='buy'),
                dcc.Store(id='sell'),
                dcc.Store(id='buy_nplist'),
                dcc.Store(id='sell_nplist'),
                dcc.Store(id='buy_tcount'),
                dcc.Store(id='sell_tcount'),
                dcc.Store(id='tot_tbuy'),
                dcc.Store(id='tot_tsell'),


              #### Header
              
              html.Div([
                 
                 html.Div([
                     html.Img(src = app.get_asset_url('logo.jpeg'),
                              style = {'height': '30px'},
                              className = 'title_image'),
         
                 ], className = 'logo_title'),

                  html.Div([
                  html.Div([
                      html.Div([
                          html.P('Tot_Buy/no.(ave)',
                                  style = {'color': 'white', 'font-size':'8px'},
                                  className = 'stock_label'),
                          html.Div(id = 'tot_buy',
                                  style = {'color': 'white', },
                                  className = 'stock_score'),
                      ],className = 'stock_score_label'),
                      html.P(['  /  '], className = 'stock_score'),
                      html.Div(id = 'buy_count',
                              style = {'color': 'white', },
                              className = 'stock_score'),
                      html.P([' ('], className = 'stock_score'),
                      html.Div(id = 'ave_b',
                              style = {'color': 'white', },
                              className = 'stock_ave'),
                      html.P([')'], className = 'stock_score'),


                  ], className = 'buy_sell_b'),

                  html.Div([
                  html.Div([
                      html.P('Tot_Sell/no.(ave):',
                              style = {'color': 'white', 'font-size':'8px'},
                              className = 'stock_label'),
                      html.Div(id = 'tot_sell',
                              style = {'color': 'white'},
                              className = 'stock_score'),
                      ],className = 'stock_score_label'),
                      html.P(['  /  '], className = 'stock_score'),
                      html.Div(id = 'sell_count',
                              style = {'color': 'white', },
                              className = 'stock_score'),
                      html.P([' ('], className = 'stock_score'),
                      html.Div(id = 'ave_s',
                              style = {'color': 'white', },
                              className = 'stock_ave'),
                      html.P([')'], className = 'stock_score'),

                  ], className = 'buy_sell_s'),

                 html.Div([
                            html.P('Buy/Short pft.:',
                                    style = {'color': '#bebfd6', 'font-size':'8px'},
                                    className = 'stock_label'),
                            html.P(['('], style = {'color': '#bebfd6', 'font-size':'12px'},
                                   className = 'profit_label'),
                             html.Div(id = 'tot_profit',
                                     className = 'stock_label'),
                             html.P(['+'], style = {'color': '#bebfd6', 'font-size':'12px'},
                                    className = 'profit_label'),
                             html.Div(id = 'tot_long',
                                     className = 'stock_label'),
                             html.P([')'], style = {'color': '#bebfd6', 'font-size':'12px'},
                                    className = 'profit_label'),
                             html.Div(id = 'profit_t',
                                     className = 'profit'),
                             html.Div(id = 'Img_t',
                                     className = 'profit_img'),
                             html.Div(id = 'profit_sl_tot',
                                     style =  {'color': '#f20540', 'fontSize' : 17, 'margin-top': '11px'}, 
                                     className = 'profit'),


                     ],className = 'buy_sell_p'),
               ], className = 'stock_score_container'),


                html.H6(id = 'get_date_time',
                        style = {'color': 'white'},
                        className = 'adjust_date_time'),
             ], className = 'title_date_time_container'),


             html.Div([
                 dcc.Interval(id = 'update_date_time',
                        interval = 1000,
                        n_intervals = 0)
             ]),
           ##### button
                   




            ##### Graph Display
                html.Div([

                        html.Div([
                            html.Title('profitchart',
                            style = {'color': 'black'},
                            ),
                            html.Div(id='table-container', className ='table_style',

                                    ),
                            ]),


                        dcc.Interval(id = 'update_value',
                                         interval = int(GRAPH_INTERVAL) ,
                                         n_intervals = 0)
                    ]),



        ])










        @app.callback(Output('df_value', 'data'),
                      Output('df_sl', 'data'),
                      Input('update_value', 'n_intervals'))
        @cache.memoize(timeout=timeout)
        
        def update_df(n_intervals):
                if n_intervals == 0:
                    raise PreventUpdate
                else:

                    ### tws data
                    file = file_name("df_sql.csv")
                    df = pd.read_csv(file, usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,19])
                    #df.drop(columns=df.columns[0], axis=1, inplace=True)
                    df.columns =['Datetime','bs_test', 'bss1', 'bss2','Open','High','Low','Close', 'Volume', 'uptrend','bar_no',
                    #                            'macd_tws','macd_tws_sig','macd_tws_slope','macd_tws_sig_slope','final_signal'
                                                'buysell', 'ret', 'count' ]
                    
                    print("df ok")
                    ### stop loss csv
                    file_sl = file_name("df_sl.csv")
                    check_file = os.path.exists(file_sl)
                    if (check_file): 
                        file = open(file_sl)
                        numline = len(file.readlines())
                        if (numline > 1):
                            df_sl = pd.read_csv(file_sl, usecols=[0,2,3,4,5])
                            df_sl.columns =['Datetime','status','exeprice', 'price', 'idx']
                            print("df_sl updated", len(df_sl))
                        else:
                            df_sl = pd.read_csv(file_sl)
                    else:
                        print("no df_sl.csv")

                    

                return df.to_dict('records'), df_sl.to_dict('records')





        @app.callback(Output('get_date_time', 'children'),
                      [Input('update_date_time', 'n_intervals')])
        def live_date_time(n_intervals):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                now = datetime.now()
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

            return [
                html.Div(dt_string)
            ]

        @app.callback([
                Output('tot_tbuy', 'data'),
                Output('tot_tsell', 'data'),
                Output('buy_tcount', 'data'),
                Output('sell_tcount', 'data'),
                Output('buy', 'children'),
                Output('sell', 'children')
                ],
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        @cache.memoize(timeout=timeout)
        def renkotws_strategy(n_intervals, data):
                if n_intervals == 0:
                    raise PreventUpdate
                else:
            
                    df = pd.DataFrame(data)
                    
                    ############### NEW CHANGE of MIN/MAX start
                    
                    df.Datetime = pd.to_datetime(df.Datetime)
                    

                
                    ############### NEW CHANGE of MIN/MAX end
                
                    ####### BUY #######
                    #T_Buy = np.where((df['Close'] == df['min_Close'] ) 
                    T_Buy = np.where((df['buysell'] == "Buy" ) 
                                    #(df['macd_vol_gt1'] == 'y') &
                                    #((df['slopes10'] < 10) ) &
                                    #((df['macd_below_dn'] == 'y') | (df['macd_above_dn'] == 'y'))
                                    ,df['Close'], np.nan)
                    df['T_Buy'] = T_Buy
                    Buylist = np.where((T_Buy==df['Close']), 1, 0)
                    
                    ####### SELL #######
                
                    #T_Sell = np.where((df['Close'] == df['max_Close']) 
                    T_Sell = np.where((df['buysell'] == "Sell") 
                                     #(df['macd_vol_gt1'] == 'y') &
                                     #((df['slopes10'] > -10) ) &
                                     #((df['macd_above_up'] == 'y') | (df['macd_below_up'] == 'y'))
                                     ,df['Close'], np.nan)
                    
                    df['T_Sell'] = T_Sell               
                    Selllist = np.where((T_Sell==df['Close']), -1, 0)
                    
                    
                    ### add Buy Sell list ###
                    
                    
                    ### call count ###
                    df['BS_list'] = Buylist + Selllist
                    df['a']= df['BS_list']
                    df['as1'] = df['a'] .fillna(0)
                    df['as1'] = df['as1'].astype(int)
                    
                    if (df['a'].any() != 0):
                        df['chk_repeat'] = bs_count(df,df['a'], df['as1'], 1)
                    else:
                        df['chk_repeat'] = 0
                         
                    ### Final Buy & Sell  
                    Buy = np.where((T_Buy == df['Close']) & 
                                    (df['chk_repeat'] != 5) &                 
                                    #(~np.isnan(df.chk_repeat)) &
                                    ((df['as1'] == 0) | (df['as1'] == 1)), 
                                     df.Close, "NaN")
                
                    Sell = np.where((T_Sell == df['Close']) & 
                                    (df['chk_repeat'] != 5) &  
                                    #(~np.isnan(df.chk_repeat)) &
                                    ((df['as1'] == 0) | (df['as1'] == -1)), 
                                    df.Close, "NaN")
                    

                    #### add Buy Sell to df
                    Buy_list = list(Buy)
                    df_Buy = pd.DataFrame(Buy_list, columns=['Buy'])
                    #df_Buy = df_Buy.T
                    df = pd.concat([df, df_Buy], axis=1)
                    
                    Sell_list = list(Sell)
                    df_Sell = pd.DataFrame(Sell_list, columns=['Sell'])
                    #df_Sell = df_Sell.T
                    df = pd.concat([df, df_Sell], axis=1)
                    
                    

                    
                    
                    # total Buy/Buy count

                    df['Buy_zero'] = np.where(df.Buy == "NaN", 0, df.Buy)
                    tot_buy = np.add.accumulate(df['Buy_zero'].astype(float))
                
                    df['Buy_count_tmp'] = np.where(df['Buy'] == 'NaN', 0, 1)
                    df['Buy_count'] = np.add.accumulate(df['Buy_count_tmp'])

                    # total SELL/count
                

                    df['Sell_zero'] = np.where(df.Sell == "NaN", 0, df.Sell)
                    tot_sell = np.add.accumulate(df['Sell_zero'].astype(float))
                
                
                    df['Sell_count_tmp'] = np.where(df['Sell'] == 'NaN', 0, 1)
                    df['Sell_count'] = np.add.accumulate(df['Sell_count_tmp'])
                    
                    #Sell_count = abs(df.Sell_count)
                    #Buy_count = abs(df.Buy_count)
                
                    Buy=Buy.tolist()
                    Sell=Sell.tolist()
                    
                    tot_buy = tot_buy.iloc[-1]
                    tot_sell =  tot_sell.iloc[-1]
                    Buy_count =  df.Buy_count.iloc[-1]
                    Sell_count =  df.Sell_count.iloc[-1]
                    
                    df.to_csv("renko_tws.csv")

                    return tot_buy, tot_sell, Buy_count, Sell_count, Buy, Sell, 


        

        @app.callback(
                    [Output('tot_buy', 'children'),
                    Output('tot_sell', 'children'),
                    Output('buy_count', 'children'),
                    Output('sell_count', 'children'),
                    Output('ave_b', 'children'),
                    Output('ave_s', 'children'),
                    ],
                    [Input('update_value', 'n_intervals'),
                     Input('tot_tbuy', 'data'),
                     Input('tot_tsell', 'data'),
                     Input('buy_tcount', 'data'),
                     Input('sell_tcount', 'data'),])
        @cache.memoize(timeout=timeout)
        def cal_count(n_intervals, tot_tbuy, tot_tsell, buy_tcount, sell_tcount):
                if n_intervals == 0:
                    raise PreventUpdate
                else:
                    if (buy_tcount is None):
                        buy_count = 0
                        tot_buy = 0
                        ave_b=0
                    else:
                        if (buy_tcount > 0):
                            ave_b = tot_tbuy / buy_tcount
                            ave_b = np.around(ave_b, 2)
                            buy_count = buy_tcount
                            tot_buy = tot_tbuy
                        else:
                             buy_count = 0
                             tot_buy = 0
                             ave_b=0

                    if (sell_tcount is None):
                        sell_count = 0
                        tot_sell = 0
                        ave_s=0
                    else:
                        if (sell_tcount > 0):
                            ave_s = tot_tsell / sell_tcount
                            ave_s = np.around(ave_s, 2)
                            sell_count = sell_tcount
                            tot_sell = tot_tsell
                        else:
                            sell_count = 0
                            tot_sell = 0
                            ave_s=0


                    print("sell_count & buy_count", buy_count, sell_count)
                    return [tot_buy, tot_sell, buy_count, sell_count, ave_b, ave_s]





        @app.callback(
                [
                 Output('tot_profit', 'children'),
                 Output('tot_long', 'children'),
                 Output('profit_t', 'children'),
                 Output('Img_t', 'children'),
                 Output('buy_nplist', 'children'),
                 Output('sell_nplist', 'children'),
                ],
                [Input('update_value', 'n_intervals'),
                State('buy', 'children'),
                State('sell', 'children'),
                Input('df_value', 'data')
                 ])

        def update_profitchart(n_intervals, buy, sell, data):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                df = pd.DataFrame(data)
                buyitem = np.array(buy).tolist()
                sellitem = np.array(sell).tolist()
                

                ###### Calculate profit START
                
                if buyitem is None:
                        buyitem = []
                if sellitem is None:
                        sellitem = []

                buy_list = []
                buy_idx = []
                buy_count = []
                
                for idx, xs in enumerate(buyitem):
                    if xs != 'NaN':
                        buy_list.append(xs)
                        buy_idx.append(idx)
                        buy_count.append(df['count'][idx])
                        

                sell_list = []
                sell_idx = []
                sell_count = []
                
                for idx, xs in enumerate(sellitem):
                    if xs != 'NaN':
                        sell_list.append(xs)
                        sell_idx.append(idx)
                        sell_count.append(df['count'][idx])

                buy_nplist = np.column_stack((buy_list, buy_idx, buy_count))
                sell_nplist =  np.column_stack((sell_list, sell_idx, sell_count))

                min_len = np.minimum(len(buy_nplist), len(sell_nplist))


                tot_profit = 0
                buy_profit = 0
                tot_long = 0
                long_profit = 0


                for i in range (min_len):
                    #buy-> sell
                    if(np.minimum(int(buy_nplist[i,1]), int(sell_nplist[i,1])) == int(buy_nplist[i,1])):
                        buy_profit = float(sell_nplist[i,0]) - float(buy_nplist[i,0])
                        tot_profit += buy_profit
                        #print("buy_profit_idx",i," ",(buy_nplist[i,1]),"<", sell_nplist[i,1],buy_profit)
                        #print("buy_profit",i," ",(buy_nplist[i,0]),"<", sell_nplist[i,0],buy_profit, tot_profit)
                    else:
                        long_profit = float(sell_nplist[i,0]) - float(buy_nplist[i,0])
                        tot_long +=  long_profit
                        #print("long_profit_idx",i," ",buy_nplist[i,1],">", (sell_nplist[i,1]), long_profit)
                        #print("long_profit",i," ",buy_nplist[i,0],">", (sell_nplist[i,0]), long_profit, tot_long)

                    #sell -> buy

                    print("total", tot_profit, tot_long)
                    profit_t = float(tot_long + tot_profit)

                ###### Calculate profit FINISH

                return [html.H6('{0:,.2f}'.format(tot_profit),style = {'color': '#bebfd6', 'font-size':'12px', 'margin-top': '8px'},),
                        html.H6('{0:,.2f}'.format(tot_long),style = {'color': '#bebfd6', 'font-size':'12px', 'margin-top': '8px'},),
                        html.H6('{0:,.2f}'.format(profit_t),style =  {'color': '#f20540', 'fontSize' : 17, 'margin-top': '11px'},),
                        html.Img(id = "Img_t",src = app.get_asset_url('money-bag.png'), style = {'height': '30px'},className = 'coin'), 
                        buy_nplist, sell_nplist]
 


        @app.callback(
                Output('table-container', 'children'),
                Output('profit_sl_tot', 'children'),
                [Input('update_value', 'n_intervals'),
                 Input('buy_nplist', 'children'),
                 Input('sell_nplist', 'children'),
                 Input('df_value', 'data'),
                 Input('df_sl', "data")])
        
        @cache.memoize(timeout=timeout)
        
        def update_table(n_intervals, buy_nplist, sell_nplist, data, sl_data):

            if n_intervals == 0:
                raise PreventUpdate
            else:
                df = pd.DataFrame(data)
                df_sl = pd.DataFrame(sl_data)
                
                df_b = pd.DataFrame(buy_nplist, columns=['buy', 'b_idx', 'b_count'])
                df_s = pd.DataFrame(sell_nplist, columns=['sell', 's_idx', 's_count'])

                df_c = pd.concat([df_b, df_s], axis=1)
                df_c = df_c.fillna(0)
                df_c.b_idx = df_c.b_idx.astype(int)
                df_c.s_idx = df_c.s_idx.astype(int)
                df_c.sell = df_c.sell.astype(float)
                df_c.buy = df_c.buy.astype(float)
                
                df_c['profit'] = np.where(df_c.b_idx > df_c.s_idx,
                                      df_c.sell - df_c.buy,
                                      df_c.sell - df_c.buy,
                                       )

                df_c.profit = np.where(
                                  (df_c.buy.astype(int) != 0) &
                                  (df_c.sell.astype(int) != 0),
                                  df_c.profit,
                                  np.nan)
                
                
                ################ start new source : STOPLOSS"
                if (len(df_sl) == 0):
                    profit_sl_tot = 0
                    print("no sl data")
                else:
                    df_c.insert(5, 'sl_idx', 0 )
                    df_c.insert(6, 'price_sl', 0.0 )
                    df_c.insert(7, 'profit_sl', 0.0)
                    df_c.insert(8, 'sl_status', np.nan)
    
                    """
                    for i in range(df_sl.shape[0]):
                        for j in range(df_c.shape[0]):
                            
                            if (((df_sl.idx[i].astype('int')>= df_c.s_idx[j].astype('int32')) and (df_sl.idx[i].astype('int32') <= df_c.b_idx[j].astype('int32'))) or 
                                ((df_sl.idx[i].astype('int32') <= df_c.s_idx[j].astype('int32')) and (df_sl.idx[i].astype('int32')>= df_c.b_idx[j].astype('int32')))):
                                print(i,j)
                                
                                if ((df_c.buy[j].astype('int32') != 0) and (df_c.sell[j].astype('int32') != 0)):
                                    df_c.price_sl[j]=df_sl.price[i]
                                    df_c.sl_idx[j]=df_sl.idx[i]
                                    df_c.sl_status[j] = df_sl.status[i]
                                    #print("i", i, df_c.price_sl[j], df_c.sl_idx, df_sl.price[i])  
                            
                            elif((df_c.s_idx[j].astype('int32')  == 0 and df_c.sell[j].astype('int32') == 0) or (df_c.b_idx[j].astype('int32') == 0 and df_c.buy[j].astype('int32')  == 0)): 
                                    df_c.price_sl.iloc[j]=df_sl.price.iloc[i]
                                    df_c.sl_idx.iloc[j]=df_sl.idx.iloc[i]
                                    df_c.sl_status[j] = df_sl.status[i]
                                    #print("0", i, df_c.price_sl.iloc[j], df_c.sl_idx.iloc[j], df_sl.price.iloc[i]) 
                     """        

    
                   
                    for i in range(df_sl.shape[0]):
                        print("inside i")
                        for j in range(df_c.shape[0]):
                            print("inside j")
                            if (str(df_sl['status'].iloc[i]) == "BUY"):
                                if (df_sl.idx[i].astype('int')>= int(df_c.s_count[j]) and
                                    (df_sl.idx[i].astype('int') <= int(df_c.b_count[j]))):
                                    df_c.price_sl[j]=df_sl.price[i]
                                    df_c.sl_idx[j]=df_sl.idx[i]
                                    df_c.sl_status[j] = df_sl.status[i]
                                    print("insert Buy", i, j)

                            elif (str(df_sl.status.iloc[i]) == "SELL"):
                                if (df_sl.idx[i].astype('int')>= int(df_c.b_count[j]) and
                                    (df_sl.idx[i].astype('int') <= int(df_c.s_count[j]))):
                                    df_c.price_sl[j]=df_sl.price[i]
                                    df_c.sl_idx[j]=df_sl.idx[i]
                                    df_c.sl_status[j] = df_sl.status[i]
                                    print("insert Sell", i, j)

                            else:
                                    df_c.price_sl[j]=" "
                                    df_c.sl_idx[j]= " "
                                    df_c.sl_status[j] = " "
                                    print("empty", i, j)
                                    
                                    
                            """   
                            if (((df_sl.idx[i].astype('int')>= int(df_c.s_count[j])) and (df_sl.idx[i].astype('int') <= int(df_c.b_count[j]))) or 
                                ((df_sl.idx[i].astype('int') <= int(df_c.s_count[j])) and (df_sl.idx[i].astype('int')>= int(df_c.b_count[j])))):
                                print(i,j)
                                
                                if ((df_c.buy[j].astype('int32') != 0) and (df_c.sell[j].astype('int32') != 0)):
                                    df_c.price_sl[j]=df_sl.price[i]
                                    df_c.sl_idx[j]=df_sl.idx[i]
                                    df_c.sl_status[j] = df_sl.status[i]
                                    #print("i", i, df_c.price_sl[j], df_c.sl_idx, df_sl.price[i])  
                            
                            elif((int(df_c.s_count[j])  == 0 and df_c.sell[j].astype('int') == 0) or (int(df_c.b_count[j]) == 0 and df_c.buy[j].astype('int')  == 0)): 
                                    df_c.price_sl.iloc[j]=df_sl.price.iloc[i]
                                    df_c.sl_idx.iloc[j]=df_sl.idx.iloc[i]
                                    df_c.sl_status[j] = df_sl.status[i]
                                    #print("0", i, df_c.price_sl.iloc[j], df_c.sl_idx.iloc[j], df_sl.price.iloc[i]) 
                            """  
                                        
                                    
                                    
                    profit_sl = np.where(df_c['sl_status'] == "BUY",
                                           df_c['sell'].astype(float) - df_c['price_sl'].astype(float),
                                           df_c['price_sl'].astype(float) - df_c['buy'].astype(float),
                                           )
                    
                    
                    profit_sl = np.where(df_c['price_sl'] == 0.0 , df_c['profit'], profit_sl)
                    
                    df_c['profit_sl'] = profit_sl
                    
                    
                    profit_sl_tot = df_c['profit_sl'].sum()
                    
                    
                    

                
                
                ################ end new source
                
                if len(df_s) > len(df_b):
                    styles = [
                              {'if': {'column_id': 'profit', 'filter_query': '{buy} != 0 && {b_idx} > {s_idx}'}, 'color': 'tomato','fontWeight':'normal'},
                              {'if': {'column_id': 'profit', 'filter_query': '{buy} != 0 && {b_idx} < {s_idx}'}, 'color': '#39CCCC','fontWeight':'normal'},
                              {"if": {'column_id': 'buy', "filter_query": "{buy} != 0 && {b_idx} > {s_idx}" }, "backgroundColor": "yellow", 'color': 'black',},
                              {"if": {'column_id': 'sell', "filter_query": "{buy} != 0 && {b_idx} < {s_idx}" }, "backgroundColor": "B10DC9",'color': 'black',},
                              ]
                else:
                    styles = [
                              {'if': {'column_id': 'profit', 'filter_query': '{sell} != 0 && {b_idx} > {s_idx}'}, 'color': 'tomato','fontWeight':'normal'},
                              {'if': {'column_id': 'profit', 'filter_query': '{sell} != 0 && {b_idx} < {s_idx}'}, 'color': '#39CCCC','fontWeight':'normal'},
                              {"if": {'column_id': 'buy', "filter_query": "{sell} != 0 && {b_idx} > {s_idx}" }, "backgroundColor": "yellow", 'color': 'black',},
                              {"if": {'column_id': 'sell', "filter_query": "{sell} != 0 && {b_idx} < {s_idx}" }, "backgroundColor": "B10DC9",'color': 'black',},
                              ]



                layout = html.Div([dash_table.DataTable(
                        columns=[{'name': i, 'id': i} for i in df_c.columns],
                        data=df_c.to_dict('records'),
                        style_cell={'font-family': 'sans-serif'},

                        style_header={
                            'backgroundColor': 'rgb(10, 10, 10)',
                            'border': '1px solid black',
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_data={
                            'backgroundColor': 'rgb(60, 60, 60)',
                            'border': '1px solid grey',
                            'color': 'rgb(255,255,255, 0.8)',
                            'fontWeight': 'normal'
                        },
                        style_data_conditional=styles,
                        )])

                return layout, profit_sl_tot
                

        #cc.register(app)
        return(app)

"""
if __name__ == '__main__':
    app.run_server()

"""




