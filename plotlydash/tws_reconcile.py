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
from stocktrends import Renko
from datetime import timedelta
import math
import sqlite3 

############ SQLITE3 new coding ###########
 

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

        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/tws_reconcile/"), prevent_initial_callbacks=True)


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
                dcc.Store(id='BS_tmp_list'),
                dcc.Store(id='df_sl'),
                dcc.Store(id='df_order'),
                


              #### Header
              
              html.Div([
                 
                 html.Div([
                     html.Img(src = app.get_asset_url('logo.jpeg'),
                              style = {'height': '30px'},
                              className = 'title_image'),
         
                 ], className = 'logo_title'),




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










        @app.callback([Output('df_value', 'data'),
                      Output('df_sl', 'data'),
                      Output('df_order', 'data')],
                      Input('update_value', 'n_intervals'))
        @cache.memoize(timeout=timeout)
        
        def update_df(n_intervals):
                if n_intervals == 0:
                    raise PreventUpdate
                else:

                    ### tws data
                    #file = file_name("df_sql.csv")
                    file = "D:\\Development\\InteractiveBrokers dev\\data\\2023-04-02 MNQ df_sql.csv"                    
                    df = pd.read_csv(file, usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
                    #df.drop(columns=df.columns[0], axis=1, inplace=True)
                    df.columns =['Datetime','bs_test', 'bss1', 'bss2','Open','High','Low','Close', 'Volume', 'uptrend','bar_no',
                    #                            'macd_tws','macd_tws_sig','macd_tws_slope','macd_tws_sig_slope','final_signal'
                                                'buysell', 'ret', 'count_no' ]
                    
                    print("df ok")
                    ### stop loss csv
                    #file_sl = file_name("df_sl.csv")
                    file_sl = "D:\\Development\\InteractiveBrokers dev\\data\\2023-04-02 MNQ df_sl.csv"
                    check_file = os.path.exists(file_sl)
                    if (check_file): 
                        file = open(file_sl)
                        numline = len(file.readlines())
                        if (numline > 1):
                            df_sl = pd.read_csv(file_sl, usecols=[0,2,3,4,5,6])
                            df_sl.columns =['Datetime','status','exeprice', 'price', 'idx', 'count_no']
                            print("df_sl updated", len(df_sl))
                        else:
                            df_sl = pd.read_csv(file_sl)
                    else:
                        print("no df_sl.csv")
                        
                    #### Order ####
                    #file_order = file_name("df_order.csv")
                    file_order = "D:\\Development\\InteractiveBrokers dev\\data\\2023-04-02 MNQ df_order.csv"
                    check_file = os.path.exists(file_order)
                    if (check_file): 
                        file = open(file_order)
                        numline = len(file.readlines())
                        if (numline > 1):
                            df_order = pd.read_csv(file_order, usecols=[0,1,3,4,5,6])
                            df_order.columns =['order_idx','Datetime','Desc', 'status', 'lmt_price', 'count_no']
                            print("df_order updated", len(df_order))
                        else:
                            df_order = pd.read_csv(file_order)
                    else:
                        print("no df_order.csv")

                    

                return df.to_dict('records'), df_sl.to_dict('records'), df_order.to_dict('records')





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
        
        

        @app.callback(Output('BS_tmp_list', 'children'),
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
                                     df.Close, np.nan)
                
                    Sell = np.where((T_Sell == df['Close']) & 
                                    (df['chk_repeat'] != 5) &  
                                    #(~np.isnan(df.chk_repeat)) &
                                    ((df['as1'] == 0) | (df['as1'] == -1)), 
                                    df.Close, np.nan)
                    

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
                
                    
                    Buy_tmp = df[df['Buy'].notna()] 
                    Buy_tmp = Buy_tmp.reset_index()
                    Buy1 = Buy_tmp[['Datetime', 'Buy', 'index']]

                    
                    Sell_tmp = df[df['Sell'].notna()] 
                    Sell_tmp = Sell_tmp.reset_index()
                    Sell1 = Sell_tmp[['Datetime', 'Sell', 'index']]
                    #df_tx_s = pd.DataFrame(Sell1, columns=['Datetime', 'Buy', 'index1'])
                    
                    tot_buy = tot_buy.iloc[-1]
                    tot_sell =  tot_sell.iloc[-1]
                    Buy_count =  df.Buy_count.iloc[-1]
                    Sell_count =  df.Sell_count.iloc[-1]
                    
                    BS_tmp = pd.concat([df.Buy, df.Sell], axis=1)
                    BS_tmp = pd.concat([df.Datetime, BS_tmp], axis=1)
                    BS_tmp = BS_tmp.fillna(0)
                    
                    BS_tmp['BS'] = (BS_tmp.Buy + BS_tmp.Sell).astype(float)
                    
                    BS_tmp['status']=np.where(BS_tmp['Buy'] != 0, "Buy", 0)
                    BS_tmp['status']=np.where(BS_tmp['Sell'] != 0, "Sell", BS_tmp['status'])    

                    BS_tmp=BS_tmp[BS_tmp['BS'] != 0]
                    BS_tmp.reset_index(inplace=True)
                    BS_tmp = BS_tmp.rename({'index':'BS_idx'}, axis=1)
                    BS_tmp.drop(['Buy','Sell'], axis=1, inplace=True)
                    BS_tmp_list = BS_tmp.values.tolist()
                   
                    return BS_tmp_list



        @app.callback(
                Output('table-container', 'children'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data'),
                 Input('df_sl', "data"),
                 Input('df_order', "data"),
                 Input('BS_tmp_list', "children")])
        
        @cache.memoize(timeout=timeout)
        
        def update_table(n_intervals,  data, sl_data, order_data, BS_tmp_list):

            if n_intervals == 0:
                raise PreventUpdate
            else:
                df = pd.DataFrame(data)
                df_sl = pd.DataFrame(sl_data)
                df_order = pd.DataFrame(order_data)

                print("....test_BS_list.....", BS_tmp_list)
                df_BS=pd.DataFrame(BS_tmp_list, columns=['BS_idx','Datetime', 'BS', 'status'])
                
                print("....test_BS_df.....", df_BS.iloc)
                
                """
                df_BS['status']=np.where(df_BS['Buy'] != 0, "Buy", 0)
                df_BS['status']=np.where(df_BS['Sell'] != 0, "Sell", df_BS['status'])    
                
                df_BS=df_BS[df_BS['BS'] != 0]
                df_BS.reset_index(inplace=True)
                df_BS = df_BS.rename({'index':'BS_idx'}, axis=1)
                df_BS.drop(['Buy','Sell'], axis=1, inplace=True)
                """
                
                df_BS['Datetime'] = pd.to_datetime(df_BS['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                df_order['Datetime'] = pd.to_datetime(df_order['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                df_sl['Datetime'] = pd.to_datetime(df_sl['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                df_order = df_order.merge(df_sl[["Datetime", "count_no", "price", "exeprice"]],left_on="count_no", right_on="count_no", how="outer")
                df_BS = df_BS.merge(df_order[['order_idx', 'Datetime_x', "count_no", "Desc", "status", "Datetime_y", "price", "exeprice"]], left_on="BS_idx", right_on="order_idx", how="outer")
                
                print('.....df_BS....', df_BS)
                #df_b_c1=pd.merge(df_tx_b, df_order_b[['count_no','Datetime', "Desc"]],left_on='Datetime', right_on="Datetime", how="outer")      
                #df_b_c2=pd.merge(df_order_b, df_sl_b[["Datetime", "count_no", "price", "exeprice"]], left_on="Datetime", right_on="Datetime", how="outer")

             
                ################ end new source
                """
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

                """

                styles = [
                         {'if': {'column_id': 'Datetime',  'header_index': 0}, 'backgroundColor': 'blue' ,'fontWeight':'normal'},
                         ]
                
                layout = html.Div([dash_table.DataTable(
                         columns=[{'name': i, 'id': i} for i in df_BS.columns],
                         data=df_BS.to_dict('records'),
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
                         style_header_conditional=[
                                 {'if': {'column_id': 'Datetime',  'header_index': 0}, 'backgroundColor': 'blue' ,'fontWeight':'normal'},
                         ],
                        
                        )],
                    )

                return layout, 

                
                     

        #cc.register(app)
        return(app)

"""
if __name__ == '__main__':
    app.run_server()

"""




