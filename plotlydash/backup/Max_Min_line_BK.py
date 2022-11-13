from dash import html,dash_table
from dash import dcc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from flask_caching.backends import FileSystemCache
from dash_extensions.callback import CallbackCache, Trigger
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


def usny_curtime():
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    fmt = '%Y-%m-%d %H:%M:%S'
    time_stamp = nyc_datetime.strftime(fmt)
    return time_stamp




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

        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/maxminline/"), prevent_initial_callbacks=True)


        ### Preformance Turning
        cache = Cache(app.server, config={
            "CACHE_TYPE": "SimpleCache",
        })

        app.config.suppress_callback_exceptions = True
        timeout = 20
        cc = CallbackCache(cache=FileSystemCache(cache_dir="cache"))

        #app.css.append_css({"external_url": "/static/css/style.css"})
        app.css.config.serve_locally = True

        GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)


        app.layout = html.Div([

            html.Link(rel='stylesheet', href='/static/css/style.css'),
            
            ##### Store data
                dcc.Store(id='df_value'),
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
                   html.Div([
                       html.Div([
                           html.P('Buy/Short pft.:',
                                   style = {'color': '#bebfd6', 'font-size':'8px'},
                                   className = 'profit_label'),
                           html.P(['('], style = {'color': '#bebfd6', 'font-size':'14px'},
                                  className = 'stock_label'),
                            html.Div(id = 'tot_profit',
                                    style = {'color': '#bebfd6', 'font-size':'14px'},
                                    className = 'stock_label'),
                            html.P(['+'], style = {'color': '#bebfd6', 'font-size':'14px'},
                                   className = 'stock_label'),
                            html.Div(id = 'tot_long',
                                    style = {'color': '#bebfd6', 'font-size':'14px'},
                                    className = 'stock_label'),
                            html.P([')'], style = {'color': '#bebfd6', 'font-size':'14px'},
                                   className = 'stock_label'),
                        ],className = 'stock_profit'),


                    html.Div(id = 'profit'),
                    ],className = 'stock_score_label'),


                  html.Div([
                      html.Div(id = 'Img'),
                      ],className = 'profit_img'),
                  ], className = 'buy_sell_p'),
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
                        html.Div([
                            dcc.Graph(id = 'price_candlesticker', 
                                style={'width': 'auto',
                                           'height': '60vh',
                                           'border': '1px #5c5c5c solid',
                                           'margin-top' : "40px"},
                                    config = {'displayModeBar': True, 'responsive': True},
                                    className = 'chart_width'),

                            ]),
                        ]),

                        html.Div([
                            html.Title('MACD',
                            style = {'color': 'black'},
                            ),

                            dcc.Graph(id = 'price_macd', animate=False,
                                    style={'width': 'auto',
                                           'height': '20vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),

                        html.Div([
                            html.Title('Volume',
                            style = {'color': 'black'},
                            ),

                            dcc.Graph(id = 'price_vol', animate=False,
                                    style={'width': 'auto',
                                           'height': '10vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),


                        html.Div([
                            html.Title('RSI',
                            style = {'color': 'black'},
                            ),

                            dcc.Graph(id = 'price_rsi', animate=False,
                                    style={'width': 'auto',
                                           'height': '10vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),

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

        @cc.callback(Output('df_value', 'data'), Input('update_value', 'n_intervals'))
        @cache.memoize(timeout=timeout)
        
        def update_df(n_intervals):
                if n_intervals == 0:
                    raise PreventUpdate
                else:

                    p_filename = "_feather_stock_data.feather"

                    #time_stamp = datetime.now() - datetime.timedelta(hours=13)
                    #time_stamp =  datetime.now().strftime('%Y-%m-%d')
                    time_stamp = usny_curtime()

                    filename = os.path.join(str(time_stamp[0:11]) +"NQ=F USTime" + p_filename)
                    #filename = "2022-09-21 NQ=F USTime_out_stock_data.csv"
                    #filename = "2022-10-06 NQ=F USTime_feather_stock_data.feather"
                    cwd = os.getcwd()
                    path = os.path.dirname(cwd)

                    #file_path = path + "/jjhui/stock_app/data/"
                    file_path = path + "\\stock\\data\\"
                    file = os.path.join(file_path, filename)

                    #df = pd.read_csv(file, names =['Datetime','Open','High','Low','Close', 'Volume'])
                    df = pd.read_feather(file)
                    df.columns =['Datetime','Open','High','Low','Close', 'Volume']
                    
                    """
                    csv_file="D:\\Development\\flask dev\\stock\\data\\test.csv"
                    df.to_csv(csv_file)
                    """
                    
                    df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
                    df.ta.rsi(close=df['Close'], length=14, append=True, signal_indicators=True, xa=70, xb=30)



                return df.to_dict('records')


        @cc.callback(Output('get_date_time', 'children'),
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

        @cc.callback([
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
        def maxminline_strategy(n_intervals, data):
                if n_intervals == 0:
                    raise PreventUpdate
                else:
            
                    df = pd.DataFrame(data)
                
                    ##### MA #####
                    #df['MA10'] = df.Close.rolling(10).mean()
                    slopes10 = df.Close.rolling(10).apply(lambda s: linregress(s.reset_index())[0])
                    #slopes20 = df['Close'].rolling(20).apply(lambda s: linregress(s.reset_index())[0])
                    
                    df['slopes10'] = (np.rad2deg(np.arctan(np.array(slopes10))))
                    df['slopes10_sign']= np.where(df['slopes10'] > 0, 1, -1)
                    #df['slopes20'] = (np.rad2deg(np.arctan(np.array(slopes20))))
                    #((df['slopes10_sign'][idx_max_tmp].shift() == 1) & (df['slopes10_sign'][idx_max_tmp] == -1))
                    #((df['slopes10_sign'][idx_min_tmp].shift() == 1) & (df['slopes10_sign'][idx_min_tmp] == -1))
                    
                    ##### MACD #####
                
                    df['vol_gt2'] = np.where(((df['MACD_12_26_9'] > 2) | (df['MACD_12_26_9'] < -2)), "y", "n")
                    df['macdh_vol_gt1'] = np.where(((df['MACDh_12_26_9'] > 1) | (df['MACDh_12_26_9'] < -1)), "y", "n")
                
                
                    df['macd_above_up'] = np.where((df['MACD_12_26_9'] > 0)
                                                   & (df['MACDs_12_26_9'] > 0)
                                                   & (df['MACD_12_26_9'] > df['MACDs_12_26_9'] )
                                                   , "y", "n")
                    df['macd_above_dn'] = np.where((df['MACD_12_26_9'] > 0)
                                                   & (df['MACDs_12_26_9'] > 0)
                                                   & (df['MACD_12_26_9'] < df['MACDs_12_26_9'] )
                                                   , "y", "n")
                
                    df['macd_below_up'] = np.where((df['MACD_12_26_9'] < 0)
                                                   & (df['MACDs_12_26_9'] < 0)
                                                   & (df['MACD_12_26_9'] > df['MACDs_12_26_9'] )
                                                   , "y", "n")
                
                    df['macd_below_dn'] = np.where((df['MACD_12_26_9'] < 0)
                                                   & (df['MACDs_12_26_9'] < 0)
                                                   & (df['MACD_12_26_9'] < df['MACDs_12_26_9'] )
                                                   , "y", "n")
                
                
                
                
                    ############### NEW CHANGE of MIN/MAX start
                    
                    df.Datetime = pd.to_datetime(df.Datetime)
                    idx_max_tmp = df.resample('20min', on='Datetime')["Close"].agg(lambda x: np.nan if x.isna().all() else x.idxmax())
                    val_max_tmp = df.resample('20min', on='Datetime')["Close"].agg(lambda x: np.nan if x.isna().all() else x.max())
                
                    #idx_max = np.where(, idx_max_tmp, np.nan)
                    idx_max = idx_max_tmp
                        
                    #max_close_val = np.where(df.Close.isin(val_max), df.Close, np.nan)
                    #df['max_close_val'] = max_close_val
                    max_close_idx = np.where(df.index.isin(idx_max), df.Close, np.nan)
                    df['max_close_idx'] = max_close_idx
                    
                    idx_min_tmp = df.resample('20min', on='Datetime')["Close"].agg(lambda x: np.nan if x.isna().all() else x.idxmin())
                    val_min_tmp = df.resample('20min', on='Datetime')["Close"].agg(lambda x: np.nan if x.isna().all() else x.min())
                    

                    #idx_min = np.where(, idx_min_tmp, np.nan)
                    idx_min = idx_min_tmp
                
                    #min_close_val= np.where(df.Close.isin(val_min), df.Close, np.nan)
                    #df['min_close_val'] = min_close_val
                    min_close_idx = np.where(df.index.isin(idx_min), df.Close, np.nan)
                    df['min_close_idx'] = min_close_idx
                
                    ############### NEW CHANGE of MIN/MAX end
                
                    ####### BUY #######
                    T_Buy = np.where((df['Close'] == df['min_close_idx'] ) &
                                    (df['macdh_vol_gt1'] == 'y') 
                                    ((df['slopes10'] > -10) & (df['slopes10'] < 10) ).any() &
                                    ((df['macd_below_dn'] == 'y') | (df['macd_above_dn'] == 'y'))
                                    ,df['Close'], np.nan)
                    Buylist = np.where((T_Buy==df['Close']), 1, 0)
                    
                    ####### SELL #######
                
                    T_Sell = np.where((df['Close'] == df['max_close_idx']) &
                                   (df['macdh_vol_gt1'] == 'y') 
                                   ((df['slopes10'] > -10) & (df['slopes10'] < 10) ).any() &
                                   ((df['macd_above_up'] == 'y') | (df['macd_below_up'] == 'y'))
                                     ,df['Close'], np.nan)
                
                    Selllist = np.where((T_Sell==df['Close']), -1, 0)
                    
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
                    
                    
                    # total Buy/Buy count
                    Buy_list = list(Buy)
                    df_Buy = pd.DataFrame(Buy_list, columns=['Buy'])
                    #df_Buy = df_Buy.T
                    df = pd.concat([df, df_Buy], axis=1)
                    df['Buy_zero'] = np.where(df.Buy == "NaN", 0, df.Buy)
                    tot_buy = np.add.accumulate(df['Buy_zero'].astype(float))
                
                    df['Buy_count_tmp'] = np.where(df['Buy'] == 'NaN', 0, 1)
                    df['Buy_count'] = np.add.accumulate(df['Buy_count_tmp'])
                    """
                    df['Buy_count_compare'] = np.where(df['Buy'] == 'NaN', np.nan, 1)
                    Buy_tot_count = df['Buy_count_compare'].count()
                    """
                    # total SELL/count
                
                    Sell_list = list(Sell)
                    df_Sell = pd.DataFrame(Sell_list, columns=['Sell'])
                    #df_Sell = df_Sell.T
                    df = pd.concat([df, df_Sell], axis=1)
                    df['Sell_zero'] = np.where(df.Sell == "NaN", 0, df.Sell)
                    tot_sell = np.add.accumulate(df['Sell_zero'].astype(float))
                
                
                    df['Sell_count_tmp'] = np.where(df['Sell'] == 'NaN', 0, 1)
                    df['Sell_count'] = np.add.accumulate(df['Sell_count_tmp'])
                    
                    Sell_count = abs(df.Sell_count)
                    Buy_count = abs(df.Buy_count)
                
                    Buy=Buy.tolist()
                    Sell=Sell.tolist()
                    
                    tot_buy = tot_buy.iloc[-1]
                    tot_sell =  tot_sell.iloc[-1]
                    Buy_count =  Buy_count.iloc[-1]
                    Sell_count =  Sell_count.iloc[-1]
                    
                    df.to_csv("maxminline.csv")
                    
                    return tot_buy, tot_sell, Buy_count, Sell_count, Buy, Sell, 




        @cc.callback(
                Output('price_candlesticker', 'figure'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data'),
                 Input('buy', 'children'),
                 Input('sell', 'children')])
        
        @cache.memoize(timeout=timeout)
        def update_graph(n_intervals, data, buy, sell):

            if n_intervals == 0:
                raise PreventUpdate
            else:

                df = pd.DataFrame(data)

                df['MA10'] = df.Close.rolling(10).mean()
                df['MA20'] = df.Close.rolling(20).mean()
                df['MA50'] = df.Close.rolling(50).mean()

                max = df.Close.max()
                #max_ind = df[['Close']].idxmax()
                min = df.Close.min()
                #min_ind = df[['Close']].idxmin()
                max_20 = df.Close.tail(20).max()
                #max_20_ind = df.Close.tail(20).idxmax()
                min_20 = df.Close.tail(20).min()
                #min_20_ind = df.Close.tail(20).idxmin()


                #t_buy, t_sell, buy, sell, tot_buy, tot_sell, buy_count, sell_count = maxminline_strategy(df)
                """
                tt_buy = tot_buy.iloc[-1]
                tt_sell =  tot_sell.iloc[-1]
                bbuy_count =  buy_count.iloc[-1]
                ssell_count =  sell_count.iloc[-1]
                """
                buyitem = np.array(buy).tolist()

                sellitem = np.array(sell).tolist()


                figure = go.Figure(
                    data = [
                            go.Scattergl(x=df.index, y=df.Close, line=dict(color='#fc0080', width=1.5),
                            name = 'Close',
                            hoverinfo = 'text',
                            hovertext =
                            '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                            '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.Close] + '<br>'),
                            
                            go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                increasing={'line_width': 1, 'line_color': "#de0740", 'fillcolor': "#de0740"},
                                decreasing={'line_width': 1, 'line_color': '#04bd1c', 'fillcolor': '#04bd1c'}),

                            go.Scattergl(x=df.index, y=df.MA10, line=dict(color='#AA76DB', width=1),
                                name = 'MA10',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA10] + '<br>'),
                            go.Scattergl(x=df.index, y=df.MA20, line=dict(color='#2ed9ff', width=1),
                                name = 'MA20',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA20] + '<br>'),
                            go.Scattergl(x=df.index, y=df.MA50, line=dict(color='#b6e880', width=1),
                                name = 'MA50',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA50] + '<br>'),


                go.Scattergl(x=[0, len(df)],
                         y=[min,min], name='min',
                         line=dict(color='rgba(152,78,163,0.5)', width=1, dash='dash'),
                         ),

                go.Scattergl(x=[0, len(df)],
                         y=[max,max], name='max',
                         line=dict(color='rgba(152,78,163,0.5)', width=1, dash='dash'),
                         ),

                go.Scattergl(x=[0, len(df)],
                         y=[min_20,min_20], name='min20',
                         line=dict(color='rgba(124,124,124,0.5)', width=1, dash='dash'),
                         ),

                go.Scattergl(x=[0, len(df)],
                         y=[max_20,max_20], name='max20',
                         line=dict(color='rgba(124,124,124,0.5)', width=1, dash='dash'),
                         ),


                go.Scattergl(x=df.index, y=buyitem, name="UP", mode="markers",
                              marker=dict(
                              symbol="5" ,
                              color="#FDDE00",
                              size=12)),

                go.Scattergl(x=df.index, y=sellitem, name="DOWN", mode="markers",
                              marker=dict(
                              symbol="6" ,
                              color="#76C7DB",
                              size=12)),

                    ],)

                
                figure.update_layout(
                #layout = go.Layout(
                   xaxis_rangeslider_visible=False,
                   hovermode = 'closest',
                   uirevision = 'dataset',
                   margin = dict(t = 35  , r = 0, l = 60, b=20),
                   xaxis = dict(
                                #range=[1,200],
                                autorange= True,
                                color = 'black',
                                matches = 'x',
                                showspikes = True,
                                showline = True,
                                showgrid = True,
                                linecolor = 'black',   
                                linewidth = 1,
                                ticks = 'outside',
                                tickfont = dict(
                                    family = 'Arial',
                                    size = 10,
                                    color = 'black'
                                )),
                   yaxis = dict(
                                #range=[min(df.Close),max(df.Close)],
                                autorange= True,
                                showspikes = True,
                                showline = True,
                                showgrid = False,
                                linecolor = 'black',
                                linewidth = 1,
                                ticks = 'outside',
                                tickfont = dict(
                                    family = 'Arial',
                                    size = 10,
                                    color = 'black'
                       )),
                   
                    transition=dict(
                            duration = 500,
                            easing = 'cubic-in-out',
                        ),
                   font = dict(
                       family = 'sans-serif',
                       size = 10,
                       color = 'black'
                   )
                
                   )
                 
                return figure
        """
        @app.callback(Output('price_candlesticker', 'extendData'), 
                      [Input('interval', 'n_intervals')], 
                      [State('price_candlesticker', "figure")])
        def update_data(n_intervals, existing):
        
        # work directly with existing['data'][0] and determine 
        # if you need to append to it or modify its [-1] entry here

            return [
                    {
                        "Datetime": [existing["data"][0]["Datetime"]],
                        "Open": [existing["data"][0]["Open"]],
                        "High": [existing["data"][0]["High"] ],
                        "Low": [existing["data"][0]["Low"]],
                        "Close": [existing["data"][0]["Close"]]
                    },
                    [0],
                    len(existing["data"][0]["Open"]) 
                ]  
        """
        

        @cc.callback(
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
                    ave_b = tot_tbuy / buy_tcount
                    ave_b = np.around(ave_b, 2)
                    buy_count = buy_tcount
                    tot_buy = tot_tbuy

                    ave_s = tot_tsell / sell_tcount
                    ave_s = np.around(ave_s, 2)
                    sell_count = sell_tcount
                    tot_sell = tot_tsell

                    buy_count = buy_tcount
                    sell_count = sell_tcount
                    print("sell_count & buy_count", buy_count, sell_count)
                    return [tot_buy, tot_sell, buy_count, sell_count, ave_b, ave_s]


        @cc.callback(
                    Output('price_macd', 'figure'),
                    [Input('update_value', 'n_intervals'),
                     Input('df_value', 'data')])

        def update_macd(n_intervals, data):
                if n_intervals == 0:
                    raise PreventUpdate
                else:
                    df = pd.DataFrame(data)

                    return{
                        "data" : [
                                    go.Scatter(
                                            x=df.index,
                                            y=df['MACD_12_26_9'],
                                            line=dict(color='#ff9900', width=1),
                                            name='macd',
                                            # showlegend=False,
                                            legendgroup='2',),

                                    go.Scatter(
                                            x=df.index,
                                            y=df['MACDs_12_26_9'],
                                            line=dict(color='#000000', width=1),
                                            # showlegend=False,
                                            legendgroup='2',
                                            name='signal'),
                                    go.Bar(
                                            x=df.index,
                                            y=df['MACDh_12_26_9'],
                                        marker_color=np.where(df['MACDh_12_26_9'] < 0, '#000', '#ff9900'),
                                        name='bar'),

                                    go.Scatter(x=[0, len(df)],
                                         y=[-5,-5], showlegend=False,
                                         line=dict(color='#000000', width=1, dash='dash'),
                                 ),


                                 ],

                        "layout" : go.Layout(
                                   hovermode = 'x unified',
                                   uirevision = 'dataset',
                                   margin = dict(t = 0  , r = 0, l = 0, b=0),
                        )

                    }


        @cc.callback(
                Output('price_rsi', 'figure'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])

        def update_rsi(n_intervals, data):
            if n_intervals == 0:
                raise PreventUpdate
            else:

                df = pd.DataFrame(data)

                return {'data': [go.Scatter(x=df.index, y=df.RSI_14, name='RSI',
                                 line=dict(color='#000000', width=1),
                                 # showlegend=False,
                                 legendgroup='3'),

                                 go.Scatter(x=[0, len(df)],
                                 y=[20,20], name='OB(20)',
                                 line=dict(color='#f705c3', width=2, dash='dash'),
                                 ),

                                 go.Scatter(x=[0, len(df)],
                                 y=[80,80], name='OS(80)',
                                 line=dict(color='#f705c3', width=2, dash='dash'),
                                 ),

                                 go.Scatter(x=[0, len(df)],
                                     y=[50,50], showlegend=False,
                                     line=dict(color='#000000', width=1, dash='dash')),
                                 ],

                         'layout': go.Layout(
                                            hovermode = 'x unified',
                                            uirevision = 'dataset',
                                            margin = dict(t = 0  , r = 0, l = 60, b=0),

                            )

                       }

        @cc.callback(
                Output('price_vol', 'figure'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])

        def update_vol(n_intervals, data):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                df = pd.DataFrame(data)
                delta_vol = df.Volume - df.Volume.shift(1)
                delta_vol[delta_vol < 0] = 0

                return {'data': [go.Bar(x=df.index, y=delta_vol, name='volume',
                                  showlegend=True,
                                 legendgroup='4'),

                                 ],

                         'layout': go.Layout(
                                            hovermode = 'x unified',
                                            uirevision = 'dataset',
                                            margin = dict(t = 0  , r = 0, l = 60, b=0),

                            )

                       }

        @cc.callback(
                [
                 Output('tot_profit', 'children'),
                 Output('tot_long', 'children'),
                 Output('buy_nplist', 'children'),
                 Output('sell_nplist', 'children'),
                ],
                [Input('update_value', 'n_intervals'),
                State('buy', 'children'),
                State('sell', 'children'),
                 ])

        def update_profitchart(n_intervals, buy, sell):
            if n_intervals == 0:
                raise PreventUpdate
            else:

                buyitem = np.array(buy).tolist()
                sellitem = np.array(sell).tolist()

                ###### Calculate profit START

                buy_list = []
                buy_idx = []
                for idx, xs in enumerate(buyitem):
                    if xs != 'NaN':
                        buy_list.append(xs)
                        buy_idx.append(idx)

                sell_list = []
                sell_idx = []
                for idx, xs in enumerate(sellitem):
                    if xs != 'NaN':
                        sell_list.append(xs)
                        sell_idx.append(idx)

                buy_nplist = np.column_stack((buy_list, buy_idx))
                sell_nplist =  np.column_stack((sell_list, sell_idx))

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
                        print("buy_profit_idx",i," ",(buy_nplist[i,1]),"<", sell_nplist[i,1],buy_profit)
                        print("buy_profit",i," ",(buy_nplist[i,0]),"<", sell_nplist[i,0],buy_profit, tot_profit)
                    else:
                        long_profit = float(sell_nplist[i,0]) - float(buy_nplist[i,0])
                        tot_long +=  long_profit
                        print("long_profit_idx",i," ",buy_nplist[i,1],">", (sell_nplist[i,1]), long_profit)
                        print("long_profit",i," ",buy_nplist[i,0],">", (sell_nplist[i,0]), long_profit, tot_long)

                    #sell -> buy

                    print("total", tot_profit, tot_long)


                ###### Calculate profit FINISH

                return [tot_profit, tot_long, buy_nplist, sell_nplist]
 

        @cc.callback([Output('profit', 'children'),
                       Output('Img', 'children'),
                       ],
                      [Input('update_value', 'n_intervals'),
                       Input('tot_profit', 'children'),
                       Input('tot_long', 'children'),
                       ])
        @cache.memoize(timeout=timeout)
        def profit(n_intervals, tot_profit, tot_long):

            if n_intervals == 0:
                raise PreventUpdate
            else:
                #sell_gain = Decimal(str(tot_long)).quantize(Decimal('0.01'))
                #buy_gain = Decimal(str(tot_profit)).quantize(Decimal('0.01'))
                profit = float(tot_long) + float(tot_profit)


                if (profit > 0):
                        return [
                                html.H6('${0:,.2f}'.format(profit),style = {'color': '#f20540', 'fontSize' : 17, 'margin-top': '11px'}),
                                html.Img(id = "Img",src = app.get_asset_url('money-bag.png'),
                                     style = {'height': '30px'},
                                     className = 'coin'), ]

                else:
                        return [
                            html.H6('${0:,.2f}'.format(profit),style = {'color': '#f20540', 'fontSize' : 17, 'margin-top': '11px'}),
                            html.Img(id = "Img",src = app.get_asset_url('cry.png'),
                                     style = {'height': '30px'},
                                     className = 'coin'), ]


        @cc.callback(
                Output('table-container', 'children'),
                [Input('update_value', 'n_intervals'),
                 Input('buy_nplist', 'children'),
                 Input('sell_nplist', 'children'),
                 ])
        @cache.memoize(timeout=timeout)
        
        def update_table(n_intervals, buy_nplist, sell_nplist):

            if n_intervals == 0:
                raise PreventUpdate
            else:
                df_b = pd.DataFrame(buy_nplist, columns=['buy', 'b_idx'])
                df_s = pd.DataFrame(sell_nplist, columns=['sell', 's_idx'])

                df_c = pd.concat([df_b, df_s], axis=1)
                df_c = df_c.fillna(0)
                df_c.b_idx = df_c.b_idx.astype(int)
                df_c.s_idx = df_c.s_idx.astype(int)
                df_c.sell = df_c.sell.astype(float)
                df_c.buy = df_c.buy.astype(float)

                profit = np.where(df_c.b_idx > df_c.s_idx,
                                      df_c.sell - df_c.buy,
                                      df_c.sell - df_c.buy,
                                       )
                df_c['profit'] = profit


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
                            'color': 'rgb(255,255,255, 0.5)',
                            'fontWeight': 'normal'
                        },
                        style_data_conditional=styles,
                        )])

                return layout

        cc.register(app)
        return(app)

"""
if __name__ == '__main__':
    app.run_server()

"""




