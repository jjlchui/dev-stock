from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash
import datetime
import os
import plotly
import pandas as pd
import pandas_ta as ta
from dash.exceptions import PreventUpdate
import numpy as np
from dash_extensions import WebSocket
import statsmodels.api as sm
from decimal import Decimal
from scipy.stats import linregress





def maxminline_strategy(data):
    df = data.copy()
    
    ##### MA #####

    slopes = df['Close'].rolling(10).apply(lambda s: linregress(s.reset_index())[0])
    slopes20 = df['Close'].rolling(20).apply(lambda s: linregress(s.reset_index())[0])
    df['slopes'] = (np.rad2deg(np.arctan(np.array(slopes))))
    df['slopes20'] = (np.rad2deg(np.arctan(np.array(slopes20))))
    df['slopess1p'] = df['slopes'].shift(1)
    df['slopes20s1p'] = df['slopes20'].shift(1)
    
    ##### MACD #####
          
    df['vol_gt2'] = np.where(((df['MACD_12_26_9'] > 1) | (df['MACD_12_26_9'] < -1)), "y", "n") 
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
    
    
   
    ##### SUPPORT & AGAINST LINE #####
    
    """
    #df.apply(lambda x:x[df.Close.values].rolling(10).max())
    df['last_max_20s1'] = df['Close'].rolling(20).apply(lambda x: x[::-1].max()).shift(1)
    df['last_min_20s1'] = df['Close'].rolling(20).apply(lambda x: x[::-1].min()).shift(1)
    
    df['cum_max'] = df.Close.cummax()
    df['cum_maxs1'] = df['cum_max'].shift(1)
    df['cum_min'] = df.Close.cummin()
    df['cum_mins1'] = df['cum_min'].shift(1)
    df['cum_min_mins'] = df[['cum_min', 'cum_mins1']].apply(lambda x: "nan" if x['cum_min'] == x['cum_mins1'] else x['cum_min'], axis=1)
    df['cum_max_maxs'] = df[['cum_max', 'cum_maxs1']].apply(lambda x: "nan" if x['cum_max'] == x['cum_maxs1'] else x['cum_max'], axis=1)
    """

    
    #df[df.index.isin(df.groupby(df.index // 5)["Close"].idxmax())]

    df.Datetime = pd.to_datetime(df.Datetime)
    
    idx_max = df.resample('10min', on='Datetime').agg({'Close': 'max'})
    idx_max = idx_max.rename(columns={'Datetime':'Datetime', 'Close':'max_Close'})
    df = df.merge(idx_max, left_on='Datetime', right_on='Datetime', how='left')
    df['max_Close'] = df['max_Close'].fillna(method='ffill')
    
    idx_min = df.resample('10min', on='Datetime').agg({'Close': 'min'})
    idx_min = idx_min.rename(columns={'Datetime':'Datetime', 'Close':'min_Close'})
    df = df.merge(idx_min, left_on='Datetime', right_on='Datetime', how='left')
    df['min_Close'] = df['min_Close'].fillna(method='ffill')
    
    #df['max_30m']= df.resample('5min', on='Datetime').max().reindex(df.index, method='ffill')
    #df['max_30m'] = df.Close.resample('30min', on='Datetime', convention='e').agg({'Close': 'max'}).reindex(df.index, method='ffill')
    #df['max_30m'] = df.Close.resample('30min', on='Datetime').agg({'Close': 'max'}).date.transform("last")
    ####### BUY #######
    Buy = np.where((df['Close'] == df['min_Close']) &
                    (df['macdh_vol_gt1'] == 'y') &
                    (slopes20 > 0) &
                    ((df['macd_below_dn'] == 'y') | (df['macd_above_dn'] == 'y'))
                    ,df['Close'], "NaN")


    
    # sell compare with the buy price
    buy_price=np.where(Buy =='NaN', "NaN", Buy)
    buy_price = buy_price.T
    df.reset_index()
    df['buy_price'] = buy_price
    df['buy_price'].fillna(method='ffill', inplace=True)
    
    # total Buy/Buy count
    Buy_list = list(Buy)
    df_Buy = pd.DataFrame(Buy_list, columns=['Buy'])
    #df_Buy = df_Buy.T
    df = pd.concat([df, df_Buy], axis=1)
    df['Buy_zero'] = np.where(df.Buy == "NaN", 0, df.Buy)
    tot_buy = np.add.accumulate(df['Buy_zero'].astype(float))
    
    df['Buy_count_tmp'] = np.where(df['Buy'] == 'NaN', 0, 1)
    df['Buy_count'] = np.add.accumulate(df['Buy_count_tmp'])
    
    df['Buy_count_compare'] = np.where(df['Buy'] == 'NaN', np.nan, 1)
    Buy_tot_count = df['Buy_count_compare'].count()
        
    
    ####### SELL #######

  

    
    Sell = np.where((df['Close'] == df['max_Close']) &
                   (df['macdh_vol_gt1'] == 'y') &
                   (slopes20 > 0) &
                   ((df['macd_above_up'] == 'y') | (df['macd_below_up'] == 'y'))
                     ,df['Close'], "NaN")
    
    # total SELL/count
    
    Sell_list = list(Sell)
    df_Sell = pd.DataFrame(Sell_list, columns=['Sell'])
    #df_Sell = df_Sell.T
    df = pd.concat([df, df_Sell], axis=1)
    df['Sell_zero'] = np.where(df.Sell == "NaN", 0, df.Sell)
    tot_sell = np.add.accumulate(df['Sell_zero'].astype(float))
    
    
    df['Sell_count_tmp'] = np.where(df['Sell'] == 'NaN', 0, 1)
    df['Sell_count'] = np.add.accumulate(df['Sell_count_tmp'])
    
      
    df.Sell = pd.to_numeric(df.Sell, errors='coerce') 

    #Sell_single = Sell_single.astype(float)
    Sell_count = abs(df.Sell_count)
    Buy_count = abs(df.Buy_count)
    

    df.to_csv("maxminline.csv")
    return Buy, Sell, tot_buy, tot_sell, Buy_count, Sell_count 
         
               


def create_dash(flask_app):
    
        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/maxminline/"), prevent_initial_callbacks=True)

        GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)
        
        
        app.layout = html.Div([
            
            ##### Store data
                dcc.Store(id='df_value'),
                dcc.Store(id='ave_b'),
                dcc.Store(id='ave_s'),
              #### Header 
                 html.Div([
                 html.Div([
                     html.Img(src = app.get_asset_url('logo.jpeg'),
                              style = {'height': '30px'},
                              className = 'title_image'),
                     html.H6('Have Fun !!! with stock ...',
                             style = {'color': 'white'},
                             className = 'title'),
         
                 ], className = 'logo_title'),
                 
                 html.Div([
                   html.Div([
                       html.Div([
                           html.P('Buy:',
                                   style = {'color': 'white', 'font-size':'12px'},
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
                       html.Div(id = 'ave_buy',
                               style = {'color': 'white', },
                               className = 'stock_ave'),
                       html.P([')'], className = 'stock_score'),
                        
                   ], className = 'buy_sell_b'),
                    
                   html.Div([
                   html.Div([
                       html.P('Sell:',
                               style = {'color': 'white', 'font-size':'12px'},
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
                       html.Div(id = 'ave_sell',
                               style = {'color': 'white', },
                               className = 'stock_ave'),
                       html.P([')'], className = 'stock_score'),
                   ], className = 'buy_sell_s'),
                  
                 html.Div([
                   html.Div([
                       html.P('Profit:',
                               style = {'color': 'white', 'font-size':'12px'},
                               className = 'stock_label'),
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
                            dcc.Graph(id = 'price_candlesticker', animate=False, 
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
                            html.Title('RSI',
                            style = {'color': 'black'},
                            ),
                            
                            dcc.Graph(id = 'price_rsi', animate=False, 
                                    style={'width': 'auto', 
                                           'height': '20vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),


        
                        dcc.Interval(id = 'update_value',
                                         interval = int(GRAPH_INTERVAL) ,
                                         n_intervals = 0)
                    ]),
        
            
                
        ])
        
        @app.callback(Output('df_value', 'data'), Input('update_value', 'n_intervals'))
        
        def update_df(n_intervals): 
                if n_intervals == 0:
                    raise PreventUpdate
                else:
                    
                    p_filename = "_out_stock_data.csv"
                    
                    #time_stamp = datetime.datetime.now() - datetime.timedelta(hours=13)
                    time_stamp =  datetime.datetime.now().strftime('%Y-%m-%d')
                    #time_stamp = datetime.now().strftime('%Y-%m-%d')
                    filename = time_stamp+" NQ=F USTime"+p_filename
                    
                    #filename = "2022-08-23 NQ=F USTime_out_stock_data.csv"
                    cwd = os.getcwd()
                    path = os.path.dirname(cwd)
                
                    file_path = path + "\\stock\\data\\"
                    file = os.path.join(file_path, filename)
                    #df = pd.read_csv(os.path.basename(filename))
                    df = pd.read_csv(file)
                    df.columns =['Datetime','Open','High','Low','Close']
                    df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
                    df.ta.rsi(close=df['Close'], length=14, append=True, signal_indicators=True, xa=70, xb=30)
                    
        
                return df.to_dict('records')
        

        @app.callback(Output('get_date_time', 'children'),
                      [Input('update_date_time', 'n_intervals')])
        def live_date_time(n_intervals):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                now = datetime.datetime.now()
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        
            return [
                html.Div(dt_string)
            ]
        
        @app.callback([
                Output('price_candlesticker', 'figure'),
                Output('tot_buy', 'children'),
                Output('tot_sell', 'children'),
                Output('buy_count', 'children'),
                Output('sell_count', 'children'),
                Output('ave_b', 'children'),
                Output('ave_s', 'children'),
                ],
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        def update_graph(n_intervals, data):
        
            if n_intervals == 0:
                raise PreventUpdate
            else:
                
                df = pd.DataFrame(data)
                
                df['MA10'] = df.Close.rolling(10).mean()
                df['MA20'] = df.Close.rolling(20).mean()
                df['MA50'] = df.Close.rolling(50).mean()
                
                max = df.Close.max()
                max_ind = df[['Close']].idxmax()
                min = df.Close.min()
                min_ind = df[['Close']].idxmin()
                max_20 = df.Close.tail(20).max()
                max_20_ind = df.Close.tail(20).idxmax()
                min_20 = df.Close.tail(20).min()
                min_20_ind = df.Close.tail(20).idxmin()
                
                
                buy, sell, tot_buy, tot_sell, buy_count, sell_count = maxminline_strategy(df)
                
                tt_buy = tot_buy.iloc[-1]
                tt_sell =  tot_sell.iloc[-1]
                bbuy_count =  buy_count.iloc[-1]
                ssell_count =  sell_count.iloc[-1]
                
                ave_b = tt_buy / bbuy_count
                ave_b = np.around(ave_b, 2)


                ave_s = tt_sell / ssell_count
                ave_s = np.around(ave_s, 2)

                
                buymacd = np.array(buy).tolist()
                buy_macd = []
                for xs in buymacd:
                    for x in xs:
                        buy_macd.append(x)
                        
                sellmacd = np.array(sell).tolist()
                #sell_macd = sum(sellmacd, [])                
                sell_macd = []
                for xs in sellmacd:
                    for x in xs:
                        sell_macd.append(x)
                
            
                
                figure = go.Figure(
                    data = [go.Scatter(x=df.index, y=df.Close, line=dict(color='#fc0080', width=2), 
                            name = 'Close',
                            hoverinfo = 'text',
                            hovertext =
                            '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                            '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.Close] + '<br>'),
                    
                            go.Candlestick(x=df.index,
                                open=df.Open,
                                high=df.High,
                                low=df.Low,
                                close=df.Close),  
                            
                            go.Scatter(x=df.index, y=df.MA10, line=dict(color='#AA76DB', width=1), 
                                name = 'MA10',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA10] + '<br>'),
                            go.Scatter(x=df.index, y=df.MA20, line=dict(color='#2ed9ff', width=1),
                                name = 'MA20',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA20] + '<br>'),
                            go.Scatter(x=df.index, y=df.MA50, line=dict(color='#b6e880', width=1),
                                name = 'MA50',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA50] + '<br>'),
                        
            
                go.Scatter(x=[0, len(df)], 
                         y=[min,min], name='min',
                         line=dict(color='rgba(152,78,163,0.5)', width=1, dash='dash'),
                         ),
            
                go.Scatter(x=[0, len(df)], 
                         y=[max,max], name='max',
                         line=dict(color='rgba(152,78,163,0.5)', width=1, dash='dash'),
                         ),
            
                go.Scatter(x=[0, len(df)], 
                         y=[min_20,min_20], name='min20',
                         line=dict(color='rgba(124,124,124,0.5)', width=1, dash='dash'),
                         ),
            
                go.Scatter(x=[0, len(df)], 
                         y=[max_20,max_20], name='max20',
                         line=dict(color='rgba(124,124,124,0.5)', width=1, dash='dash'),
                         ),

                go.Scatter(x=df.index, y=buymacd, name="UP", mode="markers",
                              marker=dict(
                              symbol="5" ,
                              color="#FDDE00",
                              size=12)), 

                go.Scatter(x=df.index, y=sellmacd, name="DOWN", mode="markers",
                              marker=dict(
                              symbol="6" ,
                              color="#76C7DB",
                              size=12)),

                                            
                    ],
                    
                    
                    layout = go.Layout(
                                   xaxis_rangeslider_visible=False,
                                   hovermode = 'closest',
                                   uirevision = 'dataset',
                                   margin = dict(t = 35  , r = 0, l = 60, b=20),
                                   xaxis = dict(autorange= True,
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
                                   yaxis = dict(autorange= True,
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
        
                                   font = dict(
                                       family = 'sans-serif',
                                       size = 10,
                                       color = 'black'
                                   )
                                   
                                   )
                 )
                return [figure, tt_buy, tt_sell, bbuy_count, ssell_count, ave_b, ave_s] 
            
            
        @app.callback([Output('profit', 'children'),
                       Output('Img', 'children'),
                       Output('ave_buy', 'children'),
                       Output('ave_sell', 'children'),
                       ],
                      [Input('update_value', 'n_intervals'),
                       Input('ave_b', 'children'),
                       Input('ave_s', 'children'),
                       ])
        
        def profit(n_intervals,ave_bb, ave_ss):
            
            if n_intervals == 0:
                raise PreventUpdate
            else:
                ave_ss = Decimal(str(ave_ss)).quantize(Decimal('0.01'))
                ave_bb = Decimal(str(ave_bb)).quantize(Decimal('0.01'))
                profit = ave_ss - ave_bb
    
                
                if (profit > 0):
                        return [html.H6('${0:,.2f}'.format(profit), style = {'color': '#84de02', 'fontSize' : 17}),
                                html.Img(id = "Img",src = app.get_asset_url('money-bag.png'),
                                     style = {'height': '30px'},
                                     className = 'coin'), ave_bb, ave_ss]
                    
                else:
                        return [
                            html.H6(profit, style = {'color': '#f20540', 'fontSize' : 17}),
                            html.Img(id = "Img",src = app.get_asset_url('cry.png'),
                                     style = {'height': '30px'},
                                     className = 'coin'), ave_bb, ave_ss]
                

        @app.callback(
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
                                   margin = dict(t = 0  , r = 0, l = 60, b=0),
                        )
                              
                    }
                    
                
        @app.callback(
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


 
        return(app)

"""
if __name__ == '__main__':
    app.run_server()
    
"""
