from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash
from datetime import datetime
import os
import plotly
import pandas as pd
import pandas_ta as ta
from dash.exceptions import PreventUpdate
import numpy as np
from dash_extensions import WebSocket
import statsmodels.api as sm

def slope20(ser, n=20):
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

def macd_strategy(data):
    
    df = data.copy()
    df['macds1'] = df['MACD_12_26_9'].shift(-1)
    df['macdss1'] = df['MACDs_12_26_9'].shift(-1)
    #df['macdss1'] = df['MACDs_12_26_9'].shift(-1)
    
    df['buy_pt'] = np.where((((df.MACD_12_26_9 <= df.MACDs_12_26_9) & (df.macds1 > df.macdss1)) | 
                ((df.MACD_12_26_9 >= df.MACDs_12_26_9) & (df.macds1  < df.macdss1))), "y", "n")
    df['sell_pt'] = np.where((((df.MACD_12_26_9 <= df.MACDs_12_26_9) & (df.macds1 > df.macdss1)) | 
                ((df.MACD_12_26_9  >= df.MACDs_12_26_9) & (df.macds1 < df.macdss1))), "y", "n")
    
    df['buy_pt_s1'] = df['buy_pt'].shift(1)
    df['sell_pt_s1'] = df['sell_pt'].shift(1)
    
    
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
               
               


def create_dash(flask_app):
    
        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/macd/"), prevent_initial_callbacks=True)

        GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)
        
        
        app.layout = html.Div([
            
            ##### Store data
                dcc.Store(id='df_value'),
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
                    file = 'D://development//2022-06-23 NQ=Fout_stock_data'
                    pwd = os.getcwd()
                    os.chdir(os.path.dirname(file))
                    df = pd.read_csv(os.path.basename("2022-06-16 NQ=Fout_stock_data.csv"))
                    df.columns =['Datetime','Open','High','Low','Close']
                    df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
                    df.ta.rsi(close=df['Close'], length=14, append=True, signal_indicators=True)
                    
        
                return df
            
        
        
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
        
        @app.callback(
                Output('price_candlesticker', 'figure'),
                [Input('update_value', 'n_intervals')])
        
        def update_graph(n_intervals):
        
            if n_intervals == 0:
                raise PreventUpdate
            else:
                
                file = 'D://development//2022-06-23 NQ=Fout_stock_data'
                pwd = os.getcwd()
                os.chdir(os.path.dirname(file))
                df = pd.read_csv(os.path.basename(file))
                df.columns =['Datetime','Open','High','Low','Close']
                df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
                
                df['MA10'] = df.Close.rolling(10).mean()
                df['MA20'] = df.Close.rolling(20).mean()
                df['MA50'] = df.Close.rolling(50).mean()
                #df['Slope20'] = slope20(df.Close)
                
                max = df.Close.max()
                max_ind = df[['Close']].idxmax()
                min = df.Close.min()
                min_ind = df[['Close']].idxmin()
                max_20 = df.Close.tail(20).max()
                max_20_ind = df.Close.tail(20).idxmax()
                min_20 = df.Close.tail(20).min()
                min_20_ind = df.Close.tail(20).idxmin()
                
                buymacd, sellmacd = macd_strategy(df)

                buymacd = np.array(buymacd).tolist()
                flat_list = []
                for xs in buymacd:
                    for x in xs:
                        flat_list.append(x)
                        
                sellmacd = np.array(sellmacd).tolist()
                sell_macd = sum(sellmacd, [])

            
                
                return{
                    'data':[go.Scatter(x=df.index, y=df.Close, line=dict(color='#fc0080', width=2), 
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
                            
                            go.Scatter(x=df.index, y=df.MA10, line=dict(color='#f5bf42', width=1), 
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

                go.Scatter(x=df.index, y=flat_list, name="macd_up", mode="markers",
                              marker=dict(
                              symbol="3" ,
                              color="#eb68fc",
                              size=8)), 

                go.Scatter(x=df.index, y=sell_macd, name="macd_dn", mode="markers",
                              marker=dict(
                              symbol="4" ,
                              color="#6d68fc",
                              size=8)),

                                            
                    ],
                    
                    
                    'layout': go.Layout(
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
                 }

        @app.callback(
                    Output('price_macd', 'figure'),
                    [Input('update_value', 'n_intervals'),
                     Input('df_value', 'data')])
        
        def update_macd(n_intervals, df_value): 
                if n_intervals == 0:
                    raise PreventUpdate
                else:
                    import pandas_ta as ta
                    file = 'D://development//flask//stock//data//2022-06-28 NQ=Fout_stock_data.csv'
                    pwd = os.getcwd()
                    os.chdir(os.path.dirname(file))
                    df = pd.read_csv(os.path.basename(file))
                    df.columns =['Datetime','Open','High','Low','Close']
                    df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
                    
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
                                   margin = dict(t = 0  , r = 10, l = 60, b=0),
                        )
                              
                    }
                    
                
        @app.callback(
                Output('price_rsi', 'figure'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        def update_rsi(n_intervals, df_value):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                import pandas_ta as ta
                file = 'D://development//flask//stock//data//2022-06-28 NQ=Fout_stock_data.csv'
                pwd = os.getcwd()
                os.chdir(os.path.dirname(file))
                df = pd.read_csv(os.path.basename(file))
                df.columns =['Datetime','Open','High','Low','Close']
                df.ta.rsi(close=df['Close'], length=14, append=True, signal_indicators=True)
                
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
