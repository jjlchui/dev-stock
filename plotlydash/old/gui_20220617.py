from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash
from flask import Flask, send_from_directory
from datetime import datetime
import os
import pandas as pd
from dash.exceptions import PreventUpdate
import numpy as np
import statsmodels.api as sm
from sstrategy import get_db, slope, ma_strategy, macd_strategy

GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)



white_button_style = {'background-color': 'white',
              'color': 'black'}
red_button_style = {'background-color': 'red',
            'color': 'white'}


server = Flask(__name__)
app = dash.Dash(__name__,server=server)


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

    
        html.Div([
            dcc.Store(id="store"),
            html.Div([ 
                html.Button('MA', id='btn-nclicks-1', n_clicks=0, style=white_button_style ),
                html.Button('MACD', id='btn-nclicks-2', n_clicks=0, style=white_button_style),
                html.Button('Ai', id='btn-nclicks-3', n_clicks=0, style=white_button_style ),
            ],className = 'btn'),    
        ],className = 'button_container'),
    

    
    ##### Graph Display
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id = 'price_candlesticker', animate=False, 
                            style={'width': 'auto', 
                                   'height': '60vh',
                                   'border': '1px #5c5c5c solid',
                                   'margin-top' : "80px"},
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
            df = get_db()

            """
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df = df.resample('1T', on='Datetime').mean()
            df['Datetime'] = df.index
            df = df.reset_index(drop=True)
            """

            df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
            df.ta.rsi(close=df['Close'], length=14, append=True, signal_indicators=True)
            
            df = df.to_json(orient='split')

        return df
    
@app.callback(
    Output('btn-nclicks-1', 'style'),
    Output('btn-nclicks-2', 'style'),
    Output('btn-nclicks-3', 'style'),
    Output("store", "data"),
    Input('btn-nclicks-1', 'n_clicks'),
    Input('btn-nclicks-2', 'n_clicks'),
    Input('btn-nclicks-3', 'n_clicks')
)
def displayClick(btn1, btn2, btn3):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        return red_button_style, white_button_style, white_button_style, "MA"
    elif 'btn-nclicks-2' in changed_id:
        return white_button_style, red_button_style, white_button_style, "MACD"
    elif 'btn-nclicks-3' in changed_id:
        return white_button_style, white_button_style, red_button_style, "AI"
    else:
        return white_button_style


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
        [Input('update_value', 'n_intervals'),
         Input("store", "data"),
         Input("df_value", "data")])

def update_graph(n_intervals, store, df_value):

    if n_intervals == 0:
        raise PreventUpdate
    else:
        #df = get_db()
        df = pd.read_json(df_value,orient='split')
        #print('.....df.....', df)
        
        if (store == "MA"):
            print("....MA....")

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

        buy, sell = ma_strategy(df) 
        buymacd, sellmacd = macd_strategy(df)
        
        buymacd = np.array(buymacd).tolist()
        flat_list = []
        for xs in buymacd:
            for x in xs:
                flat_list.append(x)
                
        sellmacd = np.array(sellmacd).tolist()
        sell_macd = sum(sellmacd, [])

       

        return{
            
            
        'data':[
                go.Candlestick(x=df.index,
                    open=df.Open,
                    high=df.High,
                    low=df.Low,
                    close=df.Close),  
            
                go.Scatter(x=df.index, y=df.Close, line=dict(color='#fc0080', width=2), 
                    name = 'Close',
                    hoverinfo = 'text',
                    hovertext =
                    '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                    '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.Close] + '<br>'),
            
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



                   go.Scatter(x=df.index, y=buy, name="buy", mode="markers",
                          marker=dict(
                          symbol="5" ,
                          color="MediumPurple",
                          size=14)), 
            
                    go.Scatter(x=df.index, y=sell, name="sell", mode="markers",
                          marker=dict(
                          symbol="6" ,
                          color="LightSkyBlue",
                          size=14)),

            
                
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
            
        'layout' : go.Layout(
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
            """
            df = get_db()
            df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
            """
            df = pd.read_json(df_value,orient='split')
            
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
                                 y=[0,0], showlegend=False,
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
        """
        traces = []
        df = get_db()
        df.ta.rsi(close=df['Close'], length=14, append=True, signal_indicators=True)
        """
        df = pd.read_json(df_value,orient='split')
        
        return {'data': [go.Scatter(x=df.index, y=df.RSI_14, name='RSI',
                         line=dict(color='#000000', width=1),
                         # showlegend=False,
                         legendgroup='3'),
                         
                         go.Scatter(x=[min(df.index), max(df.index)], 
                         y=[20,20], name='overbought(20)',
                         line=dict(color='#f705c3', width=2, dash='dash'),
                         ),
                
                         go.Scatter(x=[min(df.index), max(df.index)], 
                         y=[80,80], name='oversold(80)',
                         line=dict(color='#f705c3', width=2, dash='dash'),
                         )],
                        
                 'layout': go.Layout(
                                    hovermode = 'x unified',
                                    uirevision = 'dataset', 
                                    margin = dict(t = 0  , r = 0, l = 60, b=0),

                    )
                        
               }

if __name__ == '__main__':
    app.run_server()
    

