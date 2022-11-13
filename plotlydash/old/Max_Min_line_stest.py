from dash import dcc, html,dash_table
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



def create_dash(flask_app):

        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/maxminline/"), prevent_initial_callbacks=True)

        cc = CallbackCache(cache=FileSystemCache(cache_dir="cache"))

        #app.css.append_css({"external_url": "/static/css/style.css"})
        app.css.config.serve_locally = True

        GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)


        app.layout = html.Div([

            html.Link(rel='stylesheet', href='/static/css/style.css'),

            ##### Store data
                dcc.Store(id='df_value'),



            ##### Graph Display
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id = 'price_candlesticker', animate=False,)

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
                                           'height': '10vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),




                        dcc.Interval(id = 'update_value',
                                         interval = int(GRAPH_INTERVAL) ,
                                         n_intervals = 0)
                    ]),



        ])
        
        def update_df(n_intervals):
                if n_intervals == 0:
                    raise PreventUpdate
                else:

                    p_filename = "_out_stock_data.csv"

                    #time_stamp = datetime.now() - datetime.timedelta(hours=13)
                    time_stamp =  datetime.now().strftime('%Y-%m-%d')

                    filename = os.path.join(str(time_stamp[0:11]) +"NQ=F USTime" + p_filename)
                    filename = "2022-09-22 NQ=F USTime_out_stock_data.csv"
                    cwd = os.getcwd()
                    path = os.path.dirname(cwd)

                    #file_path = path + "/jjhui/stock_app/data/"
                    file_path = path + "\\stock\\data\\"
                    file = os.path.join(file_path, filename)

                    df = pd.read_csv(file)
                    df.columns =['Datetime','Open','High','Low','Close', 'Volume']
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
                Output('price_candlesticker', 'figure'),
                ],
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])

        def update_graph(n_intervals, data):

            if n_intervals == 0:
                raise PreventUpdate
            else:

                df = pd.DataFrame(data)
               
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                ))
                
                return fig

        cc.register(app)
        return(app)
