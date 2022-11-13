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
from functools import lru_cache



def create_dash(flask_app):

        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/maxminline/"), prevent_initial_callbacks=True)

        cc = CallbackCache(cache=FileSystemCache(cache_dir="cache"))

        #app.css.append_css({"external_url": "/static/css/style.css"})
        app.css.config.serve_locally = True

        GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)


        app.layout = html.Div([

            html.Link(rel='stylesheet', href='/static/css/style.css'),


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
                            dcc.Graph(id = 'price_candlesticker', animate=False,
                                style={'width': 'auto',
                                           'height': '100vh',
                                           'border': '1px #5c5c5c solid',
                                           'margin-top' : "40px"},
                                    config = {'displayModeBar': True, 'responsive': True},
                                    className = 'chart_width'),

                            ]),


                        dcc.Interval(id = 'update_value',
                                         interval = int(GRAPH_INTERVAL) ,
                                         n_intervals = 0)
                    ]),



        ])



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
        
        @cc.callback(
                Output('price_candlesticker', 'figure'),
                [Input('update_value', 'n_intervals'),
                 ])

                
        def display_candlestick(update_value):
            file = "D:\\Development\\flask dev\\stock\\data\\2022-10-03 NQ=F USTime_out_stock_data.csv"
            #file = "D:\\Development\\flask dev\\stock\\data\\data.csv"
            df = pd.read_csv(file)
            """
            import yfinance as yf
            df=yf.download(tickers="NQ=F", period="1d", interval="1m")
            df=df.reset_index()
            df.Datetime = df.Datetime.dt.strftime('%Y-%m-%d %H:%M:%S')
            df.to_csv('data.csv')
            #file = "D:\\Development\\data\\data.csv"
            
            #df.columns =['Datetime','Open','High','Low','Close','adj_close','Volume']
            df.to_dict(record)
            """
            #df = pd.DataFrame(file)
            
            fig = go.Figure(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            ))
            
            fig.update_layout(
                xaxis_rangeslider_visible=False,

            
            )
        
            return fig                
                 
                

            


        cc.register(app)
        return(app)

"""
if __name__ == '__main__':
    app.run_server()

"""




