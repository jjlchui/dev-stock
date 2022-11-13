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

def usny_curtime():
    nyc_datetime = datetime.now(pytz.timezone('US/Eastern'))
    fmt = '%Y-%m-%d %H:%M:%S'
    time_stamp = nyc_datetime.strftime(fmt)
    return time_stamp






def create_dash(flask_app):

        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/maxminline/"), prevent_initial_callbacks=True)

        cc = CallbackCache(cache=FileSystemCache(cache_dir="cache"))

        #app.css.append_css({"external_url": "/static/css/style.css"})
        app.css.config.serve_locally = True

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
                            dcc.Graph(id = 'price_candlesticker', animate=True,
                            animation_options = dict(
                                transition = dict(duration = 500),
                                frame = dict(duration = 500, redraw = True),
                            ),
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

        @cc.callback(Output('df_value', 'data'), Input('update_value', 'n_intervals'))

        def update_df(n_intervals):
                if n_intervals == 0:
                    raise PreventUpdate
                else:

                    p_filename = "_out_stock_data.csv"

                    #time_stamp = datetime.now() - datetime.timedelta(hours=13)
                    #time_stamp =  datetime.now().strftime('%Y-%m-%d')
                    time_stamp = usny_curtime()

                    #filename = os.path.join(str(time_stamp[0:11]) +"NQ=F USTime" + p_filename)
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

        @cc.callback(
                Output('price_candlesticker', 'extendData'),
                [Input('update_value', 'n_intervals')
                 ,Input('df_value', 'data')],
                [State('price_candlesticker', 'figure'),
                    ])

        def update_graph(n_intervals, data, existing):

            if n_intervals == 0:
                raise PreventUpdate
            else:
                df = pd.DataFrame(data)
                #gather data into dict from dataframe
                new_data = df[['Datetime', 'High', 'Low', 'Open', 'Close']].iloc[-1]
                #new_data['time'] = df['time'] + delta*n_intervals
                new_data = new_data.to_dict()
            
                #data rename
                new_data['x'] = new_data['Datetime']
                del new_data['Datetime']
            
                #data reshape
                for k in new_data.keys():
                    new_data[k] = [[new_data[k]]]
            
                return new_data, [1]



        @cc.callback(
                Output('price_candlesticker', "figure"),
                [Input('update_value', 'n_intervals'),
                Input('df_value', 'data')],
                [State('price_candlesticker', 'figure'),
                    ])

        def update_graph(n_intervals, data, existing):

            if n_intervals == 0:
                raise PreventUpdate
            else:

                df = pd.DataFrame(data)
                
                #x_new = existing['data'][0]['x'][-1] + 1
                #y_new = existing['data'][0]['y'][-1] + 1
                """
                new_x_values = 10
                new_y_values = 50
                x_new = existing['data'][0]['x'].extend(new_x_values)
                y_new = existing['data'][0]['y'].extend(new_y_values)
                x=x_new, y=y_new
                """

                #return [dict(x=[x_new], y=[y_new])], [0], 100

                figure = go.Figure(
                    data = [go.Scatter(x=df.index, y=df.Close, line=dict(color='#fc0080', width=2),
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
                                increasing={'line_width': 0.1, 'line_color': 'black', 'fillcolor': 'red'},
                                decreasing={'line_width': 0.1, 'line_color': 'black', 'fillcolor': 'green'}),





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
                 )
                return figure



            


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



        cc.register(app)
        return(app)

"""
if __name__ == '__main__':
    app.run_server()

"""

