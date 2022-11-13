import json
from datetime import datetime
import requests
import pathlib
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import *
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State

# To start an application in dash
app = dash.Dash(__name__)
server = app.server

dcc.Store(id='df_value'),
# This is the API call (requests.get)
@app.callback(Output('df_value', 'data'), Input('update_value', 'n_intervals'))

def update_df(n_intervals):
        if n_intervals == 0:
            raise PreventUpdate
        else:

            p_filename = "_out_stock_data.csv"

            #time_stamp = datetime.now() - datetime.timedelta(hours=13)
            #time_stamp =  datetime.now().strftime('%Y-%m-%d')
            time_stamp = usny_curtime()

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
            
            
df = pd.DataFrame(df_value)

newDate = df["Datetime"].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))

app.layout = html.Div(
    [
        dcc.Graph(
            id='btc_chart',
            style={"height": "99vh"}),
        dcc.Interval(
            id='btc_update',
            interval=1000,
            n_intervals=0
        )
    ]
)


@app.callback(
    Output('btc_chart', 'figure'),
    [Input('btc_update', 'n_intervals')],
    [State('btc_chart', 'figure')]
)
def updated_btc_2h(n_intervals, figure_state):
    candlesticks_btc = go.Candlestick(x=newDate,
                                      open=df["open"],
                                      high=df["high"],
                                      low=df["low"],
                                      close=df["close"],
                                      xaxis='x1',
                                      yaxis='y1',
                                      increasing=dict(line=dict(color="#00E676")),
                                      decreasing=dict(line=dict(color="#FF5252")),
                                      name="Candlesticks")
    layout_ = Layout(
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.980,
            traceorder='normal',
            bgcolor="#232e4d",
            font=dict(
                size=10,
                color="#ffffff"), ),
        paper_bgcolor='rgba(0 ,0, 0, 0)',
        plot_bgcolor='rgba(19, 23, 34, 1)',
        dragmode="pan",
        xaxis=dict(
            domain=[0, 1]
        ),
        yaxis=dict(
            domain=[0, 1]
        )
    )
    data = [candlesticks_btc]
    plot_btc = go.Figure(data=data, layout=layout_)
    plot_btc.update_layout(
        # title="BTC chart",
        font_family="Monospace",
        font_color="#000000",
        title_font_family="Monospace",
        title_font_color="#000000",
        legend_title_font_color="#000000",
        yaxis=dict(
            side="right"),  # Puts the y-axis on the right
        hovermode="closest",  # Used to be "x" but spiked didn't always show. So get date from block
        spikedistance=-1,
        # Keeps the spikes always on with -1. hoverdistance=0   # 0 means no looking for data (wont show it)
        hoverlabel=dict(  # Changes appearance settings of the hover label and x-axis hover label
            bgcolor="#232e4d",
            font_family="Monospace",
            font_size=11,
            font_color="#ffffff"
        )
    )

    # Zero lines
    plot_btc.update_xaxes(zeroline=False, zerolinewidth=2, zerolinecolor='LightPink')
    plot_btc.update_yaxes(zeroline=False, zerolinewidth=2, zerolinecolor='LightPink')

    # Overriding the axis and grids colours
    plot_btc.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#181F34')
    plot_btc.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#181F34')

    # Pointers
    plot_btc.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikesnap="cursor", spikedash='dot',
                         spikemode="across")
    plot_btc.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikesnap="cursor", spikedash='dot',
                         spikemode="across")

    # Colours of the candles
    cs = plot_btc.data[0]
    cs.increasing.fillcolor = '#131722'
    cs.increasing.line.color = '#00E676'
    cs.decreasing.fillcolor = '#FF5252'
    cs.decreasing.line.color = '#FF5252'

    # Removes rangeslider
    plot_btc.update(layout_xaxis_rangeslider_visible=False)

    # Sets the automatic x-axis range when you first open it (plots lasts 250 candles)
    # When using positive values in [], use timeReference
    # When using negative values in []., use timeReference.iloc[]
    plot_btc.update_xaxes(range=[newDate.iloc[-250], newDate.max()])

    # Sets the automatic y-axis range to the limit of
    # Lower limit: lowest value of the last 250 candles .iloc[-250:] - 100 (for room)
    # Higher limit: highest value of the last 250 candles .iloc[-250] + 100 (for room)
    # Has to be an iloc then min/max order

    plot_btc.update_yaxes(range=[(df["low"].iloc[-250:].min()) - 100, (df["high"].iloc[-250:].max()) + 100])

    # Frequency of readings on both axis
    plot_btc.update_xaxes(nticks=20)
    plot_btc.update_yaxes(nticks=30)
    config = dict({
        'scrollZoom': True,
        'displayModeBar': False,
        'showTips': False
    })

    return [candlesticks_btc, layout_, plot_btc, config]


# To run the app
if __name__ == '__main__':
    app.run_server(debug=True)
