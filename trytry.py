import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

snp500 = pd.read_csv("Datasets/SP500.csv")
symbols = snp500['Symbol'].sort_values().tolist()

ticker = st.sidebar.selectbox(
    'Choose a S&P 500 Stock',
    symbols)


numYearMA = 2
windowSizeMA = 15

def calcMovingAverage(numYearMA, windowSizeMA):
    df = numYearMA.copy()
    df['sma'] = df['Adj Close'].rolling(windowSizeMA).mean()
    df['ema'] = df['Adj Close'].ewm(span=windowSizeMA, min_periods=windowSizeMA).mean()
    df.dropna(inplace=True)
    return df

calcMovingAverage(numYearMA, windowSizeMA)

def calc_macd(numYearMA):
    df = numYearMA.copy()
    df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
    df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df.dropna(inplace=True)
    return df
calc_macd(numYearMA)

def calcBollinger(numYearMA, windowSizeMA):
    df = numYearMA.copy()
    df["sma"] = df['Adj Close'].rolling(windowSizeMA).mean()
    df["bolu"] = df["sma"] + 2 * df['Adj Close'].rolling(windowSizeMA).std(ddof=0)
    df["bold"] = df["sma"] - 2 * df['Adj Close'].rolling(windowSizeMA).std(ddof=0)
    df["width"] = df["bolu"] - df["bold"]
    df.dropna(inplace=True)
    return df
calcBollinger(numYearMA, windowSizeMA)


st.title('Technical Indicators')
st.subheader('Moving Average')



start = dt.datetime.today() - dt.timedelta(numYearMA * 365)
end = dt.datetime.today()
dataMA = yf.download(ticker, start, end)
df_ma = calcMovingAverage(dataMA, windowSizeMA)
df_ma = df_ma.reset_index()

figMA = go.Figure()

figMA.add_trace(
    go.Scatter(
        x=df_ma['Date'],
        y=df_ma['Adj Close'],
        name="Prices Over Last " + str(numYearMA) + " Year(s)"
    )
)

figMA.add_trace(
    go.Scatter(
        x=df_ma['Date'],
        y=df_ma['sma'],
        name="SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
    )
)

figMA.add_trace(
    go.Scatter(
        x=df_ma['Date'],
        y=df_ma['ema'],
        name="EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
    )
)

figMA.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

figMA.update_layout(legend_title_text='Trend')
figMA.update_yaxes(tickprefix="$")

st.plotly_chart(figMA, use_container_width=True)

st.subheader('Moving Average Convergence Divergence (MACD)')
numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2)

startMACD = dt.datetime.today() - dt.timedelta(numYearMACD * 365)
endMACD = dt.datetime.today()
dataMACD = yf.download(ticker, startMACD, endMACD)
df_macd = calc_macd(dataMACD)
df_macd = df_macd.reset_index()

figMACD = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.01)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['Adj Close'],
        name="Prices Over Last " + str(numYearMACD) + " Year(s)"
    ),
    row=1, col=1
)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['ema12'],
        name="EMA 12 Over Last " + str(numYearMACD) + " Year(s)"
    ),
    row=1, col=1
)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['ema26'],
        name="EMA 26 Over Last " + str(numYearMACD) + " Year(s)"
    ),
    row=1, col=1
)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['macd'],
        name="MACD Line"
    ),
    row=2, col=1
)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['signal'],
        name="Signal Line"
    ),
    row=2, col=1
)

figMACD.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1,
    xanchor="left",
    x=0
))

figMACD.update_yaxes(tickprefix="$")
st.plotly_chart(figMACD, use_container_width=True)

st.subheader('Bollinger Band')
coBoll1, coBoll2 = st.beta_columns(2)
with coBoll1:
    numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6)

with coBoll2:
    windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)

startBoll = dt.datetime.today() - dt.timedelta(numYearBoll * 365)
endBoll = dt.datetime.today()
dataBoll = yf.download(ticker, startBoll, endBoll)
df_boll = calcBollinger(dataBoll, windowSizeBoll)
df_boll = df_boll.reset_index()
figBoll = go.Figure()
figBoll.add_trace(
    go.Scatter(
        x=df_boll['Date'],
        y=df_boll['bolu'],
        name="Upper Band"
    )
)

figBoll.add_trace(
    go.Scatter(
        x=df_boll['Date'],
        y=df_boll['sma'],
        name="SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
    )
)

figBoll.add_trace(
    go.Scatter(
        x=df_boll['Date'],
        y=df_boll['bold'],
        name="Lower Band"
    )
)

figBoll.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1,
    xanchor="left",
    x=0
))

figBoll.update_yaxes(tickprefix="$")
st.plotly_chart(figBoll, use_container_width=True)
