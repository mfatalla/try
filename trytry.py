import requests
import lxml.html as lh
import pandas as pd
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime as dt


@st.cache(suppress_st_warning=True)
def load_data():
    components = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S" "%26P_500_companies"
    )[0]
    return components.drop("SEC filings", axis=1).set_index("Symbol")


@st.cache(suppress_st_warning=True)
def load_quotes(asset):
    return yf.download(asset)



menu = ['Overview', 'News', 'Technical Indicators', 'Company Profile', 'About']
query_params = st.experimental_get_query_params()

default = int(query_params["menubar"][0]) if "menubar" in query_params else 0
menubar = st.selectbox(
    "Menu",
    menu,
    index=default
)
if menubar:
    st.experimental_set_query_params(menubar=menu.index(menubar))

components = load_data()
title = st.empty()

st.sidebar.image('data//logo1.png')

@st.cache(suppress_st_warning=True)
def label(symbol):
    a = components.loc[symbol]
    return symbol + " - " + a.Security

st.sidebar.subheader("Select asset")
asset = st.sidebar.selectbox(
    "Click below to select a new asset",
    components.index.sort_values(),
    index=3,
    format_func=label,
)

ticker = yf.Ticker(asset)
info = ticker.info
url = 'https://stockanalysis.com/stocks/' + asset
response = requests.get(url)
soup = BeautifulSoup(response.text, 'lxml')

name = soup.find('h1', {'class': 'sa-h1'}).text
price = soup.find('span', {'id': 'cpr'}).text
currency = soup.find('span', {'id': 'cpr'}).find_next('span').text
change = soup.find('span', {'id': 'spd'}).text
rate = soup.find('span', {'id': 'spd'}).find_next('span').text
meta = soup.find('div', {'id': 'sti'}).find('span').text

after = soup.find('div', {'id': 'ext'}).find_next('span').text
after2 = soup.find('span', {'id': 'extc'}).text
aftert = soup.find('span', {'id': 'extcp'}).text
aftertime = soup.find('span', {'id': 'exttime'}).text
CR = change + " (" + rate + ")"
CT = after2 + " (" + aftert + ")"
sub = change
sub2 = after2
aye = ": After-hours"

formtab = st.sidebar.beta_container()
with formtab:
    st.image(info['logo_url'])
    qq = (info['shortName'])
    st.markdown(
        f"<p style='vertical-align:bottom;font-weight: bold; color: #FFFFFF;font-size: 40px;'>{qq}</p>",
        unsafe_allow_html=True)
    xx = price + " " + currency
    st.markdown(
        f"<p style='vertical-align:bottom;font-weight: bold; color: #FFFFFF;font-size: 20px;'>{xx}</p>",
        unsafe_allow_html=True)
    if float(sub) > 0:
        aye2 = "+"

        st.markdown(
            f"<p style='vertical-align:bottom;font-weight: bold; color: #00AC4A;font-size: 13px;'>{aye2 + CR}</p>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<p style='vertical-align:bottom;font-weight: bold; color: #D10000;font-size: 13px;'>{CR}</p>",
            unsafe_allow_html=True)

    st.markdown(
        f"<p style='vertical-align:bottom;font-weight: italic; color: #FFFFFF;font-size: 10px;'>{meta}</p>",
        unsafe_allow_html=True)

    if float(sub2) > 0:

        st.markdown(after + " " + currency)
        st.markdown(
            f"<p style='vertical-align:bottom;font-weight: bold; color: #00AC4A;font-size: 13px;'>{CT + aye}</p>",
            unsafe_allow_html=True)


    else:
        st.markdown(after + " " + currency)
        st.markdown(
            f"<p style='vertical-align:bottom;font-weight: bold; color: #D10000;;font-size: 13px;'>{CT + aye}</p>",
            unsafe_allow_html=True)

    st.markdown(
        f"<p style='vertical-align:bottom;font-weight: italic; color: #FFFFFF;font-size: 10px;'>{aftertime}</p>",
        unsafe_allow_html=True)



if menubar == 'Overview':
    left, right = st.beta_columns([1,1])
    with left:
        st.title("Line Chart")
        linechart = st.beta_container()
        with linechart:
            linechart_expander = st.beta_expander(label='Line Chart Settings')
            with linechart_expander:
                ticker = yf.Ticker(asset)
                info = ticker.info

                attri = ['SMA', 'SMA2']
                attributes = st.multiselect(
                    'Choose Chart Attributes [SMA, SMA2]',
                    attri,
                    default='SMA'
                )

                data0 = load_quotes(asset)
                data = data0.copy().dropna()
                data.index.name = None

                section = st.slider(
                    "Number of quotes",
                    min_value=30,
                    max_value=min([2000, data.shape[0]]),
                    value=500,
                    step=10,
                )
                data2 = data[-section:]["Adj Close"].to_frame("Adj Close")

                if "SMA" in attributes:
                    period = st.slider(
                        "SMA period", min_value=5, max_value=500, value=20, step=1
                    )
                    data[f"SMA {period}"] = data["Adj Close"].rolling(period).mean()
                    data2[f"SMA {period}"] = data[f"SMA {period}"].reindex(data2.index)

                if "SMA2" in attributes:
                    period2 = st.slider(
                        "SMA2 period", min_value=5, max_value=500, value=100, step=1
                    )
                    data[f"SMA2 {period2}"] = data["Adj Close"].rolling(period2).mean()
                    data2[f"SMA2 {period2}"] = data[f"SMA2 {period2}"].reindex(data2.index)
                linebutton = st.button('Linechart Set')

            st.subheader("Chart")
            st.line_chart(data2, height=400)
            if st.checkbox("View quotes"):
                st.subheader(f"{asset} historical data")
                st.write(data2)
        """
        st.title("Candlestick Chart")
        candlestick = st.beta_container()
        with candlestick:

                candlechart_expander = st.beta_expander(label='Line Chart Settings')
                with candlechart_expander:
                    intervalList = ["1m", "5m", "15m", "30m"]
                    query_paramsa5 = st.experimental_get_query_params()
                    default5 = int(query_paramsa5["sort"][0]) if "sort" in query_paramsa5 else 2
                    interval = st.selectbox(
                        'Interval in minutes',
                        intervalList,
                        index=default5
                    )

                    if interval:
                        st.experimental_set_query_params(sort=intervalList.index(interval))

                    dayList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
                    query_paramsa6 = st.experimental_get_query_params()
                    default6 = int(query_paramsa6["chartdays"][0]) if "chartdays" in query_paramsa6 else 9
                    chartdays = st.selectbox(
                        'No. of Days',
                        dayList,
                        index=default6
                    )
                    if chartdays:
                        st.experimental_set_query_params(chartdays=dayList.index(chartdays))

                    candlebutton = st.button('Candlestick Set')

                stock = yf.Ticker(asset)
                history_data = stock.history(interval=interval, period=str(chartdays) + "d")
                prices = history_data['Close']
                volumes = history_data['Volume']

                lower = prices.min()
                upper = prices.max()

                prices_ax = np.linspace(lower, upper, num=20)

                vol_ax = np.zeros(20)

                for i in range(0, len(volumes)):
                    if (prices[i] >= prices_ax[0] and prices[i] < prices_ax[1]):
                        vol_ax[0] += volumes[i]

                    elif (prices[i] >= prices_ax[1] and prices[i] < prices_ax[2]):
                        vol_ax[1] += volumes[i]

                    elif (prices[i] >= prices_ax[2] and prices[i] < prices_ax[3]):
                        vol_ax[2] += volumes[i]

                    elif (prices[i] >= prices_ax[3] and prices[i] < prices_ax[4]):
                        vol_ax[3] += volumes[i]

                    elif (prices[i] >= prices_ax[4] and prices[i] < prices_ax[5]):
                        vol_ax[4] += volumes[i]

                    elif (prices[i] >= prices_ax[5] and prices[i] < prices_ax[6]):
                        vol_ax[5] += volumes[i]

                    elif (prices[i] >= prices_ax[6] and prices[i] < prices_ax[7]):
                        vol_ax[6] += volumes[i]

                    elif (prices[i] >= prices_ax[7] and prices[i] < prices_ax[8]):
                        vol_ax[7] += volumes[i]

                    elif (prices[i] >= prices_ax[8] and prices[i] < prices_ax[9]):
                        vol_ax[8] += volumes[i]

                    elif (prices[i] >= prices_ax[9] and prices[i] < prices_ax[10]):
                        vol_ax[9] += volumes[i]

                    elif (prices[i] >= prices_ax[10] and prices[i] < prices_ax[11]):
                        vol_ax[10] += volumes[i]

                    elif (prices[i] >= prices_ax[11] and prices[i] < prices_ax[12]):
                        vol_ax[11] += volumes[i]

                    elif (prices[i] >= prices_ax[12] and prices[i] < prices_ax[13]):
                        vol_ax[12] += volumes[i]

                    elif (prices[i] >= prices_ax[13] and prices[i] < prices_ax[14]):
                        vol_ax[13] += volumes[i]

                    elif (prices[i] >= prices_ax[14] and prices[i] < prices_ax[15]):
                        vol_ax[14] += volumes[i]

                    elif (prices[i] >= prices_ax[15] and prices[i] < prices_ax[16]):
                        vol_ax[15] += volumes[i]

                    elif (prices[i] >= prices_ax[16] and prices[i] < prices_ax[17]):
                        vol_ax[16] += volumes[i]

                    elif (prices[i] >= prices_ax[17] and prices[i] < prices_ax[18]):
                        vol_ax[17] += volumes[i]

                    elif (prices[i] >= prices_ax[18] and prices[i] < prices_ax[19]):
                        vol_ax[18] += volumes[i]

                    else:
                        vol_ax[19] += volumes[i]

                fig = make_subplots(
                    rows=1, cols=2,
                    column_widths=[0.2, 0.8],
                    specs=[[{}, {}]],
                    horizontal_spacing=0.01

                )

                fig.add_trace(
                    go.Bar(
                        x=vol_ax,
                        y=prices_ax,
                        text=np.around(prices_ax, 2),
                        textposition='auto',
                        orientation='h'
                    ),

                    row=1, col=1
                )

                dateStr = history_data.index.strftime("%d-%m-%Y %H:%M:%S")

                fig.add_trace(
                    go.Candlestick(x=dateStr,
                                   open=history_data['Open'],
                                   high=history_data['High'],
                                   low=history_data['Low'],
                                   close=history_data['Close'],
                                   yaxis="y2"

                                   ),

                    row=1, col=2
                )

                fig.update_layout(
                    title_text='Market Profile Chart (US S&P 500)',  # title of plot
                    bargap=0.01,  # gap between bars of adjacent location coordinates,
                    showlegend=False,

                    xaxis=dict(
                        showticklabels=False
                    ),
                    yaxis=dict(
                        showticklabels=False
                    ),

                    yaxis2=dict(
                        title="Price (USD)",
                        side="right"

                    )

                )

                fig.update_yaxes(nticks=20)
                fig.update_yaxes(side="right")
                fig.update_layout(height=800)

                config = {
                    'modeBarButtonsToAdd': ['drawline']
                }

                st.plotly_chart(fig, use_container_width=True, config=config)
                """

    with right:
        st.image('data//logo1.png')
        """
        summarytable = st.beta_container()
        with summarytable:
            urlfortable = 'https://stockanalysis.com/stocks/'+asset
            page = requests.get(urlfortable)
            doc = lh.fromstring(page.content)
            tr_elements = doc.xpath('//tr')
            i = 0
            i2 = 0
            tablecount = 0
            mylist1 = []
            mylist2 = []
            mylist3 = []
            mylist4 = []
            for tablecount in range(9):
                for t in tr_elements[tablecount]:
                    i += 1
                    if (i % 2) == 0:
                        value1 = t.text_content()
                        mylist1.append(str(value1))
                    else:
                        name1 = t.text_content()
                        mylist2.append(str(name1))

            for tablecount2 in range(9, 18):
                for t2 in tr_elements[tablecount2]:
                    i2 += 1
                    if (i2 % 2) == 0:
                        value2 = t2.text_content()
                        mylist3.append(str(value2))
                    else:
                        name2 = t2.text_content()
                        mylist4.append(str(name2))

            final_table = pd.DataFrame(
                {"": list(mylist2), "Value": list(mylist1), " ": list(mylist4), "Value ": list(mylist3)})
            final_table.index = [""] * len(final_table)
            st.title("Summary")
            st.table(final_table) """


    urlq = 'https://stockanalysis.com/stocks/' + asset
    responseq = requests.get(urlq)
    soupq = BeautifulSoup(responseq.text, 'html.parser')

    samplenewscount = 0
    for samplenewscount in range(10):
        newsTitleq = soupq.find_all('div', {'class': 'news-side'})[samplenewscount].find('div').text
        newsThumbnailq = soupq.find_all('div', {'class': 'news-img'})[samplenewscount].find('img')
        newsBodyq = soupq.find_all('div', {'class': 'news-text'})[samplenewscount].find('p').text
        subMetaq = soupq.find_all('div', {'class': 'news-meta'})[samplenewscount].find_next('span').text
        hreflinkq = soupq.find_all('div', {'class': 'news-img'})[samplenewscount].find('a')
        linkq = hreflinkq.get('href')
        wapq = newsThumbnailq.get('data-src')


        chart1q, chart2q, chart3q = st.beta_columns([1, 2, 3])
        with chart1q:
            st.image(wapq)
        with chart2q:
            st.markdown(f"<h1 style='font-weight: bold; font-size: 17px;'>{newsTitleq}</h1>",
                        unsafe_allow_html=True)
            st.markdown(newsBodyq)
            linkq = "(" + linkq + ")"
            ayeq = '[[Link]]' + linkq
            st.markdown("Source: " + ayeq, unsafe_allow_html=True)
            st.text(" ")
            st.text(" ")

        with chart3q:
            st.markdown(subMetaq)

    st.text(" ")

elif menubar == 'News':
    st.image('data//logo1.png')

elif menubar == 'Technical Indicators':
    st.image('data//logo1.png')
elif menubar == 'Company Profile':
    st.image('data//logo1.png')


elif menubar == 'About':
    st.image('data//logo1.png')
else:
    st.error("Something has gone terribly wrong.")
