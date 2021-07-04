import requests
import lxml.html as lh
import pandas as pd
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import technical
from plotly.subplots import make_subplots
import numpy as np
import datetime as dt
st.set_page_config(
    page_title = 'SLAPSOIL',
    page_icon = '💜',
    layout= 'wide',
    initial_sidebar_state="expanded",
)
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
    with right:
        st.image('data//logo1.png')
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
            st.table(final_table)
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
    if "page" not in st.session_state:
        st.session_state.page = 0
        st.session_state.count = 5
    def next_page():
        st.session_state.page += 1
        st.session_state.count += 5
    def prev_page():
        st.session_state.page -= 1
        st.session_state.count -= 5
    if "page2" not in st.session_state:
        st.session_state.page2 = 0
        st.session_state.count2 = 5
    if "count2" not in st.session_state:
        st.session_state.page2 = 0
        st.session_state.count2 = 5
    def next_page2():
        st.session_state.page2 += 1
        st.session_state.count2 += 5
    def prev_page2():
        st.session_state.page2 -= 1
        st.session_state.count2 -= 5
    Cnews = st.beta_expander("Company News", expanded=True)
    with Cnews:
        endp = st.session_state.count
        startp = endp - 5
        url = 'https://stockanalysis.com/stocks/'+asset
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        name = soup.find('h1', {'class': 'sa-h1'}).text
        x = 0
        for x in range(startp,endp):
            newsTitle = soup.find_all('div', {'class': 'news-side'})[x].find('div').text
            newsThumbnail = soup.find_all('div', {'class': 'news-img'})[x].find('img')
            newsBody = soup.find_all('div', {'class': 'news-text'})[x].find('p').text
            subMeta = soup.find_all('div', {'class': 'news-meta'})[x].find_next('span').text
            hreflink = soup.find_all('div', {'class': 'news-img'})[x].find('a')
            link = hreflink.get('href')
            wap = newsThumbnail.get('data-src')
            chart1, chart2, chart3 = st.beta_columns([1, 2, 1])
            with chart1:
                st.image(wap)
            with chart2:
                st.markdown(f"<h1 style='font-weight: bold; font-size: 17px;'>{newsTitle}</h1>",
                            unsafe_allow_html=True)
                st.markdown(newsBody)
                link = "(" + link + ")"
                aye = '[[Link]]' + link
                st.markdown("Source: " + aye, unsafe_allow_html=True)
                st.text(" ")
                st.text(" ")
            with chart3:
                st.markdown(subMeta)
        st.text(" ")
        st.write("")
        col1, col2, col3, _ = st.beta_columns([0.1, 0.17, 0.1, 0.63])
        if st.session_state.page < 4:
            col3.button(">", on_click=next_page)
        else:
            col3.write("")  # t
            # his makes the empty column show up on mobile
        if st.session_state.page > 0:
            col1.button("<", on_click=prev_page)
        else:
            col1.write("")  # this makes the empty column show up on mobile
        col2.write(f"Page {1 + st.session_state.page} of {5}")
    Onews = st.beta_expander("Stock Market News", expanded=False)
    with Onews:
        Oendp = st.session_state.count2
        Ostartp = Oendp - 5
        url = 'https://stockanalysis.com/news'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1', {'class': 'entry-title'}).text
        x = 0
        for x in range(Ostartp,Oendp):
            newsTitle1 = soup.find_all('div', {'class': 'news-side'})[x].find('div').text
            time1 = soup.find_all('div', {'class': 'news-meta'})[x].find('span').text
            newsThumbnail1 = soup.find_all('div', {'class': 'news-img'})[x].find('img')
            newsBody1 = soup.find_all('div', {'class': 'news-text'})[x].find('p').text
            hreflink1 = soup.find_all('div', {'class': 'news-img'})[x].find('a')
            link1 = hreflink1.get('href')
            newsimg1 = newsThumbnail1.get('data-src')
            chart1, chart2, chart3 = st.beta_columns([1, 2, 1])
            with chart1:
                st.image(newsimg1)
            with chart2:
                st.markdown(f"<h1 style='font-weight: bold; font-size: 17px;'>{newsTitle1}</h1>",
                            unsafe_allow_html=True)
                st.markdown(newsBody1)
                link1 = "(" + link1 + ")"
                concatclink = '[[Link]]' + link1
                st.markdown("Source: " + concatclink, unsafe_allow_html=True)
                st.text(" ")
                st.text(" ")
            with chart3:
                st.markdown(time1)
        st.text(" ")
        st.text(" ")
        col1, col2, col3, _ = st.beta_columns([0.1, 0.17, 0.1, 0.63])
        if st.session_state.page2 < 4:
            col3.button("> ", on_click=next_page2)
        else:
            col3.write("")  # t
            # his makes the empty column show up on mobile
        if st.session_state.page > 0:
            col1.button("< ", on_click=prev_page2)
        else:
            col1.write("")  # this makes the empty column show up on mobile
        col2.write(f"Page {1 + st.session_state.page2} of {5}")

elif menubar == 'Technical Indicators':
    st.image('data//logo1.png')

elif menubar == 'Company Profile':
    st.title('Company Profile')
    st.subheader(info['longName'])
    st.markdown('** Sector **: ' + info['sector'])
    st.markdown('** Industry **: ' + info['industry'])
    st.markdown('** Phone **: ' + info['phone'])
    st.markdown(
        '** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', ' + info['country'])
    st.markdown('** Website **: ' + info['website'])
    st.markdown('** Business Summary **')
    st.info(info['longBusinessSummary'])
    fundInfo = {
        'Enterprise Value (USD)': info['enterpriseValue'],
        'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
        'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
        'Net Income (USD)': info['netIncomeToCommon'],
        'Profit Margin Ratio': info['profitMargins'],
        'Forward PE Ratio': info['forwardPE'],
        'PEG Ratio': info['pegRatio'],
        'Price to Book Ratio': info['priceToBook'],
        'Forward EPS (USD)': info['forwardEps'],
        'Beta ': info['beta'],
        'Book Value (USD)': info['bookValue'],
        'Dividend Rate (%)': info['dividendRate'],
        'Dividend Yield (%)': info['dividendYield'],
        'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
        'Payout Ratio': info['payoutRatio']
    }
    fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
    fundDF = fundDF.rename(columns={0: 'Value'})
    st.subheader('Fundamental Info')
    st.table(fundDF)
    st.subheader('General Stock Info')
    st.markdown('** Market **: ' + info['market'])
    st.markdown('** Exchange **: ' + info['exchange'])
    st.markdown('** Quote Type **: ' + info['quoteType'])
    start = dt.datetime.today() - dt.timedelta(2 * 365)
    end = dt.datetime.today()
    df = yf.download(asset, start, end)
    df = df.reset_index()
    fig = go.Figure(
        data=go.Scatter(x=df['Date'], y=df['Adj Close'])
    )
    fig.update_layout(
        title={
            'text': "Stock Prices Over Past Two Years",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)
    marketInfo = {
        "Volume": info['volume'],
        "Average Volume": info['averageVolume'],
        "Market Cap": info["marketCap"],
        "Float Shares": info['floatShares'],
        "Regular Market Price (USD)": info['regularMarketPrice'],
        'Bid Size': info['bidSize'],
        'Ask Size': info['askSize'],
        "Share Short": info['sharesShort'],
        'Short Ratio': info['shortRatio'],
        'Share Outstanding': info['sharesOutstanding']
    }
    marketDF = pd.DataFrame(data=marketInfo, index=[0])
    st.table(marketDF)
elif menubar == 'About':
    st.image('data//logo1.png')
else:
    st.error("Something has gone terribly wrong.")