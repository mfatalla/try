import requests
import lxml.html as lh
import pandas as pd
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from collections import defaultdict
import csv


snp500 = pd.read_csv("data//updatated_ticker.csv")
symbols = snp500['Symbol'].sort_values().tolist()
snp500 = pd.read_csv("data//updatated_ticker.csv")
wap = snp500[['Symbol','Name']]

columns = defaultdict(list) # each value in each column is appended to a list

with open('data//updatated_ticker.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
symbols = columns['Symbol']
company = columns['Name']

radio_list = symbols
query_params = st.experimental_get_query_params()

# Query parameters are returned as a list to support multiselect.
# Get the first item in the list if the query parameter exists.
default = int(query_params["activity"][0]) if "activity" in query_params else 0
asset = st.selectbox(
    "Choose a Company",
    radio_list,
    index=default
)
if asset:
    st.experimental_set_query_params(asset=radio_list.index(asset))


ticker = yf.Ticker(asset)
info = ticker.info
url = 'https://stockanalysis.com/stocks/'+asset
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
