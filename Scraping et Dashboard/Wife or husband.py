# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:59:12 2022

@author: Feyn
"""

import gradio as gr
import pandas as pd
import time
import plotly


from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=91, timeout=(10,25),retries=2, backoff_factor=0.1, requests_args={'verify':False})
"""Keyword='tease'
 """
date=time.strftime('%Y-%m-%d')
Words=input(">> The Word?\n")

kw_list=[ Words + ' wife',  Words + ' husband']
l = []
for i in ['GLOBAL','US']:
    pytrends.build_payload(kw_list=kw_list,timeframe=f'2020-01-01 {date}',geo='US')
    pytrends.interest_by_region()
    
    df = pytrends.interest_over_time()
    interest_over_time_df = pytrends.interest_over_time().drop(columns='isPartial')
    dfWvsUS=df
    dfWvsUS['country']=i
    l.append(dfWvsUS.reset_index())
df1 = pd.concat(l)
df1.to_excel(r'putout.xlsx')

print("total records fetached:",df.size)
df.head()

pd.options.plotting.backend = "plotly"
fig1 = interest_over_time_df[kw_list].plot()
fig1.update_layout(
    title_text='Search volume over time',
    legend_title_text='Search terms'
)
fig1.show()


fig2 = plotly.express.bar(df, x='geoName', y=kw_list)
fig2.update_layout(
    title_text=f'Search volumes by country',
    legend_title_text='Search terms'
)
fig2.update_yaxes(title_text='Volume')
fig2.update_xaxes(title_text='Country')
fig2.show()
