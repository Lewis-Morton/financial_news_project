import pandas as pd
import matplotlib.pyplot as pd
import streamlit as st


st.title('Financial News Dashboard', width='stretch')
st.divider()
st.header('Analyse the Effects of Financial News on Markets', width='stretch')
st.text('Financial news has a significant effect on markets globally')
st.text('In the following analysis we look at the effects of financial news headlines on markets')
st.text('The aim of the analysis is to uncover market trends which could serve traders who wish factor in the impact of news events on market performance and trading patterns')
with st.expander('See more details'):
    st.text('Key columns in the dataset include: \ndate\nheadline source\nmarket event\nmarket index\nindex change percent\ntrading volume\nsector\nsentiment\nimpact level\nrelated company\nnews url')
    

st.sidebar.header('Sidebar Controls')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Overview', 'Distrubutions', 'Categorical Analysis', 'Market Movement analysis', 'Correlations', 'Predictive Modelling'])