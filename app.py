import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv(r'data\financial_news_events.csv')
df_clean = pd.read_csv(r'data\financial_news_events_clean.csv')

st.title('Financial News Dashboard', width='stretch')
st.divider()
st.header('Analyse the Effects of Financial News on Markets', width='stretch')
st.text('Financial news has a significant effect on markets globally')
st.text('In the following analysis we look at the effects of financial news headlines on markets')
st.text('The aim of the analysis is to uncover market trends which could serve traders who wish factor in the impact of news events on market performance and trading patterns')
with st.expander('See more details'):
    st.text('Key columns in the dataset include: \n-date\n-headline source\n-market event\n-market index\n-index change percent\n-trading volume\n-sector\n-sentiment\n-impact level\n-related company\n-news url')
    

st.sidebar.header('Sidebar Controls')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Overview', 'Distrubutions', 'Categorical Analysis', 'Market Movement analysis', 'Correlations', 'Predictive Modelling'])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Headlines: ', '2730')
    col2.metric('Average Index Change Percent:', '-0.03')
    col3.metric('Most Common Sentiment: ', 'Negative')
    col4.metric('Most impacted Sector: ', 'Agriculture')

    st.write('Original Dataset Shape: ', df.shape)

    st.write('Missing Values Before Data cleaning:')
    missing_per_column = df.isna().sum().reset_index()
    missing_per_column.columns = ['Column', 'Missing Values']
    st.dataframe(missing_per_column)

    st.write('Rows missing Headline or Index_Change_Percent were removed. Missing sentiment scores were filled using sentiment'
            'analysis on the headlines, and missing URL values were replaced with \'missing url data\'.')
    
    st.write('Final Dataset Shape: ', df_clean.shape)
    
    st.write('Summary Statistics Post Data Cleaning: \n')
    st.write(df_clean.describe())

    st.write('Clean Dataset Preview: ')
    st.dataframe(df_clean.head())