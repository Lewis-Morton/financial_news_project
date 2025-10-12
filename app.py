import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

#load preprocessed datasets and cleaned word lists from jupyter for use in the dashboard
df = pd.read_csv(r'data\financial_news_events.csv')
df_clean = pd.read_csv(r'data\financial_news_events_clean.csv')
heatmap_df = pd.read_csv(r'data\financial_news_events_heatmap_df.csv')
clean_words = pd.read_csv(r'data\clean_words.csv')['Word'].tolist()
word_series = pd.Series(clean_words)
df_one_hot_encoded = pd.read_csv(r'data\df_one_hot_encoded.csv')

#without setting the layout to wide, only 6 tabs are rendered
st.set_page_config(layout='wide')

st.title('Financial News Dashboard', width='stretch')
st.divider()
st.header('Analyse the Effects of Financial News on Markets', width='stretch')
st.text('Financial news has a significant effect on markets globally')
st.text('In the following analysis we look at the effects of financial news headlines on markets')
st.text('The aim of the analysis is to uncover market trends which could serve traders who wish factor in the impact of news events on market performance and trading patterns')
#expander keeps key information above the fold
with st.expander('See more details'):
    st.text('Columns in the dataset include: \n-date\n-headline source\n-market event\n-market index\n-index change percent\n-trading volume\n-sector\n-sentiment\n-impact level\n-related company\n-news url')
    

st.sidebar.header('Financial News Controls')

#date range filter
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(df_clean['Date']).min())
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(df_clean['Date']).max())

#sentiment filter
selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=df_clean['Sentiment'].unique(),
    default=df_clean['Sentiment'].unique()
)

#sector filter
selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=df_clean['Sector'].unique(),
    default=df_clean['Sector'].unique()
)

#impact level filter
selected_impact = st.sidebar.multiselect(
    'Impact Level',
    options=['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

#create new df to avoid changing original df_clean
df_filtered = df_clean.copy()

#date filter
df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
df_filtered = df_filtered[(df_filtered['Date'] >= pd.to_datetime(start_date)) &
                          (df_filtered['Date'] <= pd.to_datetime(end_date))]

#sentiment filter
df_filtered = df_filtered[df_filtered['Sentiment'].isin(selected_sentiments)]

#sector filter
df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]

#impact level filter
if selected_impact:
    df_filtered = df_filtered[df_filtered['Impact_Level'].isin(selected_impact)]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Overview', 'Distributions', 'Categorical Analysis', 'Market Movement analysis', 'Correlations', 'Headline Text Analysis', 'Predictive Modelling'])

with tab1:
    #key metrics at a glance
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

import seaborn as sns

with tab2:
    #contains distributions of key features within the dataset as histograms and bar charts
    st.write('These charts provide an overview of the distribution of key features in the dataset, including news sentiment, '
             'trading activity, market movements, and event impact. By visualising these distributions, we can better understand'
            'the patterns and tendencies present in the data.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Sentiment', data=df_filtered, palette='Set2', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('News Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    
    st.write('This chart shows how news article headlines are spread across positive, neutral, '
             'and negative sentiment categories.')
   

    fig, ax = plt.subplots(figsize=(12, 6))
    df_filtered['Trading_Volume'].plot(kind='hist', bins=10, ax=ax, color='grey', edgecolor='black', linewidth=2)
    ax.set_title('Trading Volume Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Trading Volume')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    
    st.write('The histogram illustrates how often different trading volume ranges occur,' 
             ' highlighting common trading activity levels.')


    fig, ax = plt.subplots(figsize=(12,6))
    df_filtered['Index_Change_Percent'].plot(kind='hist', bins=10, ax=ax, color='grey', edgecolor='black', linewidth=2)
    ax.set_title('Index Change % Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Index Change Percent')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    
    st.write('This histogram displays how frequently different percentage changes in the index occur, '
             'showing the spread of market movements.')


    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Impact_Level', data=df_filtered, palette='Set3', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Impact Level Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)                    
    
    st.write('The chart shows the frequency of events categorized by their market impact: low, medium, or high.')

with tab3:
    # seaborn count plots showing the distribution of sectors affected by news events, top 10 news souces, and top 10 news event types
    st.write('This tab provides insights into the distribution of news events across sectors, sources, and event types. '
             'By examining these distributions, we can identify which areas of the market and which information channels'
              'are most frequently associated with reported events.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Sector', data=df_filtered, palette='Set2', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Count of Sectors Affected by News Events')
    ax.set_ylabel('Count')
    ax.set_xlabel('Sectors')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write('This chart shows how news events are distributed across different market sectors, highlighting which sectors '
             'are most frequently impacted.')

    top_10_sources = df['Source'].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Source', data=df_filtered, palette='Set3', order=top_10_sources, edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Count of top 10 News Sources')
    ax.set_ylabel('Count')
    ax.set_xlabel('News Sources')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write('Here we examine the top 10 sources of news in the dataset, showing which outlets contribute most to market-related reporting.')

    top_10_market_events = df['Market_Event'].value_counts().head(10).index
    sns.countplot(x='Market_Event', data=df_filtered, palette='pastel', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Count of top 10 Event Types')
    ax.set_ylabel('Count')
    ax.set_xlabel('Market Event')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write('This chart displays the most common types of market events reported in the dataset, helping to understand which'
              'events dominate the news landscape.')

import seaborn as sns

with tab4:
    #seaborn boxplots exploring how index percent chane and trading volume are affected by various factors
    st.write('This tab explores how market outcomes, specifically index percent changes and trading volumes, vary according to different'
              'factors such as impact level, news sentiment, and sector. The boxplots allow us to observe the distributions,'
                'spread, and potential outliers across these categories, providing insight into how market reactions differ by context.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Impact_Level', y='Index_Change_Percent', data=df_filtered, ax=ax, palette='Set2', linewidth=2)
    ax.set_title('Index Change Percent by Impact Level')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write('This chart shows how the percentage change in the index varies across low, medium, and high impact events. '
             'It highlights whether events categorized as higher impact correspond to larger market movements.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sentiment', y='Index_Change_Percent', data=df_filtered, ax=ax, palette='Set3', linewidth=2)
    ax.set_title('Index Change Percent by Sentiment')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write('Here, we examine how positive, neutral, and negative news sentiment relates to changes in the index, '
             'helping to identify patterns between sentiment and market performance.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sector', y='Index_Change_Percent', data=df_filtered, ax=ax, palette='pastel', linewidth=2)
    ax.set_title('Index Change Percent by Sector')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Sector')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write('This chart displays index changes for events affecting different sectors, revealing which sectors experience '
             'greater volatility or larger shifts in response to news.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Impact_Level', y='Trading_Volume', data=df_filtered, ax=ax, palette='Set2', linewidth=2)
    ax.set_title('Trading Volume by Impact Level')
    ax.set_ylabel('Trading Volume')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write('This chart illustrates how trading volumes vary depending on the impact level of events, showing whether'
              'high-impact events tend to drive higher market activity.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sentiment', y='Trading_Volume', data=df_filtered, ax=ax, palette='Set3', linewidth=2)
    ax.set_title('Trading Volume by Sentiment')
    ax.set_ylabel('trading Volume')
    ax.set_xlabel('Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write('Here, trading volume is compared across sentiment categories, providing insight into whether market activity'
              'is influenced more by positive, neutral, or negative news.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sector', y='Trading_Volume', data=df_filtered, ax=ax, palette='pastel', linewidth=2)
    ax.set_title('Trading Volume by Sector')
    ax.set_ylabel('Trading Volume')
    ax.set_xlabel('Sector')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write('This chart shows trading volume distributions for events across sectors, helping to identify which sectors'
              'experience more active trading in response to news.')

with tab5:
    #seaborn heatmap which explores correlations between key features, encoding was done in jupyter and heatmap_df was exported
    st.write('This heatmap visualizes the correlations between trading sentiment, trading volume, index change percent, '
             'and impact level. ')
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(heatmap_df.corr(), annot=True, cmap='RdBu_r', center=0)
    ax.set_title("Correlation Heatmap of Key Financial Features")
    st.pyplot(fig)

    st.write('While some patterns can be observed, the relationships are generally weak. The strongest '
             'correlation is only 0.037, found between sentiment and trading volume, suggesting that sentiment in headlines'
              'does not strongly align with trading behavior in this dataset.')

with tab6:
        #most common word, bigrams, and trigrams are plotted using seaborn barplots, tokenisation, removal of punctuation and 
        #stopwords was carried out in jupyter notebooks
    #try:
        from nltk import bigrams, trigrams, FreqDist

        st.write('To better understand the language patterns within the headlines, we examined the frequency of individual'
                  'words as well as common word pairings (bigrams) and triplets (trigrams). This analysis highlights recurring' 
                  ' themes and keywords that may reveal the focus or tone of the dataset.')

        #get count of each unique word in word_series, select top 20 words, and 2 new columns for the df
        most_common_words = word_series.value_counts().head(20).reset_index()
        most_common_words.columns = ['Word', 'Count']
        
        
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(data=most_common_words.head(20), x='Word', y='Count', palette='Set2', edgecolor='black', linewidth=2, ax=ax)
        ax.set_title('Top 20 Most Common Headline Words')
        ax.set_ylabel('Count')
        ax.set_xlabel('Words')
        ax.grid(True, axis='y')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.write('This chart shows the 20 most frequently used words across all headlines. It gives a clear picture of which'
                'terms dominate the dataset, providing insight into the most emphasised topics or framing.')

        #create lists of bigrams and tridgrams from clean_words
        bigrams_list = list(bigrams(clean_words))
        trigrams_list = list(trigrams(clean_words))

        #create bigram and trigram frequency distributions
        fdist_bigrams = FreqDist(bigrams_list)
        fdist_trigrams = FreqDist(trigrams_list)

        #extract 10 most common bigrams and trigrams as lists of tuples
        top10_bigrams = fdist_bigrams.most_common(10)
        top10_trigrams = fdist_trigrams.most_common(10)

        #build dfs for bigrams and trigrams with 2 columns, one for the collocation and one for the count
        df_bigrams = pd.DataFrame(top10_bigrams, columns=['Bigram', 'Count'])
        df_trigrams = pd.DataFrame(top10_trigrams, columns=['Trigram', 'Count'])

        #use the join method to remove the separator from within the tuples of bigrams and trigrams
        df_bigrams['Bigram'] = [' '.join(x) for x in df_bigrams['Bigram']]
        df_trigrams['Trigram'] = [' '.join(x) for x in df_trigrams['Trigram']]

        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(data=df_bigrams.head(10), x='Bigram', y='Count', palette='Set2', edgecolor='black', linewidth=2, ax=ax)
        ax.set_title('Top 10 Most Common Bigrams')
        ax.set_ylabel('Count')
        ax.set_xlabel('Bigrams')
        ax.grid(True, axis='y')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.write('Here we look at the most common two-word combinations found in the headlines. These bigrams often highlight' 
           'relationships between concepts or recurring phrases, offering more context than single words alone.')

        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(data=df_trigrams.head(10), x='Trigram', y='Count', palette='Set3', edgecolor='black', linewidth=2, ax=ax)
        ax.set_title('Top 10 Most Common Trigrams')
        ax.set_ylabel('Count')
        ax.set_xlabel('Trigrams')
        ax.grid(True, axis='y')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.write('This chart extends the analysis to three-word sequences. Trigrams help capture headline structures and recurring' \
            ' expressions, giving a deeper view into how ideas are framed or repeated.')

        #st.subheader('Top 10 Bigrams')
        #st.dataframe(df_bigrams)

        #st.subheader('Top 10 Trigrams')
        #st.dataframe(df_trigrams)
    #except Exception as e:
        #st.error(f'Error in Tab 6: {e}')

import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score

with tab7:
    #define classification model features and target
    features = ['Index_Change_Percent', 'Trading_Volume', 'Sentiment_Numeric', 
            'Sector_Aerospace & Defense', 'Sector_Agriculture', 'Sector_Automotive', 
            'Sector_Construction', 'Sector_Consumer Goods', 'Sector_Energy', 
            'Sector_Finance', 'Sector_Healthcare', 'Sector_Industrials', 
            'Sector_Materials', 'Sector_Media & Entertainment', 
            'Sector_Pharmaceuticals', 'Sector_Real Estate', 'Sector_Retail', 
            'Sector_Technology', 'Sector_Telecommunications', 'Sector_Transportation', 
            'Sector_Utilities', 'Market_Event_Bond Market Fluctuation', 
            'Market_Event_Central Bank Meeting', 'Market_Event_Commodity Price Shock', 
            'Market_Event_Consumer Confidence Report', 
            'Market_Event_Corporate Earnings Report', 'Market_Event_Cryptocurrency Regulation', 
            'Market_Event_Currency Devaluation', 'Market_Event_Economic Data Release', 
            'Market_Event_Geopolitical Event', 'Market_Event_Government Policy Announcement', 
            'Market_Event_IPO Launch', 'Market_Event_Inflation Data Release', 
            'Market_Event_Interest Rate Change', 'Market_Event_Major Merger/Acquisition', 
            'Market_Event_Market Rally', 'Market_Event_Regulatory Changes', 
            'Market_Event_Stock Market Crash', 'Market_Event_Supply Chain Disruption', 
            'Market_Event_Trade Tariffs Announcement', 
            'Market_Event_Unemployment Rate Announcement']
    x = df_one_hot_encoded[features]
    y = df_one_hot_encoded.Impact_Level_Numeric

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    dtree = DecisionTreeClassifier(max_depth=4)
    dtree = dtree.fit(x_train, y_train)

    st.write('  Introduction to Decison Tree')
    st.write('We trained a decision tree classifier to predict the impact level of market events'
      '(Low, Medium, High) based on numeric features like Index_Change_Percent, '
      'Trading_Volume, sentiment scores, sector indicators, and market event flags.\n')
    st.write('  Explanation of Features')
    st.write('The model uses a combination of numeric features and one-hot encoded categorical features:\n'
      '-Market indicators (Index_Change_Percent, Trading_Volume)\n'
      '-Sentiment scores (Sentiment_Numeric)\n'
      '-Sector dummies (Sector_Technology, Sector_Healthcare, etc.)\n'
      '-Market event flags (Market_Event_Bond Market Fluctuation, Market_Event_IPO Launch, etc.)\n')
    st.write('  How the Tree Works')
    st.write('Each node in the tree represents a decision based on a feature threshold.The tree splits'
      'the data at points that maximise class separation (using Gini impurity or entropy).Leaf'
      'nodes show the predicted class (Low, Medium, High) and can be interpreted as the model’s'
      'predicted impact level for the observations that reach that node.\n')
    st.write('  Interpreting the Tree')
    st.write('The top node represents the most important feature for predicting impact.Branches split'
      'based on feature thresholds, and the path from root to leaf indicates the sequence of'
      'decisions. Node colours indicate class distribution — darker colours represent a stronger class signal.\n \n')

    fig, ax = plt.subplots(figsize=(30,25))
    class_mapping = {0: "Low", 1: 'Medium', 2: 'High'}
    class_labels = [class_mapping[c] for c in dtree.classes_]
    tree.plot_tree(dtree, feature_names=features, class_names=class_labels, filled=True, fontsize=15, rounded=True)
    st.pyplot(fig)
    print('Decision Tree Classes: ',dtree.classes_,'\n')


    y_test_predict = dtree.predict(x_test)

    #st.write('Confusion matrix tree: \n', confusion_matrix(y_test, y_test_predict), '\n')
    st.write('Accuracy: ', accuracy_score(y_test ,y_test_predict), '\n')
    #st.write(metrics.classification_report(y_test, y_test_predict),'\n')

    # Feature importance chart
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": dtree.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=importance_df.head(10), x='Feature', y='Importance', palette='Blues_r', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title("Top 10 Feature Importances")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=90, ha='right')
    st.pyplot(fig)



    


