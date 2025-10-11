import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv(r'data\financial_news_events.csv')
df_clean = pd.read_csv(r'data\financial_news_events_clean.csv')
heatmap_df = pd.read_csv(r'data\financial_news_events_heatmap_df.csv')
clean_words = pd.read_csv(r'data\clean_words.csv')['Word'].tolist()
word_series = pd.Series(clean_words)
df_one_hot_encoded = pd.read_csv(r'data\df_one_hot_encoded.csv')

st.set_page_config(layout='wide')

st.title('Financial News Dashboard', width='stretch')
st.divider()
st.header('Analyse the Effects of Financial News on Markets', width='stretch')
st.text('Financial news has a significant effect on markets globally')
st.text('In the following analysis we look at the effects of financial news headlines on markets')
st.text('The aim of the analysis is to uncover market trends which could serve traders who wish factor in the impact of news events on market performance and trading patterns')
with st.expander('See more details'):
    st.text('Key columns in the dataset include: \n-date\n-headline source\n-market event\n-market index\n-index change percent\n-trading volume\n-sector\n-sentiment\n-impact level\n-related company\n-news url')
    

st.sidebar.header('Sidebar Controls')

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Overview', 'Distributions', 'Categorical Analysis', 'Market Movement analysis', 'Correlations', 'Headline Text Analysis', 'Predictive Modelling'])

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

with tab2:
    fig, ax = plt.subplots(figsize=(12, 6))
    sentiments = df_clean.groupby('Sentiment').size()
    sentiments.plot(kind='bar', ax=ax, color='maroon', edgecolor='black', linewidth=2)
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('News Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    st.write('News Sentiment')
    st.write('This chart shows how news articles are spread across positive, neutral, '
             'and negative sentiment categories.')
   

    fig, ax = plt.subplots(figsize=(12, 6))
    df_clean['Trading_Volume'].plot(kind='hist', bins=10, ax=ax, color='yellow', edgecolor='black', linewidth=2)
    ax.set_title('Trading Volume Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Trading Volume')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    st.write('Trading Volume')
    st.write('The histogram illustrates how often different trading volume ranges occur,' 
             'highlighting common trading activity levels.')
    

    fig, ax = plt.subplots(figsize=(12,6))
    df_clean['Index_Change_Percent'].plot(kind='hist', bins=10, ax=ax, color='cyan', edgecolor='black', linewidth=2)
    ax.set_title('Index Change % Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Index Change Percent')
    ax.grid(True, axis='y')
    st.pyplot(fig)
    st.write('Index Change Percent')
    st.write('This histogram displays how frequently different percentage changes in the index occur, '
             'showing the spread of market movements.')


    fig, ax = plt.subplots(figsize=(12,6))
    impact_level = df_clean.groupby('Impact_Level').size()
    impact_level.plot(kind='bar', ax=ax, color='orange', edgecolor='black', linewidth=2)
    ax.set_title('Impact Level Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)                    
    st.write('Impact Level')
    st.write('The chart shows the frequency of events categorized by their market impact: low, medium, or high.')

with tab3:
    fig, ax = plt.subplots(figsize=(12,6))
    sector = df_clean.groupby('Sector').size()
    sector.plot(kind='bar', ax=ax, color='salmon', edgecolor='black', linewidth=2)
    ax.set_title('Count of Sectors Affected bby News Events')
    ax.set_ylabel('Count')
    ax.set_xlabel('Sectors')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12,6))
    source = df['Source'].value_counts().head(10)
    source.plot(kind='bar', ax=ax, color='grey', edgecolor='black', linewidth=2)
    ax.set_title('Count of top 10 News Sources')
    ax.set_ylabel('Count')
    ax.set_xlabel('News Sources')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12,6))
    market_event = df['Market_Event'].value_counts().head(10)
    market_event.plot(kind='bar', ax=ax, color='green', edgecolor='black', linewidth=2)
    ax.set_title('Count of top 10 Event Types')
    ax.set_ylabel('Count')
    ax.set_xlabel('Market Event')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

import seaborn as sns

with tab4:
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Impact_Level', y='Index_Change_Percent', data=df_clean, ax=ax, palette='Set1', linewidth=2)
    ax.set_title('Index Change Percent by Impact Level')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sentiment', y='Index_Change_Percent', data=df_clean, ax=ax, palette='Set2', linewidth=2)
    ax.set_title('Index Change Percent by Sentiment')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sector', y='Index_Change_Percent', data=df_clean, ax=ax, palette='Set3', linewidth=2)
    ax.set_title('Index Change Percent by Sector')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Sector')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Impact_Level', y='Trading_Volume', data=df_clean, ax=ax, palette='Set1', linewidth=2)
    ax.set_title('Trading Volume by Impact Level')
    ax.set_ylabel('Trading Volume')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sentiment', y='Trading_Volume', data=df_clean, ax=ax, palette='Set2', linewidth=2)
    ax.set_title('Trading Volume by Sentiment')
    ax.set_ylabel('trading Volume')
    ax.set_xlabel('Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sector', y='Trading_Volume', data=df_clean, ax=ax, palette='Set3', linewidth=2)
    ax.set_title('Trading Volume by Sector')
    ax.set_ylabel('Trading Volume')
    ax.set_xlabel('Sector')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

with tab5:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(heatmap_df.corr(), annot=True, cmap='RdBu_r', center=0)
    ax.set_title("Correlation Heatmap of Key Financial Features")
    st.pyplot(fig)

with tab6:
    try:
        from nltk import bigrams, trigrams, FreqDist
    
        fig, ax = plt.subplots(figsize=(12,6))
        most_common_words = word_series.value_counts().head(20)
        most_common_words.plot(kind='bar', ax=ax, color='cyan', edgecolor='black', linewidth=2)
        ax.set_title('Top 20 Most Common Headline Words')
        ax.set_ylabel('Count')
        ax.set_xlabel('Words')
        ax.grid(True, axis='y')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        bigrams_list = list(bigrams(clean_words))
        trigrams_list = list(trigrams(clean_words))

        fdist_bigrams = FreqDist(bigrams_list)
        fdist_trigrams = FreqDist(trigrams_list)

        top10_bigrams = fdist_bigrams.most_common(10)
        top10_trigrams = fdist_trigrams.most_common(10)

        df_bigrams = pd.DataFrame(top10_bigrams, columns=['Bigram', 'Count'])
        df_trigrams = pd.DataFrame(top10_trigrams, columns=['Trigram', 'Count'])

        df_bigrams['Bigram'] = [' '.join(x) for x in df_bigrams['Bigram']]
        df_trigrams['Trigram'] = [' '.join(x) for x in df_trigrams['Trigram']]

        st.subheader('Top 10 Bigrams')
        st.dataframe(df_bigrams)

        st.subheader('Top 10 Trigrams')
        st.dataframe(df_trigrams)
    except Exception as e:
        st.error(f'Error in Tab 6: {e}')

import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score

with tab7:
    #Define classification model features and target
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
    fig, ax = plt.subplots(figsize=(30,25))
    class_mapping = {0: "Low", 1: 'Medium', 2: 'High'}
    class_labels = [class_mapping[c] for c in dtree.classes_]
    tree.plot_tree(dtree, feature_names=features, class_names=class_labels, filled=True, fontsize=15, rounded=True)
    #plt.show()
    st.pyplot(fig)
    print('Decision Tree Classes: ',dtree.classes_,'\n')


    y_test_predict = dtree.predict(x_test)

    st.write('Confusion matrix tree: \n', confusion_matrix(y_test, y_test_predict), '\n')
    st.write('Accuracy: ', accuracy_score(y_test ,y_test_predict), '\n')
    st.write(metrics.classification_report(y_test, y_test_predict),'\n')

    # Feature importance chart
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": dtree.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df.head(10).plot(kind="bar", x="Feature", y="Importance", ax=ax, legend=False, color='skyblue', edgecolor='black')
    ax.set_title("Top 10 Feature Importances")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=90, ha='right')
    st.pyplot(fig)


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
      'the data at points that maximize class separation (using Gini impurity or entropy).Leaf'
      'nodes show the predicted class (Low, Medium, High) and can be interpreted as the model’s'
      'predicted impact level for the observations that reach that node.\n')
    st.write('  Interpreting the Tree')
    st.write('The top node represents the most important feature for predicting impact.Branches split'
      'based on feature thresholds, and the path from root to leaf indicates the sequence of'
      'decisions. Node colors indicate class distribution — darker colors represent a stronger class signal.\n \n')


