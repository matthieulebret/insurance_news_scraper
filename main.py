import requests
from bs4 import BeautifulSoup
import pandas as pd
from newspaper import Article
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.set_page_config(layout='wide')

#
@st.cache_resource
def nltkvader():
    return nltk.download('vader_lexicon')

@st.cache_resource
def nltkpunkt():
    return nltk.download('punkt')

nltkvader()
nltkpunkt()

# Search Query


st.title("Business news scraper")

with st.form('Select search criteria and horizon'):
    query = st.text_input('Please input news query', value='Canopius')
    timeline = st.selectbox('Select time horizon', ['1d', '7d', '1m'])
    st.form_submit_button('Submit')


# Encode special characters in a text string
def encode_special_characters(text):
    encoded_text = ''
    special_characters = {'&': '%26', '=': '%3D', '+': '%2B', ' ': '%20'}  # Add more special characters as needed
    for char in text.lower():
        encoded_text += special_characters.get(char, char)
    return encoded_text


query2 = encode_special_characters(query)
url = f"https://news.google.com/search?q={query2}+when:" + timeline + "&hl=en-UK&gl=UK&ceid=UK%3Aen"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

articles = soup.find_all('article')
links = [article.find('a')['href'] for article in articles]
links = [link.replace("./articles/", "https://news.google.com/articles/") for link in links]

news_text = [article.get_text(separator='\n') for article in articles]
news_text_split = [text.split('\n') for text in news_text]

if len(articles) == 0:
    st.write("No news found over the selected time period")
else:

    news_df = pd.DataFrame({
        'Title': [text[2] for text in news_text_split],
        'Source': [text[0] for text in news_text_split],
        'Time': [text[3] if len(text) > 3 else 'Missing' for text in news_text_split],
        'Author': [text[4].split('By ')[-1] if len(text) > 4 else 'Missing' for text in news_text_split],
        'Link': links
    })

    links = news_df['Link'].to_list()
    keywords = []
    summaries = []
    texts = []
    for link in links:
        try:
            article = Article(link)
            article.download()
            article.parse()
            article.nlp()
            keywords.append(article.keywords)
            summaries.append(article.summary)
            texts.append(article.text)
        except:
            keywords.append(['N/A'])
            summaries.append(['N/A'])
            texts.append(['N/A'])
    news_df['Keywords'] = keywords
    news_df['Summary'] = summaries
    news_df['Full Text'] = texts

    fulltext = ' '.join(map(news_df['Full Text'].tolist()))

    sentiments = []
    for text in texts:
        try:
            # sentiments.append(SentimentIntensityAnalyzer().polarity_scores(text)['compound'])
            art = TextBlob(text)
            sentiments.append(art.sentiment)
        except:
            sentiments.append(['N/A'])

    news_df['Polarity'] = [sentiment[0] for sentiment in sentiments]
    news_df['Subjectivity'] = [sentiment[1] for sentiment in sentiments]
    news_df = news_df[['Time', 'Title', 'Summary', 'Polarity', 'Subjectivity','Link', 'Source', 'Author', 'Keywords']]
    news_df['Link'] = news_df['Link'].apply(lambda x: str(x))

    st.data_editor(news_df, column_config={'Link': st.column_config.LinkColumn(display_text='Full article')})

    fig = px.scatter(news_df, x='Polarity',y='Subjectivity',range_x=[-1,1],range_y=[0,1],hover_data='Title',width=600,height=600)
    fig.update_traces(marker=dict(size=20))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Sentiment analysis')
        st.plotly_chart(fig)

    with col2:
        st.subheader('Wordcloud')
        wordcloud = WordCloud().generate(fulltext)
        # Display the word cloud
        fig,ax= plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
