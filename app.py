import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
from wordcloud import WordCloud, STOPWORDS


now = dt.date.today()
now = now.strftime('%m-%d-%Y')
yesterday = dt.date.today() - dt.timedelta(days = 1)
yesterday = yesterday.strftime('%m-%d-%Y')

nltk.download('punkt')
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

# Streamlit Dashboard          
st.set_page_config(page_title ="Woods and Pop Ltd", page_icon =":guardsman:", layout ="centered")
st.image("logo.jpeg", width = 400)
st.title('Woods and Pop Ltd')
st.header('🔎AI Stock Forecaster App')
st.subheader("NLP Analysis of the selected Stock based on Latest News Articles")
# save the company name in a variable
company_name = st.text_input("Please provide the name of the Company: ")

#As long as the company name is valid not empty...
if company_name != '':
    st.write(f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')
    #Extract News with Google News
    googlenews = GoogleNews(start=yesterday,end=now)
    googlenews.search(company_name)
    result = googlenews.result()
    #store the results
    df = pd.DataFrame(result)
    #st.write(df.tail())

    try:
        list =[] #creating an empty list 
        for i in df.index:
            dict = {} #creating an empty dictionary to append an article in every single iteration
            article = Article(df['link'][i],config=config) #providing the link
            try:
              article.download() #downloading the article 
              article.parse() #parsing the article
              article.nlp() #performing natural language processing (nlp)
            except:
               pass 
            #storing results in our empty dictionary
            dict['Date']=df['date'][i] 
            dict['Media']=df['media'][i]
            dict['Title']=article.title
            dict['Article']=article.text
            dict['Summary']=article.summary
            dict['Key_words']=article.keywords
            list.append(dict)
        check_empty = not any(list)
        # print(check_empty)
        if check_empty == False:
          news_df=pd.DataFrame(list) #creating dataframe
          st.write(news_df.tail())

    except Exception as e:
        #exception handling
        st.write("exception occurred:" + str(e))
        st.write('Looks like, there is some error in retrieving the data, Please try again or try with a different company.' )

        
    def percentage(part,whole):
        return 100 * float(part)/float(whole)

    #Assigning Initial Values
    positive = 0
    negative = 0
    neutral = 0
    #Creating empty lists
    news_list = []
    neutral_list = []
    negative_list = []
    positive_list = []

    #Iterating over the tweets in the dataframe
    for news in news_df['Summary']:
        news_list.append(news)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
        neg = analyzer['neg']
        neu = analyzer['neu']
        pos = analyzer['pos']
        comp = analyzer['compound']
        if neg > pos:
            negative_list.append(news) #appending the news that satisfies this condition
            negative += 1 #increasing the count by 1
        elif pos > neg:
            positive_list.append(news) #appending the news that satisfies this condition
            positive += 1 #increasing the count by 1
        elif pos == neg:
            neutral_list.append(news) #appending the news that satisfies this condition
            neutral += 1 #increasing the count by 1 

    positive = percentage(positive, len(news_df)) #percentage is the function defined above
    negative = percentage(negative, len(news_df))
    neutral = percentage(neutral, len(news_df))

    #Converting lists to pandas dataframe
    news_list = pd.DataFrame(news_list)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)
    #using len(length) function for counting
    #st.write("BUY SIDE:",'%.0f' % len(positive_list), end='\n')
    #st.write("NEUTRAL SIDE:", '%.2f' % len(neutral_list), end='\n')
    #st.write("SELL SIDE:", '%.2f' % len(negative_list), end='\n')

    #Creating PieCart
    labels = ['BUY ['+str(round(positive))+'%]' , 'THINK ['+str(round(neutral))+'%]','SELL ['+str(round(negative))+'%]']
    sizes = [positive, neutral, negative]
    colors = ['green', 'blue','red']
    plt.legend(labels)
    patches, texts = plt.pie(sizes,colors=colors, startangle=90)
    plt.axis('equal')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig1, ax1 = plt.subplots()
    #plt.style.use('default')
    plt.title("Sentiment Analysis Result for the stock "+company_name+"" )
    ax1.pie(sizes,labels=labels,colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

    # Word cloud visualization
    def word_cloud(text):
        stopwords = set(STOPWORDS)
        allWords = ' '.join([nws for nws in text])
        wordCloud = WordCloud(background_color='black',width = 1600, height = 800,stopwords = stopwords,min_font_size = 20,max_font_size=150,colormap='prism').generate(allWords)
        fig, ax = plt.subplots(figsize=(20,10), facecolor='k')
        plt.imshow(wordCloud, interpolation='bilinear')
        ax.axis("off")
        fig.tight_layout(pad=0)
        # Display the generated image:
        plt.show()
        st.pyplot()

    st.write('Wordcloud for ' + company_name)
    word_cloud(news_df['Summary'].values)
