#!/usr/bin/env python
# coding: utf-8

# In[367]:


import pandas as pd
import spacy
import numpy as np


# In[368]:


from spacy.lang.en.stop_words import STOP_WORDS as stop_words


# In[369]:


df = pd.read_csv("twitter4000.csv",encoding='latin1')


# In[370]:


df


# In[371]:


df['sentiment'].value_counts()


# In[372]:


## Word Count
df['word_count']=df['twitts'].apply(lambda x:len(str(x).split()))


# In[373]:


df.head()


# In[374]:


df['word_count'].max()


# In[375]:


df['word_count'].min()


# In[376]:


df[df['word_count']==1]


# In[377]:


## Character Count 
df['char_count'] = df['twitts'].apply(lambda x: len(str(x).replace(" ","")))


# In[378]:


df.sample(5)


# In[379]:


## Average Word Length


# In[380]:


x= "this is" # 2 Words | 6 Characters | Average Word Length = 6/2 =3
y="thankyou guys" # Average Word Length = 12/2 =6


# In[381]:


df['avg_word_length']=df['char_count']/df['word_count']


# In[382]:


df.sample(5)


# In[383]:


## Stop Words


# In[384]:


print(stop_words)


# In[385]:


x = "i am a boy"
x.split()


# In[386]:


def stop_words_count(x):
    tmp = 0
    tmp_sentence = x.split()
    for i in tmp_sentence:
        if i in stop_words:
            tmp += 1
    return tmp


# In[387]:


df['stop_words_count']=df['twitts'].apply(stop_words_count)


# In[388]:


df.sample(5)


# In[389]:


# Count #HashTags and @Mentions


# In[390]:


x = "this is #hashtag and this is @mention"


# In[391]:


x.split()


# In[392]:


[i for i in x.split() if i.startswith('@')]


# In[393]:


[i for i in x.split() if i.startswith('#')]


# In[394]:


df['hashtags_count']=df['twitts'].apply(lambda x : len([i for i in x.split() if i.startswith('#')]))


# In[395]:


df['mentions_count']=df['twitts'].apply(lambda x : len([i for i in x.split() if i.startswith('@')]))


# In[396]:


df.sample(5)


# In[397]:


# If numeric digits are present in twiits


# In[398]:


x = 'this is 1 and 2'


# In[399]:


x.split()


# In[400]:


def numeric_count(x):
    tmp = 0
    for i in x.split():
        if i.isdigit():
            tmp+=1
    return tmp


# In[401]:


df['numerical_count']=df['twitts'].apply(numeric_count)


# In[402]:


df.sample(5)


# In[403]:


# Upper case words count


# In[404]:


x = "I AM HAPPY"
y = "i am happy"


# In[405]:


def calculate_upper(x):
    tmp = 0
    for i in x.split():
        if i.isupper():
            tmp +=1
    return tmp


# In[406]:


df['upper_case_count']=df['twitts'].apply(calculate_upper)


# In[407]:


df.sample(5)


# In[408]:


## Lower case conversion


# In[409]:


x = 'This is Text'


# In[410]:


x.lower()


# In[411]:


df['twitts']=df['twitts'].apply(lambda x: str(x).lower())


# In[412]:


df.sample(5)


# In[413]:


## Contaction to extraction


# In[414]:


contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}


# In[415]:


x = "i'm don't he'll" # i am do not he will


# In[416]:


def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key,value)
        return x
    else:
        return x


# In[417]:


cont_to_exp(x)


# In[418]:


df['twitts']=df['twitts'].apply(cont_to_exp)


# In[419]:


df.sample(5)


# In[420]:


df[df['twitts'].str.contains('hotmail\.com')]


# In[421]:


df.iloc[3713]['twitts']


# In[422]:


x = "@secureness arghh me please ts741127_56@gmail.com"


# In[423]:


import re


# In[424]:


re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',x)


# In[425]:


df['emails']=df['twitts'].apply(lambda x : re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',x))


# In[426]:


df['email_count']=df['emails'].apply(lambda x : len(x))


# In[427]:


df[df['email_count']>0]


# In[428]:


re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"",x)


# In[429]:


df['twitts']=df['twitts'].apply(lambda x : re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"",x))


# In[430]:


df[df['email_count']>0]


# In[431]:


# Count URLS and remove it


# In[432]:


x = "hi, thanks to order. for more visit https://localbazer.com/?q=1"


# In[433]:


re.findall(r'(ftp|http|https):\/\/(\w+:{0,1}\w*@)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%@!\-\/]))?',x)


# In[434]:


df['url_flags']=df['twitts'].apply(lambda x : len(re.findall(r'(ftp|http|https):\/\/(\w+:{0,1}\w*@)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%@!\-\/]))?',x)))


# In[435]:


df[df['url_flags']>0]['twitts']


# In[436]:


x


# In[437]:


re.sub(r'(ftp|http|https):\/\/(\w+:{0,1}\w*@)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%@!\-\/]))?',"",x)


# In[438]:


df['twitts']=df['twitts'].apply(lambda x : re.sub(r'(ftp|http|https):\/\/(\w+:{0,1}\w*@)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%@!\-\/]))?',"",x))


# In[439]:


# Remove RT (Retweet)


# In[440]:


x = 'rt@username: hello hirt'


# In[441]:


re.sub(r'\brt\b','',x).strip()


# In[442]:


df['twitts']=df['twitts'].apply(lambda x : re.sub(r'\brt\b','',x).strip())


# In[443]:


# Special chars removal or punctuation removal


# In[444]:


x = '@tanmoy741127 i am good .....'


# In[445]:


re.sub(r'[^\w ]+',"",x)


# In[446]:


df['twitts']=df['twitts'].apply(lambda x:re.sub(r'[^\w ]+',"",x))


# In[447]:


df.sample(5)


# In[448]:


## Remove multiple spaces


# In[449]:


x = 'hi      hello   how are you'


# In[450]:


' '.join(x.split())


# In[451]:


df['twitts']=df['twitts'].apply(lambda x: ' '.join(x.split()))


# In[452]:


## Reomve HTML tags


# In[453]:


from bs4 import BeautifulSoup


# In[454]:


x = '<html><head><title>My cool Website</title></head><body>OK</body></html>'


# In[455]:


BeautifulSoup(x,'lxml').get_text()


# In[456]:


df['twitts']=df['twitts'].apply(lambda x : BeautifulSoup(x,'lxml').get_text())


# In[457]:


## Remove Accented Chars


# In[458]:


x="soupçon"


# In[459]:


import unicodedata


# In[460]:


def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD',x).encode('ascii','ignore').decode('utf-8','ignore')
    return x


# In[461]:


remove_accented_chars(x)


# In[462]:


df['twitts']=df['twitts'].apply(remove_accented_chars)


# In[463]:


# remove stop words


# In[464]:


x = "this is a stop words"


# In[465]:


stop_words


# In[466]:


' '.join([t for t in x.split() if t not in stop_words])


# In[467]:


df['twitts_no_stop']=df['twitts'].apply(lambda x:' '.join([t for t in x.split() if t not in stop_words]))


# In[468]:


df.sample(5)


# In[469]:


# convert into base or root form of word


# In[470]:


nlp = spacy.load('en_core_web_sm')


# In[471]:


x = "chocolates balls times. this is chocolates. what is times?"


# In[472]:


def make_to_base(x):
    x_list = []
    doc = nlp(x)
    for token in doc:
        lemmatized = str(token.lemma_)
        if lemmatized == "-PRON-" or lemmatized == "be":
            lemmatized = token.text
        x_list.append(lemmatized)
    return ' '.join(x_list)


# In[473]:


make_to_base(x)


# In[474]:


df['twitts']=df['twitts'].apply(lambda x: make_to_base(x))


# In[475]:


## Common words removal


# In[476]:


x = "this is this okay bye"


# In[477]:


text = ' '.join(df['twitts'])
len(text)


# In[478]:


text=text.split()


# In[479]:


len(text)


# In[481]:


freq_com = pd.Series(text).value_counts()


# In[484]:


f20=freq_com[:20]


# In[485]:


f20


# In[487]:


df['twitts']=df['twitts'].apply(lambda x : ' '.join([t for  t in x.split() if t not in f20]))


# In[488]:


df.sample(5)


# In[489]:


# Rare words removal


# In[491]:


rare20=freq_com.tail(20)


# In[492]:


df['twitts']=df['twitts'].apply(lambda x : ' '.join([t for  t in x.split() if t not in rare20]))


# In[493]:


df.sample(5)


# In[494]:


# Word cloud visualization


# In[496]:


get_ipython().system('pip install wordcloud')


# In[497]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[499]:


text = ' '.join(df['twitts'])


# In[503]:


wc = WordCloud(width=1600,height=800).generate(text)


# In[507]:


plt.figure(figsize=(20,10))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[508]:


## Spelling Correction


# In[509]:


get_ipython().system('pip install -U textblob')


# In[511]:


## python -m textblob.download_corpora


# In[512]:


from textblob import TextBlob


# In[517]:


x = 'thankks for waching it'


# In[518]:


TextBlob(x).correct()


# In[519]:


## Tokentization using TextBlob


# In[523]:


x = "thanks#watching this video. please make it"


# In[524]:


TextBlob(x).words


# In[525]:


doc = nlp(x)


# In[526]:


for token in doc:
    print(token)


# In[527]:


# Detecting Nouns


# In[539]:


x = "Breaking News : Donal trump, the president of USA is looking to play football"


# In[540]:


doc = nlp(x)


# In[541]:


for noun in doc.noun_chunks:
    print(noun)


# In[542]:


## Language translation and detection


# In[543]:


x


# In[544]:


tb = TextBlob(x)


# In[545]:


tb.detect_language()


# In[548]:


tb.translate(to='bn')


# In[549]:


## Sentiment Prediction


# In[550]:


from textblob.sentiments import NaiveBayesAnalyzer


# In[552]:


x = "we all standds together. we are gonna win this fight"


# In[556]:


TextBlob(x,analyzer=NaiveBayesAnalyzer()).sentiment


# In[ ]:




