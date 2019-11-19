
### In this project we will use natural language processing, specifically sentiment analysis, to detect hate speech in tweets

Given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' indicating that the tweet is not racist/sexist, we will predict the labels of a test dataset.

## Data Preprocessing and Cleaning


```
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

%matplotlib inline
```


```
from google.colab import drive
```


```
drive.mount('./gdrive')
```


```
import os

```


```
!ls
```

    gdrive	sample_data



```
os.chdir('./gdrive/My Drive/Google Colaboratory/Colab Notebooks')
```


```
!ls
```

    'Data Science   Machine Learning'  'Finance Stuff'  'Freelance Work'



```
os.chdir('./Data Science   Machine Learning')
```


```
!ls
```

    '911 Calls Project'		  'NBA Salary Prediction'
    'American Ninja Warrior Project'  'Neural Transfer Project'
    'Beatles NLP'			  'Recommender Systems'
    'Computer Vision'		  'Scrapy Web Scraping'
    'Machine Learning Projects'	  'Twitter Sentiment Analysis'



```
os.chdir('./Twitter Sentiment Analysis/Files')
```


```
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```


```
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>@user when a father is dysfunctional and is s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>@user @user thanks for #lyft credit i can't us...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>bihday your majesty</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>#model   i love u take with u all the time in ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>factsguide: society now    #motivation</td>
    </tr>
  </tbody>
</table>
</div>




```
train[train['label'] == 1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>1</td>
      <td>@user #cnn calls #michigan middle school 'buil...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>1</td>
      <td>no comment!  in #australia   #opkillingbay #se...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>1</td>
      <td>retweet if you agree!</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>1</td>
      <td>@user @user lumpy says i am a . prove it lumpy.</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>1</td>
      <td>it's unbelievable that in the 21st century we'...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31934</th>
      <td>31935</td>
      <td>1</td>
      <td>lady banned from kentucky mall. @user  #jcpenn...</td>
    </tr>
    <tr>
      <th>31946</th>
      <td>31947</td>
      <td>1</td>
      <td>@user omfg i'm offended! i'm a  mailbox and i'...</td>
    </tr>
    <tr>
      <th>31947</th>
      <td>31948</td>
      <td>1</td>
      <td>@user @user you don't have the balls to hashta...</td>
    </tr>
    <tr>
      <th>31948</th>
      <td>31949</td>
      <td>1</td>
      <td>makes you ask yourself, who am i? then am i a...</td>
    </tr>
    <tr>
      <th>31960</th>
      <td>31961</td>
      <td>1</td>
      <td>@user #sikh #temple vandalised in in #calgary,...</td>
    </tr>
  </tbody>
</table>
<p>2242 rows × 3 columns</p>
</div>



Things We Should Clean Up

*   Twitter handles
*   Punctuation, numbers, special characters
*   Smaller words that don't add value






```
# Removing Twitter handles

combi = train.append(test, ignore_index=True)
```

    /usr/local/lib/python3.6/dist-packages/pandas/core/frame.py:7138: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      sort=sort,



```
def remove_pattern(input_txt, pattern):
  r = re.findall(pattern, input_txt)
  for i in r:
    input_txt = re.sub(i, '', input_txt)

  return input_txt
```


```
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
```


```
# Remove special characters, numbers, punctuation

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " " )
```


```
# Removing short words

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
```


```
combi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
      <th>tweet</th>
      <th>tidy_tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
      <td>@user when a father is dysfunctional and is s...</td>
      <td>when father dysfunctional selfish drags kids i...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.0</td>
      <td>@user @user thanks for #lyft credit i can't us...</td>
      <td>thanks #lyft credit cause they offer wheelchai...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.0</td>
      <td>bihday your majesty</td>
      <td>bihday your majesty</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.0</td>
      <td>#model   i love u take with u all the time in ...</td>
      <td>#model love take with time</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.0</td>
      <td>factsguide: society now    #motivation</td>
      <td>factsguide society #motivation</td>
    </tr>
  </tbody>
</table>
</div>



## Tokenization


```
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
```


```
tokenized_tweet.head()
```




    0    [when, father, dysfunctional, selfish, drags, ...
    1    [thanks, #lyft, credit, cause, they, offer, wh...
    2                              [bihday, your, majesty]
    3                     [#model, love, take, with, time]
    4                   [factsguide, society, #motivation]
    Name: tidy_tweet, dtype: object



## Stemming


```
from nltk.stem.porter import *
stemmer = PorterStemmer()
```


```
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])


```


```
tokenized_tweet.head()
```




    0    [when, father, dysfunct, selfish, drag, kid, i...
    1    [thank, #lyft, credit, caus, they, offer, whee...
    2                              [bihday, your, majesti]
    3                     [#model, love, take, with, time]
    4                         [factsguid, societi, #motiv]
    Name: tidy_tweet, dtype: object




```
for i in range(len(tokenized_tweet)):
  tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tidy_tweet'] = tokenized_tweet
```


```
combi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
      <th>tweet</th>
      <th>tidy_tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
      <td>@user when a father is dysfunctional and is s...</td>
      <td>when father dysfunct selfish drag kid into dys...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.0</td>
      <td>@user @user thanks for #lyft credit i can't us...</td>
      <td>thank #lyft credit caus they offer wheelchair ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.0</td>
      <td>bihday your majesty</td>
      <td>bihday your majesti</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.0</td>
      <td>#model   i love u take with u all the time in ...</td>
      <td>#model love take with time</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.0</td>
      <td>factsguide: society now    #motivation</td>
      <td>factsguid societi #motiv</td>
    </tr>
  </tbody>
</table>
</div>



## Data Visualization


*   What are the most common words in the dataset?
*   What are the most common words for negative and positive tweets?
*   How many hashtags are in a tweet?
*   What trends are associated?





### Common words


```
all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```


![png](images/Twitter%20Sentiment%20Analysis_33_0.png)


### Words in positive tweets


```
normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

```


![png](images/Twitter%20Sentiment%20Analysis_35_0.png)


### Words in negative tweets


```
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```


![png](images/Twitter%20Sentiment%20Analysis_37_0.png)


## Dealing with hashtags


```
def hashtag_extract(x):
  hashtags = []
  for i in x:
    ht = re.findall(r"#(\w+)", i)
    hashtags.append(ht)

  return hashtags
```


```
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

HT_regular = sum(HT_regular, [])
HT_negative = sum(HT_negative, [])
```

### Plotting most common hashtags


```
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

d = d.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x = "Hashtag", y="Count")
ax.set(ylabel = 'Count')
plt.show()
```


![png](images/Twitter%20Sentiment%20Analysis_42_0.png)



```
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()),
                  'Count': list(b.values())})

e = e.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=e, x = "Hashtag", y="Count")
ax.set(ylabel = 'Count')
plt.show()
```


![png](images/Twitter%20Sentiment%20Analysis_43_0.png)


## Extracting Features from Cleaned Tweets

### Bag-of-Words


```
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
```

### TF-IDF


```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
```

## Model Building

### Building with Bag-of-words features


```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
```


```
train_bow = bow[:31962, :]
test_bow = bow[31962:, :]
```


```
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)
```


```
lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```
prediction = lreg.predict_proba(xvalid_bow)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)
```




    0.5307820299500832



### Building model using TF-IDF features


```
train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962:, :]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    0.5446507515473032




```
0.52184032
```




    0.52184032




```

```
