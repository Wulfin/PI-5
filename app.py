import requests
from flask import Flask, jsonify, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import string
import warnings

# for all NLP related operations on text
import nltk
from jedi.api.refactoring import inline
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.classify import NaiveBayesClassifier
from pyspark.pandas.plot import matplotlib
from pyspark.resource import requests as pyspark_requests
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# To mock web-browser and scrap tweets
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# To consume Twitter's API
import tweepy
from tweepy import OAuthHandler

# To identify the sentiment of text
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.np_extractors import ConllExtractor

# ignoring all the warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# downloading stopwords corpus
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('conll2000')
nltk.download('brown')
stopwords = set(stopwords.words("english"))

# for showing all the plots inline
# %matplotlib inline

app = Flask(__name__)


def fetch_sentiment_using_SIA(text):
    if text is None:
        return None
    sid = SentimentIntensityAnalyzer()
    polarity_scores = sid.polarity_scores(text)
    return 'Negative' if polarity_scores['neg'] > polarity_scores['pos'] else 'Positive'


# Removing '@names'
def remove_pattern(text, pattern_regex):
    if text is None:
        return ""
    r = re.findall(pattern_regex, text)
    for i in r:
        text = re.sub(i, '', text)
    return text

# Helper class, will help in preprocessing phrase terms
class PhraseExtractHelper(object):
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()

    def leaves(self, tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            yield subtree.leaves()

    def normalise(self, word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        word = word.lower()
        # word = self.stemmer.stem_word(word) # We will loose the exact meaning of the word
        word = self.lemmatizer.lemmatize(word)
        return word

    def acceptable_word(self, word):
        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
        accepted = bool(3 <= len(word) <= 40
                        and word.lower() not in stopwords
                        and 'https' not in word.lower()
                        and 'http' not in word.lower()
                        and '#' not in word.lower()
                        )
        return accepted

    def get_terms(self, tree):
        for leaf in self.leaves(tree):
            term = [self.normalise(w) for w, t in leaf if self.acceptable_word(w)]
            yield term

def generate_wordcloud(all_words):
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5,
                          colormap='Dark2').generate(all_words)

    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

# function to collect hashtags
def hashtag_extract(text_list):
    hashtags = []
    # Loop over the words in the tweet
    for text in text_list:
        ht = re.findall(r"#(\w+)", text)
        hashtags.append(ht)
    return hashtags

def generate_hashtag_freqdist(hashtags):
    a = nltk.FreqDist(hashtags)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    # selecting top 15 most frequent hashtags
    d = d.nlargest(columns="Count", n=25)
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(data=d, x="Hashtag", y="Count")
    plt.xticks(rotation=80)
    ax.set(ylabel='Count')
    plt.show()

def plot_confusion_matrix(matrix):
    plt.clf()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Set2_r)
    classNames = ['Positive', 'Negative']
    plt.title('Confusion Matrix')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TP', 'FP'], ['FN', 'TN']]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(matrix[i][j]))
    plt.show()


def naive_model(X_train, X_test, y_train, y_test):
    naive_classifier = GaussianNB()
    naive_classifier.fit(X_train.toarray(), y_train)
    # predictions over test set
    predictions = naive_classifier.predict(X_test.toarray())
    # calculating Accuracy Score
    print(f'Accuracy Score - {accuracy_score(y_test, predictions)}')
    conf_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    plot_confusion_matrix(conf_matrix)

def save_to_json(tweet_data):
    filename = "tweets.json"  # JSON filename
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(tweet_data, jsonfile, indent=4)  # Write tweet data to JSON file


@app.route('/sentiments/<search_term>')
def sentiment_analysis(search_term):
    scrapper_url = f'http://localhost:5001/tweets/{search_term}'

    # Use requests.get to make the actual request
    response = requests.get(scrapper_url)

    if response.status_code == 200:
        tweet_data = response.json()
        if not tweet_data:
            return jsonify({"message": "No tweets found."})

        # Create a DataFrame from the fetched tweet data
        df = pd.DataFrame(tweet_data)

        # Perform sentiment analysis using your custom logic (e.g., SIA)
        df['sentiment'] = df['tweet_text'].apply(lambda tweet: fetch_sentiment_using_SIA(tweet))

        df['tidy_tweets'] = np.vectorize(remove_pattern)(df['tweet_text'], "@[\w]*: | *RT*")

        # Removing links (http | https)
        cleaned_tweets = []
        for index, row in df.iterrows():
            # Here we are filtering out all the words that contains link
            words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]
            cleaned_tweets.append(' '.join(words_without_links))
        df['tidy_tweets'] = cleaned_tweets

        # Removing tweets with empty text
        df = df[df['tidy_tweets'] != '']
        #
        # # Dropping duplicate rows
        # df.drop_duplicates(subset=['tidy_tweets'], keep=False)
        #
        # # Resetting index
        # df = df.reset_index(drop=True)
        #
        # # Removing Punctuations, Numbers and Special characters
        # df['absolute_tidy_tweets'] = df['tidy_tweets'].str.replace("[^a-zA-Z# ]", "")
        #
        # # Removing Stop words
        # stopwords_set = set(stopwords)
        # cleaned_tweets = []
        # for index, row in df.iterrows():
        #     # filerting out all the stopwords
        #     words_without_stopwords = [word for word in row.absolute_tidy_tweets.split() if
        #                                not word in stopwords_set and '#' not in word.lower()]
        #
        #     # finally creating tweets list of tuples containing stopwords(list) and sentimentType
        #     cleaned_tweets.append(' '.join(words_without_stopwords))
        # df['absolute_tidy_tweets'] = cleaned_tweets
        #
        # # Tokenize *'absolute_tidy_tweets'*
        # tokenized_tweet = df['absolute_tidy_tweets'].apply(lambda x: x.split())
        #
        # # Converting words to Lemma
        # word_lemmatizer = WordNetLemmatizer()
        # tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])
        #
        # # Joining all tokens into sentences
        # for i, tokens in enumerate(tokenized_tweet):
        #     tokenized_tweet[i] = ' '.join(tokens)
        # df['absolute_tidy_tweets'] = tokenized_tweet
        #
        # # Grammatical rule to identify phrases
        # sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
        # grammar = r"""
        #     NBAR:
        #         {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        #
        #     NP:
        #         {<NBAR>}
        #         {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        # """
        # chunker = nltk.RegexpParser(grammar)
        #
        # # New feature called 'key_phrases', will contain phrases for corresponding tweet
        # textblob_key_phrases = []
        # extractor = ConllExtractor()
        # for index, row in df.iterrows():
        #     # filerting out all the hashtags
        #     words_without_hash = [word for word in row.tidy_tweets.split() if '#' not in word.lower()]
        #     hash_removed_sentence = ' '.join(words_without_hash)
        #     blob = TextBlob(hash_removed_sentence, np_extractor=extractor)
        #     textblob_key_phrases.append(list(blob.noun_phrases))
        # textblob_key_phrases[:10]
        # df['key_phrases'] = textblob_key_phrases
        #
        # # Story Generation and Visualization
        #
        # all_words = ' '.join([text for text in df['absolute_tidy_tweets'][df.sentiment == 'Positive']])
        # generate_wordcloud(all_words)
        #
        # # Most common words in negative tweets
        # all_words = ' '.join([text for text in df['absolute_tidy_tweets'][df.sentiment == 'Negative']])
        # generate_wordcloud(all_words)
        #
        # # Most commonly used Hashtags
        #
        # hashtags = hashtag_extract(df['tidy_tweets'])
        # hashtags = sum(hashtags, [])
        #
        # # For sake of consistency, we are going to discard the records which contains no phrases i.e where tweets_df['key_phrases'] contains []
        # df2 = df[df['key_phrases'].str.len() > 0]
        #
        # # Feature Extraction
        #
        # # Feature Extraction for 'Key Words'
        # # BOW features
        # bow_word_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words='english')
        # # bag-of-words feature matrix
        # bow_word_feature = bow_word_vectorizer.fit_transform(df2['absolute_tidy_tweets'])
        # # TF-IDF features
        # tfidf_word_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english')
        # # TF-IDF feature matrix
        # tfidf_word_feature = tfidf_word_vectorizer.fit_transform(df2['absolute_tidy_tweets'])
        #
        # # Feature Extraction for 'Key Phrases'
        # phrase_sents = df2['key_phrases'].apply(lambda x: ' '.join(x))
        # # BOW phrase features
        # bow_phrase_vectorizer = CountVectorizer(max_df=0.90, min_df=2)
        # bow_phrase_feature = bow_phrase_vectorizer.fit_transform(phrase_sents)
        # # TF-IDF phrase feature
        # tfidf_phrase_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2)
        # tfidf_phrase_feature = tfidf_phrase_vectorizer.fit_transform(phrase_sents)
        #
        # # Model Building: Sentiment Analysis
        #
        # # Map target variables to {0, 1}
        # target_variable = df2['sentiment'].apply(lambda x: 0 if x == 'neg' else 1)
        #
        # # Predictions on 'key words' based features
        # # BOW word features
        # X_train, X_test, y_train, y_test = train_test_split(bow_word_feature, target_variable, test_size=0.3,
        #                                                     random_state=272)
        # naive_model(X_train, X_test, y_train, y_test)
        # # TF-IDF word features
        # X_train, X_test, y_train, y_test = train_test_split(tfidf_word_feature, target_variable, test_size=0.3,
        #                                                     random_state=272)
        # naive_model(X_train, X_test, y_train, y_test)
        #
        # # Predictions on 'key phrases' based features
        # # BOW Phrase features
        # X_train, X_test, y_train, y_test = train_test_split(bow_phrase_feature, target_variable, test_size=0.3,
        #                                                     random_state=272)
        # naive_model(X_train, X_test, y_train, y_test)
        # # TF-IDF Phrase features
        # X_train, X_test, y_train, y_test = train_test_split(tfidf_phrase_feature, target_variable, test_size=0.3,
        #                                                     random_state=272)
        # naive_model(X_train, X_test, y_train, y_test)

        # Extract the required information for each tweet
        tweets_result = []
        for index, row in df.iterrows():
            timestamp = row.get('datetime', '')
            username = row.get('username', '')

            tweet_result = {
                "content": row['tidy_tweets'],
                "timestamp": timestamp,
                "sentiment": row['sentiment'],
                "username": username,
            }
            tweets_result.append(tweet_result)

    # print(df.head())
    # print(df.iterrows())
    # print(df['tidy_tweets'])
    # print(df['sentiment'])
    # print(tokenized_tweet)
    # generate_hashtag_freqdist(hashtags)
    # print(df2)
    save_to_json(tweets_result)
    return tweets_result

if __name__ == '__main__':
    app.run(debug=True, port=5002)
