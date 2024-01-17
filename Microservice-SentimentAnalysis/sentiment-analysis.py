import re
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
import json
from flask import Flask, jsonify
import requests

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

# Fonction qui importe le model et vectoriser
def load_models():
    '''
    Chargement des modèles à partir des fichiers sauvegardés.
    '''
    file = open('vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    file = open('Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    return vectoriser, LRmodel


# Fonction Pour le pre processing du texte
def preprocess(textdata):
    processedText = []
    if not textdata:
        return processedText
    wordLemm = WordNetLemmatizer()
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    for tweet in textdata:
        tweet = tweet.lower()
        tweet = re.sub(urlPattern,' URL',tweet)
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        tweet = re.sub(userPattern,' USER', tweet)        
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        tweetwords = ''
        for word in tweet.split():
            if len(word)>1:
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
        processedText.append(tweetwords)
    return processedText

def predict(vectoriser, model, text):
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

def analyze_sentiment_on_json(input_json_file, output_json_file):
    vectoriser, LRmodel = load_models()
    with open(input_json_file, 'r') as file:
        input_data = json.load(file)
    tweets = [entry['tweet_text'] for entry in input_data]
    df = predict(vectoriser, LRmodel, tweets)
    for entry, sentiment in zip(input_data, df['sentiment']):
        entry['sentiment'] = sentiment
    output_data = [
        {
            'username': entry['username'],
            'datetime': entry['datetime'],
            'tweet_text': entry['tweet_text'],
            'sentiment': entry['sentiment']
        }
        for entry in input_data
    ]
    with open(output_json_file, 'w') as file:
        json.dump(output_data, file, indent=2)

def processJSON(input_data):
    filtered_data = [entry for entry in input_data if 'tweet_text' in entry and entry['tweet_text']]
    vectoriser, LRmodel = load_models()
    tweets = [entry['tweet_text'] for entry in filtered_data]
    df = predict(vectoriser, LRmodel, tweets)
    for entry, sentiment in zip(filtered_data, df['sentiment']):
        entry['sentiment'] = sentiment
    return filtered_data

app = Flask(__name__)


# Endpoint pour acceder aux resultats
@app.route('/sentiments/<search_term>')
def receive_tweets(search_term):
    url = f'http://localhost:5001/tweets/{search_term}'
    response = requests.get(url)
    if response.status_code == 200:
        try:
            input_json_data = response.json()
        except ValueError as e:
            return jsonify({"error": "Impossible de parser les données JSON de l'endpoint spécifié."}), 500
        result_data = processJSON(input_json_data)
        return jsonify(result_data)
    else:
        return jsonify({"error": "Échec de la récupération des données JSON de l'endpoint spécifié."}), 500


if __name__ == '__main__':
    app.run(port=5002)
