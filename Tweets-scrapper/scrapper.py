from flask import Flask, jsonify
import tweepy
import os
import json
from twitter_scraper_selenium import scrape_keyword
import pandas as pd
import asyncio

app = Flask(__name__)

# Authenticate to Twitter
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")

# auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
# api = tweepy.API(auth, wait_on_rate_limit=True)

search_url = "https://api.twitter.com/2/tweets/search/recent"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
query_params = {'query': '(from:twitterdev -is:retweet) OR #twitterdev','tweet.fields': 'author_id'}


def scrape_profile_tweets_since_2023(username: str):
    kword = "from:" + username
    path = './users/' + username
    file_path = path + '.csv'
    tweets = scrape_keyword(
                            headless=True,
                            keyword=kword,
                            browser="chrome",
                            tweets_count=2, # Just last 2 tweets
                            filename=path,
                            output_format="csv",
                            since="2023-01-01",
                            # until="2025-03-02", # Until Right now
                            )
    data = pd.read_csv(file_path)
    data = json.loads(data.to_json(orient='records'))
    return data



@app.route('/tweets/<username>')
def get_tweets(username):
    return scrape_profile_tweets_since_2023(username)

if __name__ == '__main__':
    app.run(debug=True, port=5001)