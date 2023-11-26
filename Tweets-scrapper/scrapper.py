from flask import Flask, jsonify
import tweepy
import os

app = Flask(__name__)

# Authenticate to Twitter
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Route to get tweets from a specific user
@app.route('/tweets/<username>')
def get_tweets(username):
    try:
        # Retrieve tweets from the user's timeline
        tweets = api.user_timeline(screen_name=username, count=10, tweet_mode='extended')

        # Extract tweet texts
        tweet_texts = [tweet.full_text for tweet in tweets]

        return jsonify({'tweets': tweet_texts})
    except tweepy.TweepError as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)