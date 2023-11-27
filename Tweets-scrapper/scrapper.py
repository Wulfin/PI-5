from flask import Flask
import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import csv

app = Flask(__name__)


# Authenticate to Twitter
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")


def save_to_json(tweet_data):
    filename = "tweets.json"  # JSON filename
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(tweet_data, jsonfile, indent=4)  # Write tweet data to JSON file


# Function to save tweet data to a CSV file
def save_to_csv(tweet_data):
    filename = "tweets.csv"
    fieldnames = ["username", "datetime", "tweet_text", "replies", "retweets", "likes"]  # Update fieldnames to match dictionary keys

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tweet in tweet_data:
            writer.writerow(tweet)  # Write the tweet dictionary directly

    print("Tweet data saved to", filename)


# Function to perform the search and scrape tweets
def search(search_term):
    login_url = "https://twitter.com/login"

    # Initialize Chrome driver with the configured options
    # driver = webdriver.Chrome(ChromeDriverManager().install())
    driver = webdriver.Firefox()
    amount = 50
    
    driver.get(login_url)
    time.sleep(2)
    driver.find_element("xpath", '//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input').send_keys('saifelislambettaoui1955@gmail.com')
    time.sleep(1)
    driver.find_element("xpath", '//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[6]/div').click()
    time.sleep(2)
    # driver.find_element("xpath", '/html/body/div/div/div/div[2]/main/div/div/div[1]/form/div/div[2]/label/div/div[2]/div/input').send_keys('20SAIF02saif')
    time.sleep(1)
    driver.find_element("xpath", '//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div[1]/div/div/div/div').click()
    time.sleep(2)
    
    search_url = "https://twitter.com/search?q=" + str(search_term) + "&src=typed_query"
    driver.get(search_url)
    time.sleep(5)
    start_time = time.time()
    tweet_data = []

    while len(tweet_data) < amount:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(1)
        resp = driver.page_source
        new_tweets = parse(resp, tweet_data)
        tweet_data.extend(new_tweets)

        elapsed_time = time.time() - start_time
        if elapsed_time >= amount/10:
            break

    print("Number of tweets found:", len(tweet_data))
    driver.close()
    return tweet_data[:amount]


# Function to parse tweet HTML and extract relevant information
def parse(resp, tweet_data):
    soup = BeautifulSoup(resp, 'html.parser')
    tweets = soup.find_all("div", {"data-testid": "cellInnerDiv"})
    result = []

    for tweet in tweets:
        tweet_info = {}

        try:
            tweet_info["username"] = tweet.find("div", {"data-testid": "User-Name"}).text
        except AttributeError:
            tweet_info["username"] = None
        
        try:
            tweet_info["datetime"] = tweet.find("time").get("datetime")  # Extract the "datetime" attribute value
        except AttributeError:
            tweet_info["datetime"] = None

        try:
            tweet_info["tweet_text"] = tweet.find("div", {"data-testid": "tweetText"}).text
        except AttributeError:
            tweet_info["tweet_text"] = None
        
        try:
            tweet_info["replies"] = tweet.find("div", {"data-testid": "reply"}).text
        except AttributeError:
            tweet_info["replies"] = None
        
        try:
            tweet_info["retweets"] = tweet.find("div", {"data-testid": "retweet"}).text
        except AttributeError:
            tweet_info["retweets"] = None
        
        try:
            tweet_info["likes"] = tweet.find("div", {"data-testid": "like"}).text
        except AttributeError:
            tweet_info["likes"] = None

        if not tweet_info["tweet_text"] in tweet_data:
            result.append(tweet_info)

    return result


@app.route('/tweets/<search_term>')
def get_tweets(search_term):
    
    tweet_data = search(search_term)
    save_to_csv(tweet_data)
    save_to_json(tweet_data)
    return tweet_data
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)