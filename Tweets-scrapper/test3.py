import requests
import pandas as pd

twitter_data = []

payload = { 'api_key': 'ce95f4709f96288f90647b6471482848', 'url': 'https://workshops.hackclub.com/selenium/' } 

response = requests.get('https://api.scraperapi.com/', params=payload)

data = response.json(
    
)
data