import pandas as pd
from ntscraper import Nitter

scraper = Nitter()

tweets = scraper.get_tweets('imVkohli', mode = 'user', number=5)

tweets