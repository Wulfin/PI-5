# Projet Integre S5 - BETTAOUI / LAAMOUMRI / MALKI

## Execution du Projet :

Apres avoir telecharge les differents microservices, il faut extraire le contenue de ressources.rar dans le meme repertoire qui se trouve dans le microservice Sentiment analysis

Ensuite il faut lancer les differents microservices avec l'ordre suivant (Scrapper -> Sentiment Analysis -> Backend -> Frontend)

Dans le microservice Sentiment Analysis lancer l'application flask nommé sentiment-analysis.py

Apres avoir lancer les differents microservices, vous pouvez lancer une requete depuis le frontend dans l'onglet Request Settings et visualiser le resultat depuis l'onglet Tweets.

<br>
<br>
## Microservice du Twitter Scraper Flask API

Ce microservice implémente une API Flask pour le scraping de tweets en utilisant un navigateur sans tête (Selenium) et BeautifulSoup. Il vous permet de rechercher des tweets en fonction d'un terme de recherche spécifié et de récupérer des informations pertinentes telles que les noms d'utilisateur, le texte des tweets, la date et l'heure, les réponses, les retweets et les likes.

Prérequis
Avant d'exécuter le projet, assurez-vous d'avoir les dépendances suivantes installées :

  Python (version 3.6 ou supérieure)
  Flask
  Selenium
  BeautifulSoup
  Firefox 

```bash
pip install flask selenium beautifulsoup4
```

Après on accède au dossier du microservice et on lance le microservice :
```bash
cd Microservice-Scraper
pyhton scraper.py
```
![Scraper server](screenshots/screenshot1.png)

## Microservice du Sentiment Analysis Flask API
Ce microservice implémente une API Flask pour l'analyse de sentiment des tweets. Il utilise un modèle d'apprentissage automatique pré-entraîné pour prédire le sentiment des tweets fournis. Le modèle d'analyse de sentiment est basé sur la régression logistique et est entraîné sur un ensemble de données contenant des tweets positifs et négatifs.

Prérequis
Avant d'exécuter le projet, assurez-vous d'avoir les dépendances suivantes installées :

Python (version 3.6 ou supérieure)
Flask
NLTK
scikit-learn
pandas
requests
Vous pouvez installer ces dépendances en utilisant la commande suivante :
```bash
pip install flask nltk scikit-learn pandas requests
```
Puis on accède au dossier du microservice et on lance le server :
```bash
cd Microservice-SentimentAnalysis
pyhton app.py
```
![Sentiment Analysis server](screenshots/screenshot2.png)

