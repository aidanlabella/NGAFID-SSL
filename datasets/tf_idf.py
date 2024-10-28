#!/usr/bin/env python 

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import math

CSV_FILE = "events.csv"
FLIGHT_ID_FILE = "flight_ids.csv"

NUM_FLIGHTS = 7679

class ScoreDatasetGenerator():
  def __init__(self):
    self.events = pd.read_csv(CSV_FILE)
    self.flight_ids = pd.read_csv(FLIGHT_ID_FILE)
    self.scores = None
    self.get_scores()
      
  def get_scores(self):
    event_flight_counts = self.events.groupby('name')['flight_id'].nunique().reset_index(name='NUM_FLIGHTS')
    event_flight_counts = event_flight_counts.set_index('name')['NUM_FLIGHTS'].to_dict()
    occurences = self.events.groupby(['flight_id', 'name']).size().reset_index(name='counts')

    flights_tfidf = []

    for flight_id in self.events["flight_id"].unique():
      event_counts = occurences[occurences["flight_id"] == flight_id]
      tf_idf = 0
      for _, row in event_counts.iterrows():
        event = row["name"]
        tf = row["counts"]
        flights_u_event = event_flight_counts[event]
        idf = math.log10(NUM_FLIGHTS/flights_u_event)
        tf_idf += tf*idf

      flights_tfidf.append([flight_id, tf_idf])

    flights_tfidf = pd.DataFrame(flights_tfidf, columns=['flight_id', 'tfidf'])

    self.scores = pd.merge(self.flight_ids, flights_tfidf, on='flight_id', how='left')
    self.scores['tfidf'] = self.scores['tfidf'].fillna(0)

  def plot_non_zero_scores(self):
    sns.histplot(self.scores[self.scores['tfidf'] > 0]['tfidf'], kde=True)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Non-Zero Score Distribution")
    plt.savefig("Non_Zero_Scores.png")
    plt.close()

  def plot_all_scores(self):
    sns.histplot(self.scores[self.scores['tfidf'] > 0]['tfidf'], kde=True)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Complete Score Distribution")
    plt.savefig("Scores.png")
    plt.close()

  def pair_generator(self, non_zero=False):
    data = self.scores['tfidf']
    if non_zero:
      data = self.scores[self.scores['tfidf'] > 0]['tfidf']

    data = data.sort_values()
    pairs = [(data.index[i], data.index[i+1]) for i in range(0,len(data)-1, 2)]
    
    pair_df = pd.DataFrame(
      {
        "Positive Pairs": pairs
      }
    )

    return pair_df    


s = ScoreDatasetGenerator()
pairs = s.pair_generator(True)
print(pairs)


