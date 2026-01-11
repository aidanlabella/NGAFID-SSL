#!/usr/bin/env python 

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# from ../sample_flights.combine_flight_data import flight_paths
#
import math
import glob
import os

# CSV_FILE = "events.csv"
# FLIGHT_ID_FILE = "flight_ids.csv"
# NUM_FLIGHTS = 7679

CSV_FILE = "/mnt/crucial/data/ngafid/exports/loci_dataset_fixed_keys/events/all_events.csv"
FLIGHTS_PATH = "/mnt/crucial/data/ngafid/exports/loci_dataset_fixed_keys/flights"
FLIGHT_ID_FILE = "/mnt/crucial/data/ngafid/exports/loci_dataset_fixed_keys/flight_ids.csv"


FLIGHTS = glob.glob(os.path.join(FLIGHTS_PATH, "*.csv"))


class ScoreDatasetGenerator():
  def __init__(self):
    self.events = pd.read_csv(CSV_FILE)
    self.flight_ids = pd.read_csv(FLIGHT_ID_FILE)
    self.scores = None

    self.num_flights = len(self.flight_ids)

    self.get_scores()
      
  def get_scores(self):
      pass

  def plot_non_zero_scores(self):
      pass

  def plot_all_scores(self):
      pass

  def pair_generator(self, non_zero=False):
      pass

# s = ScoreDatasetGenerator()
# print(s.pair_generator())
