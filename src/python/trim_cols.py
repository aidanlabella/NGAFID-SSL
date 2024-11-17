#!/usr/bin/env python3

import pandas as pd
import glob
import os

cols_set = {'vspdg', 'e1egtdivergence', 'crs', 'vspdcalculated', 'trk', 'normac', 'altmsl', 'vspd', 'oat', 'vplwas', 'hplwas', 'baroa', 'e1oilp', 'ias', 'latac', 'e1egt1', 'densityratio', 'e1oilt', 'altmsllagdiff', 'pitch', 'tas', 'fqtyr', 'totalfuel', 'trueairspeed(ft/min)', 'hplfd', 'magvar', 'e1egt2', 'altgps', 'amp1', 'fqtyl', 'volt1', 'e1fflow', 'altagl', 'altb', 'roll', 'stallindex', 'e1egt3', 'e1rpm', 'e1egt4', 'hdg', 'aoasimple', 'gndspd'}
max_rows = 0

DATA_PATH = '/mnt/crucial/data/ngafid/exports/loci_dataset_fixed_keys'

flights = glob.glob(os.path.join(DATA_PATH, "*.csv"))

for flight in flights:
  flight_data = pd.read_csv(flight)
  n_rows = len(flight_data)
  flight_cols = set(flight_data.columns)
  mutual = flight_cols & cols_set
  if len(mutual) == len(cols_set) and (1000 <= n_rows <= 10000):
    print("Processing: ", flight)
    max_rows = max(max_rows, len(flight_data))
    flight_data = flight_data[list(cols_set)]
    flight_data.to_csv(flight)
  else:
    print("Removing: ", flight)
    os.remove(flight)


  
flights = glob.glob(os.path.join(DATA_PATH, "*.csv"))
for flight in flights:
  print("Padding: ", flight)
  flight_data = pd.read_csv(flight)
  flight_data = flight_data.fillna(0)
  flight_data = flight_data.reindex(range(max_rows)).ffill()
  flight_data.to_csv(flight)
