import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# key = name of event, item = list containing starting day, ending day,
# parking per hour cost, accommodation per night cost and base daily profit
EVENTS = {"Warszawa": [3, 12, 10.00, 110, 1918.04], "Krakow1": [1, 6, 9.00, 100, 1198.91],
          "Krakow2": [14, 21, 9.00, 100, 1198.91], "Lodz1": [8, 13, 8.40, 90, 1113.55],
          "Lodz2": [16, 21, 8.40, 90, 1113.55], "Wroclaw1": [11, 16, 8.00, 100, 967.42],
          "Wroclaw2": [18, 26, 8.00, 100, 967.42], "Poznan1": [3, 10, 8.00, 80, 816.06],
          "Poznan2": [23, 28, 8.00, 80, 816.06], "Gdansk1": [10, 14, 7.80, 80, 800.34],
          "Gdansk2": [18, 23, 7.80, 80, 800.34], 'Szczecin1': [7, 12, 7.00, 80, 747.78],
          "Szczecin2": [26, 31, 7.00, 80, 747.78], "Bydgoszcz1": [5, 9, 6.40, 70, 652.41],
          "Bydgoszcz2": [13, 18, 6.40, 70, 652.41], "Lublin1": [1, 6, 6.20, 70, 629.26],
          "Lublin2": [18, 24, 6.20, 70, 629.26], "Katowice1": [4, 9, 6.00, 60, 596.62],
          "Katowice2": [26, 31, 6.00, 60, 596.62]}

# tuple version
EVENTSt = [("Warszawa", 3, 12, 10.00, 110, 1918.04), ("Krakow1", 1, 6, 9.00, 100, 1198.91),
           ("Krakow2", 14, 21, 9.00, 100, 1198.91), ("Lodz1", 8, 13, 8.40, 90, 1113.55),
           ("Lodz2", 16, 21, 8.40, 90, 1113.55), ("Wroclaw1", 11, 16, 8.00, 100, 967.42),
           ("Wroclaw2", 18, 26, 8.00, 100, 967.42), ("Poznan1", 3, 10, 8.00, 80, 816.06),
           ("Poznan2", 23, 28, 8.00, 80, 816.06), ("Gdansk1", 10, 14, 7.80, 80, 800.34),
           ("Gdansk2", 18, 23, 7.80, 80, 800.34), ('Szczecin1', 7, 12, 7.00, 80, 747.78),
           ("Szczecin2", 26, 31, 7.00, 80, 747.78), ("Bydgoszcz1", 5, 9, 6.40, 70, 652.41),
           ("Bydgoszcz2", 13, 18, 6.40, 70, 652.41), ("Lublin1", 1, 6, 6.20, 70, 629.26),
           ("Lublin2", 18, 24, 6.20, 70, 629.26), ("Katowice1", 4, 9, 6.00, 60, 596.62),
           ("Katowice2", 26, 31, 6.00, 60, 596.62)]
fuel_cost = 4.5  # per l
fuel_usage = 13  # per 100km
path1 = Path("../data/dist_mat.csv")
dist_matrix = pd.read_csv(path1, index_col=0)
FUEL_COST = fuel_cost * fuel_usage * dist_matrix/100
#print(EVENTS['Warszawa'][1])


