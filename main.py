import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from copy import deepcopy

# Constants
fuel_cost = 4.5 # per l
fuel_usage = 13 # per 100km
city = {"Warszawa" : 0, "Krakow" : 1, "Lodz" : 2, "Wroclaw" : 3, "Poznan" : 4, "Gdansk" : 5, "Szczecin" : 6, "Bydgoszcz" : 7, "Lublin" : 8, "Katowice" : 9}

path1 = Path("data/dist_mat.csv")
path2 = Path("data/population.csv")
dist_matrix = pd.read_csv(path1, index_col=0)
fuel_cost_matrix = fuel_cost * fuel_usage * dist_matrix/100
population_vec = pd.read_csv(path2, index_col=0)

#print(dist_matrix)

# Interest function
x = np.arange(1,12)
y = -1/(1+0.5*np.exp(-0.5*x+3))+1
plt.plot(x, y, 'ro')
plt.grid()
plt.xlabel('Czas trwania [w dniach]')
plt.ylabel('Poziom zainteresowania')
plt.title('Zainteresowanie festiwalem')
plt.show()

# event = (miasto, dzień rozpoczęcia, dzień zakończenia)
war1 = ('Warszawa', 3, 12, 10)


krk1 = ('Kraków', 1, 6, 9)
krk2 = ('Kraków', 14, 21, 9)


lod1 = ('Łódź', 8, 13, 8)
lod2 = ('Łódź', 16, 21, 8)


wro1 = ('Wrocław', 11, 16, 7)
wro2 = ('Wrocław', 18, 26, 7)


poz1 = ('Poznań', 3, 10, 6)
poz2 = ('Poznań', 23, 28, 6)


gda1 = ('Gdańsk', 10, 14, 5)
gda2 = ('Gdańsk', 18, 23, 5)


szc1 = ('Szczecin', 7, 12, 4)
szc2 = ('Szczecin', 26, 31, 4)


byd1 = ('Bydgoszcz', 5, 9, 3)
byd2 = ('Bydgoszcz', 13, 18, 3)


lub1 = ('Lublin', 1, 6, 2)
lub2 = ('Lublin', 18, 24, 2)


kat1 = ('Katowice', 4, 9, 1)
kat2 = ('Katowice', 26, 31, 1)


#10
plt.plot([war1[1], war1[2]], [10, 10], linewidth=5.0, color='lightblue', label=war1[0])

#9
plt.plot([krk1[1], krk1[2]], [9, 9], linewidth=5.0, color='coral', label=krk1[0])
plt.plot([krk2[1], krk2[2]], [9, 9], linewidth=5.0, color='coral')

#8
plt.plot([lod1[1], lod1[2]], [8, 8], linewidth=5.0, color='indigo', label=lod1[0])
plt.plot([lod2[1], lod2[2]], [8, 8], linewidth=5.0, color='indigo')

#7
plt.plot([wro1[1], wro1[2]], [7, 7], linewidth=5.0, color='chartreuse', label=wro1[0])
plt.plot([wro2[1], wro2[2]], [7, 7], linewidth=5.0, color='chartreuse')

#6
plt.plot([poz1[1], poz1[2]], [6, 6], linewidth=5.0, color='gold', label=poz1[0])
plt.plot([poz2[1], poz2[2]], [6, 6], linewidth=5.0, color='gold')

#5
plt.plot([gda1[1], gda1[2]], [5, 5], linewidth=5.0, color='brown', label=gda1[0])
plt.plot([gda2[1], gda2[2]], [5, 5], linewidth=5.0, color='brown')

#4
plt.plot([szc1[1], szc1[2]], [4, 4], linewidth=5.0, color='yellow', label=szc1[0])
plt.plot([szc2[1], szc2[2]], [4, 4], linewidth=5.0, color='yellow')

#3
plt.plot([byd1[1], byd1[2]], [3, 3], linewidth=5.0, color='fuchsia', label=byd1[0])
plt.plot([byd2[1], byd2[2]], [3, 3], linewidth=5.0, color='fuchsia')

#2
plt.plot([lub1[1], lub1[2]], [2, 2], linewidth=5.0, color='teal', label=lub1[0])
plt.plot([lub2[1], lub2[2]], [2, 2], linewidth=5.0, color='teal')

#1
plt.plot([kat1[1], kat1[2]], [1, 1], linewidth=5.0, color='darkgreen', label=kat1[0])
plt.plot([kat2[1], kat2[2]], [1, 1], linewidth=5.0, color='darkgreen')



plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# Parking cost function
basic_price_per_hour = {"Warszawa" : 10.0, "Krakow" : 9.0, "Lodz" : 8.4, "Wroclaw" : 8.0, "Poznan" : 8.0, "Gdansk" : 7.8, "Szczecin" : 7.0, "Bydgoszcz" : 6.4, "Lublin" : 6.2, "Katowice" : 6.0}
# Cena za godzine jest uzalezniona od liczby dni parkingowych, spada o 20%.
# PRZYKLAD: w Warszawie spędzamy 3 dni, opłata za parking wyniesie 24*10+24*10*0.8+24*10*0.8*0.8=585.6