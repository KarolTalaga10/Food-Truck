import numpy as np
from src.data_prep import EVENTS


class Event:
    def __init__(self, name, time_start, time_stay):
        self.name = name
        self.time_start = time_start
        self.time_stay = time_stay

    def calculate_cost(self) -> float:
        # Parking
        multiplier = 1
        parking = 0
        for i in range(self.time_stay):
            day = 24*EVENTS[self.name][2]*multiplier
            multiplier = multiplier*0.8
            parking += day
        # Accommodation
        accommodation = EVENTS[self.name][3]*self.time_stay
        cost = parking + accommodation
        return round(cost, 2)

    def calculate_profit(self) -> float:
        profit = 0
        d = self.time_start
        y = lambda x: -1 / (1 + 0.5 * np.exp(-0.5 * x + 3)) + 1
        for i in range(self.time_stay):
            if self.time_start + i > EVENTS[self.name][1]:
                break
            day_number = d - EVENTS[self.name][0] + 1
            daily_profit = y(day_number) * EVENTS[self.name][4]
            d += 1
            profit += daily_profit
        return round(profit, 2)




