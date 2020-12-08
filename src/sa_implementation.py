import numpy as np
from src.data_prep import EVENTS
from enum import Enum
import matplotlib.pyplot as plt

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


class SollMethod(Enum):
    GEO = 1
    BOLZ = 2
    CAUCHY = 3


class Sollution:
    def __init__(self, soll_init, T_init, soll_method, event_depo):
        self.soll_init = soll_init
        self.T_init = T_init
        self.soll_method = soll_method
        self.event_depo = event_depo
        self.T_list = []
        self.acc_list = []
        self.profit_list = []

    def get_temp(self, k):
        T = 0
        alpha = 0.999
        if self.soll_method is SollMethod.GEO:
            T = self.T_init * (alpha**k)
        if self.soll_method is SollMethod.BOLZ:
            T = self.T_init * (1/(1+np.log10(k)))
        if self.soll_method is SollMethod.CAUCHY:
            T = self.T_init * (1/(1+k))
        return T

    def check_accessibility(self, time):  # works on tuples only
        poss_events = []
        for tuple in self.event_storage:
            if time <= tuple[1] + 1 and time >= tuple[2] + 1:
                poss_events.append(tuple)
        while poss_events == []:
            poss_events = self.check_accessibility(time+1)
        return poss_events

    @staticmethod
    def calculate_max_profit(sollution):
        max_profit = 0
        for event in sollution:
            max_profit += event.calculate_profit() - event.calculate_cost()  # - fuel cost
        return max_profit

    def sym_ann_algotirhm(self):
        profit0 = Sollution.calculate_max_profit(self.soll_init)
        soll0 = self.soll_init
        Temp_iter = 1000
        for i in range(Temp_iter):  # Set
            print(i, 'profit = ', profit0)

            T = self.get_temp(i)
            self.T_list.append(T)

            for j in range(50):  # Set
                acc = 1
                # Delete one event
                r1 = np.random.randint(0, len(self.soll_init))
                depo = soll0
                del_event = soll0[r1]
                # Replace event
                replacement_time = del_event.time_start
                t1 = np.random.randint(1, soll0[2] - soll0[1] + 1)
                if t1 > del_event.time_stay:
                    t1 = del_event.time_stay

                possible_events = self.check_accessibility(replacement_time)
                r2 = np.random.randint(0, len(possible_events))
                soll0[r1] = Event(possible_events[r2][0], possible_events[r2][1], t1)

                # Get the new profit
                profit1 = Sollution.calculate_max_profit(soll0)

                if profit1 > profit0:
                    # Accept new sollution
                    profit0 = profit1
                else:
                    # Accept new (worse) sollution with a given probability
                    acc = np.random.uniform()
                    if acc < np.exp(-(profit0 - profit1) / T):
                        # Accept the sollution
                        profit0 = profit1
                    else:
                        # Do not accept the sollution
                        soll0 = depo

            self.acc_list.append(acc)
            self.profit_list.append(profit0)

            # Plot the result
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        # Profit plot
        ax2.axis([0, Temp_iter, 0, 10])
        ax2.scatter(np.linspace(0, Temp_iter, num=Temp_iter, endpoint=False), self.profit_list, s=1.0, color='darkgreen')
        # Acceptance plot
        ax3.axis([0, Temp_iter, 0, 1])
        ax3.scatter(np.linspace(0, Temp_iter, num=Temp_iter, endpoint=False), self.acc_list, s=1.0, color='darkgreen')
        # Temperature plot
        ax4.axis([0, Temp_iter, 0, T_init])
        ax4.scatter(np.linspace(0, Temp_iter, num=Temp_iter, endpoint=False), self.T_list, s=1.0, color='darkgreen')
        '''
        ax1.clear()
        for first, second in zip(coords[:-1], coords[1:]):
            ax1.plot([first.x, second.x], [first.y, second.y], 'b')
        for c in coords:
            ax1.plot(c.x, c.y, 'ro')
        plt.pause(0.0001)
        '''
        plt.show()
        return soll0




