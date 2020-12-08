import numpy as np
from src.data_prep import EVENTS, EVENTSt
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


class SolMethod(Enum):
    GEO = 1
    BOLZ = 2
    CAUCHY = 3


class Solution:
    def __init__(self, sol_init, T_init, sol_method, event_depo):
        self.sol_init = sol_init
        self.T_init = T_init
        self.sol_method = sol_method
        self.event_depo = event_depo
        self.T_list = []
        self.acc_list = []
        self.profit_list = []

    def get_temp(self, k):
        T = 0
        alpha = 0.999
        if self.sol_method is SolMethod.GEO:
            T = self.T_init * (alpha**k)
        if self.sol_method is SolMethod.BOLZ:
            T = self.T_init * (1/(1+np.log10(k)))
        if self.sol_method is SolMethod.CAUCHY:
            T = self.T_init * (1/(1+k))
        return T

    def check_accessibility(self, time, del_event):  # works on tuples only
        poss_events = []
        for tup in self.event_depo:
            if time + 1 >= tup[1] and time + 1 <= tup[2] and tup[0] is not del_event.name:
                poss_events.append(tup)
        return poss_events

    @staticmethod
    def calculate_max_profit(solution):
        max_profit = 0
        for event in solution:
            max_profit += event.calculate_profit() - event.calculate_cost()  # - fuel cost
        return max_profit

    def sym_ann_algorithm(self):
        T = self.T_init
        profit0 = Solution.calculate_max_profit(self.sol_init)
        sol_0 = self.sol_init
        #Temp_iter = 1000  # Set number of temperature iterations
        #for i in range(Temp_iter):
        i = 0
        while T > 0.001:
            print(i, 'profit = ', profit0)

            T = self.get_temp(i)
            self.T_list.append(T)
            i += 1
            for j in range(100):  # Set

                # Delete one event
                r1 = np.random.randint(0, len(self.sol_init))
                depo = sol_0
                del_event = sol_0[r1]
                # Replace event
                replacement_time = del_event.time_start

                # Check possible destination and replace it in solution
                possible_events = self.check_accessibility(replacement_time, del_event)
                r2 = np.random.randint(0, len(possible_events))
                t1 = np.random.randint(1, possible_events[r2][2] - possible_events[r2][1] + 1)
                if t1 > del_event.time_stay:
                    t1 = del_event.time_stay
                sol_0[r1] = Event(possible_events[r2][0], possible_events[r2][1], t1)

                # Get the new profit
                profit1 = Solution.calculate_max_profit(sol_0)

                if profit1 > profit0:
                    # Accept new solution
                    profit0 = profit1
                else:
                    # Accept new (worse) solution with a given probability
                    acc = np.random.uniform()
                    if acc < np.exp(-(profit0 - profit1) / T):
                        # Accept the solution
                        profit0 = profit1
                    else:
                        # Do not accept the solution
                        sol_0 = depo

            self.acc_list.append(np.exp(-(profit0 - profit1) / T))
            self.profit_list.append(profit0)

            # Plot the result
        fig = plt.figure(figsize=(15, 5))
        #ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(131)
        ax3 = fig.add_subplot(132)
        ax4 = fig.add_subplot(133)

        # Profit plot
        ax2.axis([0, len(self.T_list), 0, 15000])
        #ax2.scatter(np.linspace(0, len(self.T_list), num=len(self.T_list), endpoint=False), self.profit_list, s=1.0, color='darkgreen')
        ax2.plot(np.linspace(0, len(self.T_list), num=len(self.T_list), endpoint=False), self.profit_list, linewidth=1.0,
                    color='darkgreen')
        # Acceptance plot
        ax3.axis([0, len(self.T_list), 0, 1])
        ax3.scatter(np.linspace(0, len(self.T_list), num=len(self.T_list), endpoint=False), self.acc_list, s=1.0, color='darkgreen')
        #ax3.plot(np.linspace(0, len(self.T_list), num=len(self.T_list), endpoint=False), self.acc_list, linewidth=1.0, color='darkgreen')

        # Temperature plot
        ax4.axis([0, len(self.T_list), 0, self.T_init])
        ax4.scatter(np.linspace(0, len(self.T_list), len(self.T_list), endpoint=False), self.T_list, s=1.0, color='darkgreen')
        '''
        ax1.clear()
        for first, second in zip(coords[:-1], coords[1:]):
            ax1.plot([first.x, second.x], [first.y, second.y], 'b')
        for c in coords:
            ax1.plot(c.x, c.y, 'ro')
        plt.pause(0.0001)
        '''
        plt.show()
        return sol_0


if __name__ == '__main__':
    # tests
    init_route = [Event("Krakow1", 1, 1), Event("Poznan1", 3, 4), Event("Szczecin1", 8, 2), Event("Wroclaw1", 11, 4), Event("Lodz2", 16, 5), Event("Wroclaw2", 21, 5), Event("Katowice2", 27, 4)]
    init_solution = Solution(init_route, 5000, SolMethod.GEO, EVENTSt)
    init_solution.sym_ann_algorithm()
    print(init_solution)



