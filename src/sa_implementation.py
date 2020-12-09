import numpy as np
from src.data_prep import EVENTS, EVENTSt, FUEL_COST, NAMES
from enum import Enum
import matplotlib.pyplot as plt
import random


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

    @staticmethod
    def calculate_max_profit(solution):
        max_profit = 0
        for i in range(len(solution)):
            fuel_cost = 0
            if i < len(solution)-1:
                # print("miasto 1",solution[i].name[:-1])
                # print("miasto 2", solution[i+1].name[:-1])
                fuel_cost = FUEL_COST[solution[i].name[:-1]][solution[i+1].name[:-1]]
            max_profit += solution[i].calculate_profit() - solution[i].calculate_cost() - fuel_cost
        return round(max_profit, 2)

    @staticmethod
    def get_neighbour(solution):
        draw = random.choice(NAMES)
        i = random.randint(0, len(solution)-1)
        solution[i].name = draw
        return solution

    def sym_ann_algorithm(self):
        T = self.T_init
        current_solution = self.sol_init
        # print('Aktualne rozwiazanie: ', current_solution)
        current_profit = Solution.calculate_max_profit(current_solution)
        best_profit = current_profit
        best_solution = current_solution
        acc = 0
        i = 1
        while T > 0.1:
            print(i, 'profit = ', current_profit)
            i += 1
            for j in range(20):
                '''
                # Delete one event
                r1 = np.random.randint(0, len(self.sol_init))
                depo = sol_0
                del_event = sol_0[r1]
                # Replace event
                replacement_time = del_event.time_start

                neighbour = get_neighbour()
                # Check possible destination and replace it in solution
                possible_events = self.check_accessibility(replacement_time, del_event)
                r2 = np.random.randint(0, len(possible_events))
                t1 = np.random.randint(1, possible_events[r2][2] - possible_events[r2][1] + 1)
                
                
                if t1 > del_event.time_stay:
                    t1 = del_event.time_stay
                sol_0[r1] = Event(possible_events[r2][0], replacement_time, t1)
                '''
                # Get the new profit
                # profit1 = Solution.calculate_max_profit(sol_0)
                neighbour = Solution.get_neighbour(current_solution)
                neighbour_profit = Solution.calculate_max_profit(neighbour)
                if neighbour_profit > best_profit:
                    best_profit = neighbour_profit
                    best_solution = neighbour
                if neighbour_profit > current_profit:
                    # Accept new solution (better than previous best)
                    current_solution = neighbour
                    current_profit = neighbour_profit
                    acc = 1
                else:
                    acc = np.random.uniform(0, 1)
                    if acc < np.exp((neighbour_profit-current_profit)/T):
                        # Accept worse solution with probability acc
                        current_solution = neighbour
                        current_profit = neighbour_profit
                    else:
                        # Do not accept the solution
                        acc = 0
            T = self.get_temp(i)
            self.T_list.append(T)
            print('Acceptance: ', acc)
            self.acc_list.append(acc)
            self.profit_list.append(current_profit)

        # Plot the result
        fig = plt.figure(figsize=(15, 5))
        # ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(131)
        ax3 = fig.add_subplot(132)
        ax4 = fig.add_subplot(133)

        # Profit plot
        ax2.axis([0, len(self.T_list), -5000, 20000])
        ax2.scatter(np.linspace(0, len(self.T_list), num=len(self.T_list), endpoint=False), self.profit_list, s=1.0,
                    color='darkgreen')
        ax2.set_title('Profit graph')

        # Acceptance plot
        ax3.axis([0, len(self.T_list), -0.1, 1.1])
        ax3.scatter(np.linspace(0, len(self.T_list), num=len(self.T_list), endpoint=False), self.acc_list, s=1.0,
                    color='darkgreen')
        ax3.set_title('Acceptance graph')

        # Temperature plot
        ax4.axis([0, len(self.T_list), 0, self.T_init])
        ax4.scatter(np.linspace(0, len(self.T_list), len(self.T_list), endpoint=False), self.T_list, s=1.0,
                    color='darkgreen')
        ax4.set_title('Temperature graph')
        plt.show()
        '''
        ax1.clear()
        for first, second in zip(coords[:-1], coords[1:]):
            ax1.plot([first.x, second.x], [first.y, second.y], 'b')
        for c in coords:
            ax1.plot(c.x, c.y, 'ro')
        plt.pause(0.0001)
        '''
        print('Best solution is: ')
        for event in best_solution:
            print('{0}, {1}, {2}'.format(event.name, event.time_start, event.time_stay))
        print('Profit: ', best_profit)


def generate_init_route(num_events: int) -> list:
    init = []
    calendar = 31
    dividers = sorted(random.sample(range(1, calendar - num_events), num_events - 1))
    random_numbers = [a - b for a, b in zip(dividers + [calendar - num_events], [0] + dividers)]
    time_start = 1
    time_sum = 0
    name = random.sample(NAMES, num_events)
    for i in range(num_events):
        time_stay = random_numbers[i]
        event = Event(name[i], time_start, time_stay)
        time_sum += time_stay
        time_start = time_sum + 2 + i
        init.append(event)
    return init


if __name__ == '__main__':
    '''
    # init_route = [Event("Krakow1", 1, 1), Event("Poznan1", 3, 4), Event("Szczecin1", 8, 2), Event("Wroclaw1", 11, 4),
     #             Event("Lodz2", 16, 5), Event("Wroclaw2", 21, 5)]
    init_route = generate_init_route(5)
    init_solution = Solution(init_route, 2200, SolMethod.GEO, EVENTSt)
    init_solution.sym_ann_algorithm()
    '''
    init_route = generate_init_route(6)
    init_solution = Solution(init_route, 200, SolMethod.GEO, EVENTSt)
    init_solution.sym_ann_algorithm()
