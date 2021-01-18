import unittest
from src.sa_implementation import Event, Solution


class TestEvent(unittest.TestCase):
    def test_create_instance(self):
        event = Event('Warszawa1', 10, 5)
        self.assertEqual(event.name, 'Warszawa1')
        self.assertEqual(event.time_start, 10)
        self.assertEqual(event.time_stay, 5)

    def test_calculate_cost(self):
        event1 = Event('Lodz1', 10, 2)
        event2 = Event('Lublin2', 23, 3)
        event3 = Event('Wroclaw2', 1, 2)
        event4 = Event('Katowice1', 4, 1)
        self.assertEqual(event1.calculate_cost(), 542.88)
        self.assertEqual(event2.calculate_cost(), 573.07)
        self.assertEqual(event3.calculate_cost(), 545.6)
        self.assertEqual(event4.calculate_cost(), 204)

    def test_calculate_profit(self):
        event1 = Event('Warszawa1', 16, 5)
        event2 = Event('Krakow1', 3, 2)
        event3 = Event('Poznan1', 8, 7)
        event4 = Event('Gdansk1', 2, 4)
        event5 = Event('Wroclaw1', 10, 3)
        event6 = Event('Bydgoszcz1', 3, 9)
        self.assertEqual(event1.calculate_profit(), 0)
        self.assertEqual(event2.calculate_profit(), 1519.68)
        self.assertEqual(event3.calculate_profit(), 588.70)
        self.assertEqual(event4.calculate_profit(), 0)
        self.assertEqual(event5.calculate_profit(), 1592.34)
        self.assertEqual(event6.calculate_profit(), 2195.61)

    def test_max_profit(self):
        sol1 = [Event('Szczecin1', 1, 2), Event('Poznan1', 4, 2)]
        profit1 = Solution.calculate_max_profit(sol1)
        # sol2 = [Event('Krakow1', 1, 2), Event('Warszawa1', 4, 6), Event('Wroclaw1', 11, 5), Event('Gdansk2', 17, 4), Event('Poznan2', 22, 6), Event('Szczecin2', 29, 2)]
        # profit2 = Solution.calculate_max_profit(sol2)
        self.assertEqual(profit1, 79.36)
        # self.assertEqual(profit2, -373.8)


if __name__ == '__main__':
    unittest.main()
