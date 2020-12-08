import unittest
from src.sa_implementation import Event


class TestEvent(unittest.TestCase):
    def test_create_instance(self):
        event = Event('Warszawa', 10, 5)
        self.assertEqual(event.name, 'Warszawa')
        self.assertEqual(event.time_start, 10)
        self.assertEqual(event.time_stay, 5)

    def test_calculate_cost(self):
        event1 = Event('Lodz1', 10, 2)
        event2 = Event('Lublin2', 23, 3)
        self.assertEqual(event1.calculate_cost(), 542.88)
        self.assertEqual(event2.calculate_cost(), 573.07)

    def test_calculate_profit(self):
        event1 = Event('Warszawa', 16, 5)
        event2 = Event('Krakow1', 3, 2)
        event3 = Event('Poznan1', 8, 7)
        self.assertEqual(event1.calculate_profit(), 0)
        self.assertEqual(event2.calculate_profit(), 1519.68)
        self.assertEqual(event3.calculate_profit(), 588.70)


if __name__ == '__main__':
    unittest.main()
