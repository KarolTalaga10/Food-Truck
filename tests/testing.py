import unittest
from src.sa_implementation import Event


class TestEvent(unittest.TestCase):
    def test_create_instance(self):
        event = Event('Warszawa1', 10, 5)
        self.assertEqual(event.name, 'Warszawa1')
        self.assertEqual(event.time_start, 10)
        self.assertEqual(event.time_stay, 5)

    def test_calculate_cost(self):
        event1 = Event('Lodz1', 10, 2)
        event2 = Event('Lublin2', 23, 3)
        self.assertEqual(event1.calculate_cost(), 542.88)
        self.assertEqual(event2.calculate_cost(), 573.07)

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


if __name__ == '__main__':
    unittest.main()
