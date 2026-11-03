from datetime import date, timedelta
import unittest

from src.evaluation import chronological_date_splits


class ChronologicalDateSplitTests(unittest.TestCase):
    def test_split_is_deterministic_ordered_and_keeps_duplicate_dates_together(self):
        dates = [date(2025, 1, 1) + timedelta(days=offset) for offset in range(10)]
        supplied = [dates[7], dates[0], dates[3], dates[3], *dates[1:]]

        split = chronological_date_splits(supplied)

        self.assertEqual(split, chronological_date_splits(reversed(supplied)))
        self.assertEqual(split["train"], tuple(dates[:6]))
        self.assertEqual(split["validation"], tuple(dates[6:8]))
        self.assertEqual(split["test"], tuple(dates[8:]))
        self.assertEqual(len(set().union(*(set(values) for values in split.values()))), 10)
        self.assertEqual(sum(map(len, split.values())), 10)

    def test_invalid_fractions_are_rejected(self):
        dates = [date(2025, 1, 1) + timedelta(days=offset) for offset in range(10)]
        with self.assertRaises(ValueError):
            chronological_date_splits(dates, train_fraction=0.8, validation_fraction=0.2)

    def test_too_few_dates_are_rejected(self):
        with self.assertRaises(ValueError):
            chronological_date_splits([date(2025, 1, 1), date(2025, 1, 1)])


if __name__ == "__main__":
    unittest.main()
