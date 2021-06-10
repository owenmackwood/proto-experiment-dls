import unittest

from template.scripts.minimal_experiment import main


class TestMinimalExperiment(unittest.TestCase):
    def test_experiment(self):
        self.assertIsNone(main())
