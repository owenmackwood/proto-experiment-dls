import unittest

from template.scripts.initialization import main


class TestInitializationScript(unittest.TestCase):
    def test_main(self):
        num_initializations = 5
        self.assertIsNone(main(num_initializations))
